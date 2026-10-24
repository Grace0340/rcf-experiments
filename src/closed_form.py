"""Closed-form channel-independent linear forecasters on CycleNet's splits.

Methods
  linear : y = A^T x + b
  rcf    : per-phase mean table Q (training segment), ridge on residual windows,
           y = A^T (x - Q_past) + b + Q_fut          (errors are identical in residual space)
  free   : y = A^T x + D[c, phase]                   (unconstrained phase intercept, solved by FWL)
Ridge penalises A only; intercepts are unpenalised. lambda is chosen on the validation split.
"""
import argparse
import math
import time

import numpy as np
import torch

from common import PERIOD, config_hash, file_sha, load, log_run

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64
LAMBDAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
CHUNK = 16


def windows(seg, L, H):
    """(n, C) -> view (C, n - L - H + 1, L + H)."""
    return seg.T.unfold(1, L + H, 1)


def phase_table(series, start, end, W, phase_offset=0):
    """Mean of series[t] over t in [start, end) grouped by (t + phase_offset) mod W."""
    ph = (torch.arange(start, end, device=series.device) + phase_offset) % W
    cnt = torch.bincount(ph, minlength=W).to(series.dtype)
    q = torch.zeros(W, series.shape[1], dtype=series.dtype, device=series.device)
    return q.index_add_(0, ph, series[start:end]) / cnt[:, None]


class Moments:
    """Sufficient statistics of (window, target) pairs pooled over channels."""

    def __init__(self, L, H, C=None, W=None):
        def z(*s):
            return torch.zeros(*s, dtype=DT, device=DEV)

        self.L, self.H, self.W = L, H, W
        self.n = 0
        self.sx, self.sy = z(L), z(H)
        self.Sxx, self.Sxy = z(L, L), z(L, H)
        if W:
            self.gx, self.gy, self.gn = z(C, W, L), z(C, W, H), z(C, W)

    def add(self, seg, start, valid=None):
        L, H = self.L, self.H
        for c0 in range(0, seg.shape[1], CHUNK):
            v = windows(seg[:, c0:c0 + CHUNK], L, H)
            if valid is not None:
                v = v[:, valid]
            X, Y = v[..., :L], v[..., L:]
            Xf, Yf = X.reshape(-1, L), Y.reshape(-1, H)
            self.n += Xf.shape[0]
            self.sx += Xf.sum(0)
            self.sy += Yf.sum(0)
            self.Sxx += Xf.T @ Xf
            self.Sxy += Xf.T @ Yf
            if self.W:
                cc, nw = v.shape[0], v.shape[1]
                ph = (torch.arange(nw, device=DEV) + start) % self.W
                self.gx[c0:c0 + cc].index_add_(1, ph, X)
                self.gy[c0:c0 + cc].index_add_(1, ph, Y)
                self.gn[c0:c0 + cc] += torch.bincount(ph, minlength=self.W).to(DT)

    def covariances(self):
        n = self.n
        if not self.W:
            mx, my = self.sx / n, self.sy / n
            return self.Sxx / n - torch.outer(mx, mx), self.Sxy / n - torch.outer(mx, my), (mx, my)
        mx = self.gx / self.gn[..., None]
        my = self.gy / self.gn[..., None]
        cxx = (self.Sxx - torch.einsum("cwl,cwm->lm", mx, self.gx)) / n
        cxy = (self.Sxy - torch.einsum("cwl,cwh->lh", mx, self.gy)) / n
        return cxx, cxy, (mx, my)


def ridge(cxx, cxy, lam):
    a = cxx + lam * torch.eye(cxx.shape[0], dtype=DT, device=DEV)
    try:
        return torch.linalg.solve(a, cxy)
    except RuntimeError:
        return torch.linalg.pinv(a) @ cxy


def intercept(A, means, free):
    mx, my = means
    if free:
        return {"D": my - torch.einsum("cwl,lh->cwh", mx, A)}
    return {"b": my - A.T @ mx}


def evaluate(seg, start, L, H, A, b=None, D=None, W=None):
    sse = sae = 0.0
    cnt = 0
    for c0 in range(0, seg.shape[1], CHUNK):
        v = windows(seg[:, c0:c0 + CHUNK], L, H)
        pred = v[..., :L] @ A
        if D is None:
            pred = pred + b
        else:
            ph = (torch.arange(v.shape[1], device=DEV) + start) % W
            pred = pred + D[c0:c0 + v.shape[0]][:, ph, :]
        err = v[..., L:] - pred
        sse += float((err ** 2).sum())
        sae += float(err.abs().sum())
        cnt += err.numel()
    return sse / cnt, sae / cnt


def fit_closed(data, b1, b2, L, H, method, W=None, lambdas=LAMBDAS):
    x = torch.tensor(data, dtype=DT, device=DEV)
    if method == "rcf":
        q = phase_table(x, b1[0], b2[0], W)
        x = x - q[torch.arange(x.shape[0], device=DEV) % W]
    free = method == "free"
    mom = Moments(L, H, x.shape[1], W if free else None)
    mom.add(x[b1[0]:b2[0]], b1[0])
    cxx, cxy, means = mom.covariances()
    by_lam = []
    for lam in lambdas:
        A = ridge(cxx, cxy, lam)
        kw = intercept(A, means, free)
        val = evaluate(x[b1[1]:b2[1]], b1[1], L, H, A, W=W, **kw)
        by_lam.append((lam, val, A, kw))
    lam, val, A, kw = min(by_lam, key=lambda r: r[1][0] if math.isfinite(r[1][0]) else float("inf"))
    test = evaluate(x[b1[2]:b2[2]], b1[2], L, H, A, W=W, **kw)
    return {
        "lambda": lam,
        "val_mse": val[0], "val_mae": val[1],
        "test_mse": test[0], "test_mae": test[1],
        "val_mse_by_lambda": {str(r[0]): r[1][0] for r in by_lam},
        "n_train_windows": mom.n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--H", type=int, nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["linear", "rcf", "free"])
    ap.add_argument("--W", type=int, nargs="*", default=None, help="phase-table lengths (default: dataset period)")
    args = ap.parse_args()
    shas = {f: file_sha(f) for f in ["closed_form.py", "common.py"]}
    for ds in args.datasets:
        data, b1, b2 = load(ds, args.L)
        for H in args.H:
            for method in args.methods:
                Ws = [None] if method == "linear" else (args.W or [PERIOD[ds]])
                for W in Ws:
                    cfg = {"exp": args.exp, "dataset": ds, "method": method, "L": args.L, "H": H, "W": W,
                           "lambdas": LAMBDAS, "revin": 0}
                    t0 = time.time()
                    try:
                        out = fit_closed(data, b1, b2, args.L, H, method, W)
                        status = "ok"
                    except Exception as e:  # keep failed runs in the log
                        out, status = {"error": repr(e)}, "failed"
                    rec = dict(cfg, **out, status=status, source="closed_form", config_hash=config_hash(cfg),
                               seconds=round(time.time() - t0, 2), code_sha256=shas)
                    log_run(rec)
                    print(ds, H, method, W, status, {k: rec.get(k) for k in ["lambda", "val_mse", "test_mse", "test_mae"]},
                          flush=True)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
