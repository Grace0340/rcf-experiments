"""Causal resampling to a grid of W' points per period around a closed-form RCF forecaster.

The hourly series is interpolated onto tau_k = k*s, s = P_res/W', with 4-point cubic Lagrange
interpolation (exact sample when tau_k is an integer). A grid point is usable at origin t0 only if
every hourly sample in its stencil is <= t0 (cumulative maximum, so usable points are a prefix).
The forecaster predicts H' grid points; forecasts are mapped back to the hourly targets with the
same interpolator, using the known past grid values plus the forecast grid values.
"""
import math

import numpy as np
import torch

from closed_form import DEV, DT, LAMBDAS, Moments, intercept, phase_table, ridge

ORIGIN_CHUNK = 256


def lagrange_weights(f):
    """Weights for nodes -1, 0, 1, 2 at fractional offset f in [0, 1)."""
    return torch.stack([-f * (f - 1) * (f - 2) / 6, (f + 1) * (f - 1) * (f - 2) / 2,
                        -(f + 1) * f * (f - 2) / 2, (f + 1) * f * (f - 1) / 6], -1)


def split_pos(pos):
    near = torch.round(pos)
    is_int = (pos - near).abs() < 1e-9
    fl = torch.where(is_int, near, torch.floor(pos))
    return fl.long(), torch.where(is_int, torch.zeros_like(pos), pos - fl), is_int


def to_grid(x, s):
    T = x.shape[0]
    K = int(math.floor((T - 1) / s + 1e-9)) + 1
    tau = torch.arange(K, dtype=DT, device=DEV) * s
    i0, f, is_int = split_pos(tau)
    ok = is_int | ((i0 - 1 >= 0) & (i0 + 2 <= T - 1))
    idx = (i0[:, None] + torch.arange(-1, 3, device=DEV)[None]).clamp(0, T - 1)
    z = (lagrange_weights(f)[..., None] * x[idx]).sum(1)
    z[is_int] = x[i0[is_int]]
    z[~ok] = float("nan")
    raw_av = torch.where(is_int, i0, i0 + 2)
    raw_av = torch.where(ok, raw_av, torch.full_like(raw_av, T + 10 ** 6))
    head_bad = torch.nonzero(~ok & (i0 < 1))
    first_ok = int(head_bad.max()) + 1 if len(head_bad) else 0
    raw_av[:first_ok] = -1  # never used: windows must start at or after first_ok
    cav = torch.cummax(raw_av, 0).values
    return z, cav, first_ok


class ResampledRCF:
    def __init__(self, data, b1, b2, L, H, P_res, Wp):
        self.x = torch.tensor(data, dtype=DT, device=DEV)
        self.b1, self.b2, self.L, self.H = b1, b2, L, H
        self.s = P_res / Wp
        self.Wp = Wp
        self.Lp = math.ceil(L / self.s - 1e-9)
        self.Hp = math.ceil((H + 3) / self.s - 1e-9) + 2
        self.z, self.cav, self.k0 = to_grid(self.x, self.s)
        k_end = int(torch.searchsorted(self.cav, torch.tensor(b2[0] - 1, device=DEV), right=True))
        self.q = phase_table(self.z, self.k0, k_end, Wp)
        K = self.z.shape[0]
        self.zr = self.z - self.q[torch.arange(K, device=DEV) % Wp]
        self.mom = Moments(self.Lp, self.Hp)
        self.mom.add(self.zr[self.k0:k_end], self.k0)

    def origins(self, j):
        b1, b2, L, H = self.b1, self.b2, self.L, self.H
        return torch.arange(b1[j] + L - 1, b2[j] - H, device=DEV)

    def evaluate(self, t0s, A, b):
        H, Hp, Lp, s = self.H, self.Hp, self.Lp, self.s
        sse = sae = 0.0
        cnt = 0
        for o0 in range(0, len(t0s), ORIGIN_CHUNK):
            t0 = t0s[o0:o0 + ORIGIN_CHUNK]
            kl = torch.searchsorted(self.cav, t0, right=True) - 1
            idx_in = kl[:, None] + torch.arange(-Lp + 1, 1, device=DEV)[None]
            assert int(idx_in.min()) >= self.k0
            xin = self.zr[idx_in]                                    # (o, Lp, C)
            out = torch.einsum("olc,lh->ohc", xin, A) + b[None, :, None]
            fut = kl[:, None] + torch.arange(1, Hp + 1, device=DEV)[None]
            out = out + self.q[fut % self.Wp]
            past = self.z[kl[:, None] + torch.arange(-3, 1, device=DEV)[None]]
            seq = torch.cat([past, out], 1)                          # local j <-> grid kl - 3 + j
            u = t0[:, None] + torch.arange(1, H + 1, device=DEV)[None]
            fl, f, _ = split_pos(u.to(DT) / s)
            li = fl - (kl[:, None] - 3)
            g = li[..., None] + torch.arange(-1, 3, device=DEV)      # (o, H, 4)
            assert int(g.min()) >= 0 and int(g.max()) < seq.shape[1]
            C = seq.shape[2]
            vals = torch.gather(seq, 1, g.reshape(len(t0), -1, 1).expand(-1, -1, C)).reshape(len(t0), H, 4, C)
            yhat = (lagrange_weights(f)[..., None] * vals).sum(2)
            err = self.x[u] - yhat
            sse += float((err ** 2).sum())
            sae += float(err.abs().sum())
            cnt += err.numel()
        return sse / cnt, sae / cnt

    def fit(self, lambdas=LAMBDAS):
        cxx, cxy, means = self.mom.covariances()
        val_o, test_o = self.origins(1), self.origins(2)
        by_lam = []
        for lam in lambdas:
            A = ridge(cxx, cxy, lam)
            b = intercept(A, means, False)["b"]
            by_lam.append((lam, self.evaluate(val_o, A, b), A, b))
        lam, val, A, b = min(by_lam, key=lambda r: r[1][0] if np.isfinite(r[1][0]) else float("inf"))
        test = self.evaluate(test_o, A, b)
        return {
            "lambda": lam, "val_mse": val[0], "val_mae": val[1], "test_mse": test[0], "test_mae": test[1],
            "val_mse_by_lambda": {str(r[0]): r[1][0] for r in by_lam},
            "n_train_windows": self.mom.n, "s": self.s, "Lp": self.Lp, "Hp": self.Hp,
            "n_test_origins": len(test_o),
        }
