"""E5: learned Q of official CycleNet (E1 checkpoints) compared with the closed-form table m.

Official indexing: Q[k] multiplies absolute phase (k - L) mod W, so m is rolled by L before comparison."""
import json

import torch

from closed_form import DEV, DT, phase_table
from common import PERIOD, ROOT, file_sha, load, log_run


def intercept_operator(B, L, W):
    H = B.shape[0]
    i = torch.arange(W, device=DEV)
    P = torch.zeros(W, L, W, dtype=DT, device=DEV)
    P[i[:, None], torch.arange(L, device=DEV)[None], (i[:, None] + torch.arange(L, device=DEV)[None]) % W] = 1
    F = torch.zeros(W, H, W, dtype=DT, device=DEV)
    F[i[:, None], torch.arange(H, device=DEV)[None], (i[:, None] + L + torch.arange(H, device=DEV)[None]) % W] = 1
    M = F - torch.einsum("hl,ilw->ihw", B, P)
    return (M - M.mean(0, keepdim=True)).reshape(W * H, W)


def main():
    rows = [json.loads(l) for l in open(f"{ROOT}/results/runs.jsonl")]
    rows = [r for r in rows if r.get("exp") == "E1" and r.get("source") == "official" and r.get("status") == "ok"]
    sha = {f: file_sha(f) for f in ["e5_nullspace.py", "closed_form.py", "common.py"]}
    cache = {}
    for r in sorted(rows, key=lambda r: (r["dataset"], r["H"], r["seed"])):
        ds, H, L, W = r["dataset"], r["H"], r["L"], r["W"]
        if ds not in cache:
            data, b1, b2 = load(ds, L)
            x = torch.tensor(data, dtype=DT, device=DEV)
            cache[ds] = phase_table(x, b1[0], b2[0], W)
        m = torch.roll(cache[ds], shifts=L, dims=0)
        sd = torch.load(r["checkpoint"], map_location=DEV)
        Q = sd["cycleQueue.data"].to(DT)
        B = sd["model.weight"].to(DT)
        G = intercept_operator(B, L, W)
        s = torch.linalg.svdvals(G)
        mc, Qc = m - m.mean(0), Q - Q.mean(0)
        dQ = Qc - mc
        rel_q = float(dQ.norm() / mc.norm())
        rel_d = float((G @ dQ).norm() / (G @ mc).norm())
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        weak = Vh[S < 0.1 * S[0]]
        frac_weak = float(((weak @ dQ) ** 2).sum() / (dQ ** 2).sum()) if len(weak) else 0.0
        weak_ratio = float((weak @ Qc).norm() / (weak @ mc).norm()) if len(weak) and (weak @ mc).norm() > 0 else None
        corr = float((Qc * mc).sum() / (Qc.norm() * mc.norm()))
        rec = {"exp": "E5", "dataset": ds, "H": H, "L": L, "W": W, "seed": r["seed"], "checkpoint": r["checkpoint"],
               "rel_err_Q_vs_m": rel_q, "rel_err_intercepts": rel_d, "corr_Q_m": corr,
               "frac_diff_in_weak_dirs": frac_weak, "n_weak_dirs": len(weak), "weak_component_ratio_Q_over_m": weak_ratio,
               "version": 2, "sv_max": float(S[0]),
               "sv_min": float(S[-1]), "code_sha256": sha, "status": "ok", "source": "analysis"}
        log_run(rec)
        print(ds, H, r["seed"], {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()
                                 if k in ("rel_err_Q_vs_m", "rel_err_intercepts", "frac_diff_in_weak_dirs", "n_weak_dirs",
                                          "weak_component_ratio_Q_over_m")},
              flush=True)


if __name__ == "__main__":
    main()
