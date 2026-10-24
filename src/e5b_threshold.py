"""E5b: sensitivity of the E5 weak-direction statistics to the threshold tau."""
import json

import torch

from closed_form import DEV, DT, phase_table
from common import ROOT, file_sha, load, log_run
from e5_nullspace import intercept_operator

TAUS = [0.02, 0.05, 0.1, 0.2, 0.3]


def main():
    rows = [json.loads(l) for l in open(f"{ROOT}/results/runs.jsonl")]
    rows = [r for r in rows if r.get("exp") == "E1" and r.get("source") == "official" and r.get("status") == "ok"]
    sha = {f: file_sha(f) for f in ["e5b_threshold.py", "e5_nullspace.py", "closed_form.py", "common.py"]}
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
        G = intercept_operator(sd["model.weight"].to(DT), L, W)
        _, S, Vh = torch.linalg.svd(G, full_matrices=False)
        mc, Qc = m - m.mean(0), Q - Q.mean(0)
        dQ = Qc - mc
        stats = {}
        for tau in TAUS:
            weak = Vh[S < tau * S[0]]
            frac = float(((weak @ dQ) ** 2).sum() / (dQ ** 2).sum()) if len(weak) else 0.0
            ratio = float((weak @ Qc).norm() / (weak @ mc).norm()) if len(weak) and (weak @ mc).norm() > 0 else None
            stats[str(tau)] = {"n_weak": len(weak), "frac_diff_in_weak_dirs": frac, "weak_component_ratio_Q_over_m": ratio}
        log_run({"exp": "E5b", "dataset": ds, "H": H, "L": L, "W": W, "seed": r["seed"], "checkpoint": r["checkpoint"],
                 "by_tau": stats, "code_sha256": sha, "status": "ok", "source": "analysis"})
        print(ds, H, r["seed"], {t: (v["n_weak"], round(v["frac_diff_in_weak_dirs"], 4)) for t, v in stats.items()},
              flush=True)


if __name__ == "__main__":
    main()
