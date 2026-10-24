"""E15: E3(b) forecasting with a mismatched table at small training lengths n."""
import time

import numpy as np

from common import config_hash, file_sha, log_run
from e3_synth import CASES, REPS, forecast_mse, signal

NS = [5, 10, 20, 30, 50]


def main():
    sha = {f: file_sha(f) for f in ["e15_synth.py", "e3_synth.py", "common.py"]}
    cfg = {"exp": "E15", "cases": CASES, "ns": NS, "reps": REPS}
    t0 = time.time()
    rows = []
    for P, W in CASES:
        L, H = W, 2 * W
        n_va, n_te = 20 * W, 50 * W
        for n in NS:
            res = {"rcf_sa": [], "oracle": [], "linear": []}
            for rep in range(REPS):
                rng = np.random.default_rng(10_000 + rep)
                n_tr = n * W
                x, m, _ = signal(n_tr + n_va + n_te, P, rng)
                ph = np.arange(len(x)) % W
                q = np.bincount(ph[:n_tr], weights=x[:n_tr], minlength=W) / np.bincount(ph[:n_tr], minlength=W)
                res["rcf_sa"].append(forecast_mse(x, q[ph], n_tr, n_va, L, H))
                res["oracle"].append(forecast_mse(x, m, n_tr, n_va, L, H))
                res["linear"].append(forecast_mse(x, None, n_tr, n_va, L, H))
            row = {"P": P, "W": W, "n": n, "L": L, "H": H,
                   **{k + "_mse_mean": float(np.mean(v)) for k, v in res.items()},
                   **{k + "_mse_std": float(np.std(v)) for k, v in res.items()}}
            rows.append(row)
            print(row, flush=True)
    log_run({**cfg, "rows": rows, "config_hash": config_hash(cfg), "code_sha256": sha, "status": "ok",
             "seconds": round(time.time() - t0, 2)})


if __name__ == "__main__":
    main()
