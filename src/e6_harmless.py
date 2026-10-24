"""E6: with P_res = W' = P (step 1) the resampling pipeline must reproduce direct RCF-SA."""
import time

from closed_form import fit_closed
from common import PERIOD, config_hash, file_sha, load, log_run
from resample import ResampledRCF

L, H = 336, 96


def main():
    sha = {f: file_sha(f) for f in ["e6_harmless.py", "resample.py", "closed_form.py", "common.py"]}
    for ds in ["ETTh1", "Electricity"]:
        P = PERIOD[ds]
        data, b1, b2 = load(ds, L)
        t0 = time.time()
        direct = fit_closed(data, b1, b2, L, H, "rcf", P)
        res = ResampledRCF(data, b1, b2, L, H, float(P), P).fit()
        rel = abs(res["test_mse"] - direct["test_mse"]) / direct["test_mse"]
        cfg = {"exp": "E6", "dataset": ds, "L": L, "H": H, "P_res": P, "Wp": P}
        log_run({**cfg, "direct": direct, "resampled": res, "rel_diff_test_mse": rel, "pass": rel < 0.005,
                 "config_hash": config_hash(cfg), "code_sha256": sha, "seconds": round(time.time() - t0, 2),
                 "source": "e6"})
        print(ds, "direct", direct["test_mse"], direct["lambda"], "resampled", res["test_mse"], res["lambda"],
              "rel", rel, flush=True)


if __name__ == "__main__":
    main()
