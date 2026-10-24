"""E8: tides with short look-back windows. M1 Linear, M2 RCF (integer W),
M3 RCF + resampling; identical to E4 except L in {24, 48, 96} and H in {24, 96}."""
import shutil
import time

import torch

from closed_form import LAMBDAS, fit_closed
from common import DATA, ROOT, config_hash, file_sha, load, log_run
from e4_tides import M2_W, M3_VARIANTS
from resample import ResampledRCF

LS = [24, 48, 96]
HS = [24, 96]


def main():
    import json
    st = json.load(open(f"{ROOT}/data/noaa/stations.json"))
    shas = {f: file_sha(f) for f in ["e8_tides_short.py", "e4_tides.py", "closed_form.py", "resample.py", "common.py"]}
    for sid in st["chosen"]:
        shutil.copy(f"{ROOT}/data/noaa/{sid}.csv", f"{DATA}/noaa_{sid}.csv")
        ds = f"Tide{sid}"
        for L in LS:
            data, b1, b2 = load(ds, L)
            for H in HS:
                runs = [("M1", "linear", None, None)] + [("M2", "rcf", W, None) for W in M2_W] \
                    + [("M3", "resampled", Wp, Pr) for Pr, Wp in M3_VARIANTS]
                for tag, method, W, P_res in runs:
                    cfg = {"exp": "E8", "dataset": ds, "method": method, "tag": tag, "L": L, "H": H, "W": W,
                           "P_res": P_res, "lambdas": LAMBDAS, "revin": 0}
                    t0 = time.time()
                    try:
                        if method == "resampled":
                            out = ResampledRCF(data, b1, b2, L, H, P_res, W).fit()
                        else:
                            out = fit_closed(data, b1, b2, L, H, method, W)
                        status = "ok"
                    except Exception as e:
                        out, status = {"error": repr(e)}, "failed"
                    rec = dict(cfg, **out, status=status, source="closed_form", config_hash=config_hash(cfg),
                               seconds=round(time.time() - t0, 2), code_sha256=shas)
                    log_run(rec)
                    print(ds, L, H, tag, W, P_res, status, {k: rec.get(k) for k in ["lambda", "val_mse", "test_mse"]},
                          flush=True)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
