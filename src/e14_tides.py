"""E14: replicate E4 (L = 336, M1-M5), E8 (L in {24, 48, 96}, M1-M3) and the
phase-table energy diagnostic on the reserve stations. Methods and code paths are those of E4/E8, unchanged."""
import json
import math
import shutil
import time

import numpy as np
import torch

from closed_form import DEV, DT, LAMBDAS, fit_closed, phase_table
from common import DATA, ROOT, config_hash, file_sha, load, log_run
from e4_tides import M2_W, M3_VARIANTS, harmonic_fit, seasonal_naive
from resample import ResampledRCF, to_grid


def run_one(cfg, fn, sha, extra=None):
    t0 = time.time()
    try:
        out, status = fn(), "ok"
    except Exception as e:
        out, status = {"error": repr(e)}, "failed"
    rec = dict(cfg, **out, **(extra or {}), status=status, source="closed_form", config_hash=config_hash(cfg),
               seconds=round(time.time() - t0, 2), code_sha256=sha)
    log_run(rec)
    print(cfg["dataset"], cfg["L"], cfg["H"], cfg["tag"], cfg["W"], cfg["P_res"], status,
          {k: rec.get(k) for k in ["val_mse", "test_mse"]}, flush=True)
    torch.cuda.empty_cache()


def main():
    st = json.load(open(f"{ROOT}/data/noaa/stations_extra.json"))
    sha = {f: file_sha(f) for f in ["e14_tides.py", "e4_tides.py", "closed_form.py", "resample.py", "common.py"]}
    for sid in st["chosen"]:
        shutil.copy(f"{ROOT}/data/noaa/{sid}.csv", f"{DATA}/noaa_{sid}.csv")
        ds = f"Tide{sid}"
        data, b1, b2 = load(ds, 336)
        raw = np.loadtxt(f"{DATA}/noaa_{sid}.csv", delimiter=",", skiprows=1, usecols=1)
        sd = float(raw[b1[0]:b2[0]].std())
        hfit = harmonic_fit(data[:, 0], b2)[:, None]

        x = torch.tensor(data, dtype=DT, device=DEV)
        diag = {"exp": "E14", "part": "energy", "dataset": ds, "status": "ok", "code_sha256": sha}
        for P_res, Wp in M3_VARIANTS:
            z, cav, k0 = to_grid(x, P_res / Wp)
            k_end = int(torch.searchsorted(cav, torch.tensor(b2[0] - 1, device=DEV), right=True))
            q = phase_table(z, k0, k_end, Wp)
            diag[f"phase_table_energy_res_{Wp}"] = float(((q - q.mean(0)) ** 2).mean())
        for W in (12, 24, 25, 50):
            q = phase_table(x, b1[0], b2[0], W)
            diag[f"phase_table_energy_int_{W}"] = float(((q - q.mean(0)) ** 2).mean())
        log_run(diag)
        print(diag, flush=True)

        for H in [24, 96, 336]:
            for tag, method, W, P_res in [("M1", "linear", None, None)] + [("M2", "rcf", w, None) for w in M2_W] \
                    + [("M3", "resampled", wp, pr) for pr, wp in M3_VARIANTS] \
                    + [("M4", "harmonic", None, None), ("M5", "seasonal_naive", None, None)]:
                cfg = {"exp": "E14", "part": "E4", "dataset": ds, "method": method, "tag": tag, "L": 336, "H": H,
                       "W": W, "P_res": P_res, "lambdas": LAMBDAS, "revin": 0}

                def fn(method=method, W=W, P_res=P_res, H=H):
                    if method in ("linear", "rcf"):
                        out = fit_closed(data, b1, b2, 336, H, method, W)
                    elif method == "resampled":
                        out = ResampledRCF(data, b1, b2, 336, H, P_res, W).fit()
                    elif method == "harmonic":
                        out = fit_closed(data - hfit, b1, b2, 336, H, "linear")
                    else:
                        vm, va, _ = seasonal_naive(data, b1, b2, 1, H)
                        tm, ta, kmax = seasonal_naive(data, b1, b2, 2, H)
                        out = {"val_mse": vm, "val_mae": va, "test_mse": tm, "test_mae": ta, "k_max": kmax}
                    out["test_rmse_m"] = math.sqrt(out["test_mse"]) * sd
                    return out
                run_one(cfg, fn, sha, {"train_sd_m": sd})

        for L in [24, 48, 96]:
            dL, b1L, b2L = load(ds, L)
            for H in [24, 96]:
                for tag, method, W, P_res in [("M1", "linear", None, None)] + [("M2", "rcf", w, None) for w in M2_W] \
                        + [("M3", "resampled", wp, pr) for pr, wp in M3_VARIANTS]:
                    cfg = {"exp": "E14", "part": "E8", "dataset": ds, "method": method, "tag": tag, "L": L, "H": H,
                           "W": W, "P_res": P_res, "lambdas": LAMBDAS, "revin": 0}

                    def fn(method=method, W=W, P_res=P_res, H=H, L=L):
                        if method == "resampled":
                            return ResampledRCF(dL, b1L, b2L, L, H, P_res, W).fit()
                        return fit_closed(dL, b1L, b2L, L, H, method, W)
                    run_one(cfg, fn, sha)


if __name__ == "__main__":
    main()
