"""E4: tides. M1 Linear, M2 RCF (integer W), M3 RCF + resampling,
M4 harmonic regression + linear residual, M5 fractional-lag seasonal naive. All closed form, L = 336.
All methods share the test origins t0 = b1[2]+L-1 .. b2[2]-H-1 and use only samples <= t0."""
import json
import math
import shutil
import sys
import time

import numpy as np
import torch

from closed_form import DEV, DT, LAMBDAS, fit_closed
from common import DATA, ROOT, config_hash, file_sha, load, log_run
from resample import ResampledRCF, lagrange_weights, split_pos

L = 336
HS = [24, 96, 336]
M2_W = [12, 25, 50]
M3_VARIANTS = [(12.4206, 13), (24.8412, 25)]
P_LUNAR = 24.8412
# Constituent speeds in degrees per mean solar hour (Schureman / NOAA CO-OPS).
SPEED = {"M2": 28.9841042, "S2": 30.0, "N2": 28.4397295, "K2": 30.0821373, "K1": 15.0410686,
         "O1": 13.9430356, "P1": 14.9589314, "Q1": 13.3986609, "M4": 57.9682084, "MS4": 58.9841042,
         "M6": 86.9523127, "Sa": 0.0410686, "Ssa": 0.0821373}


def harmonic_fit(x, b2):
    """LS fit of constant + cos/sin of the 13 constituents on the training segment; returns fit on all t."""
    t = np.arange(x.shape[0], dtype=np.float64)
    w = np.deg2rad(np.array(list(SPEED.values())))
    X = np.column_stack([np.ones_like(t)] + [f(wi * t) for wi in w for f in (np.cos, np.sin)])
    coef, *_ = np.linalg.lstsq(X[:b2[0]], x[:b2[0]], rcond=None)
    return X @ coef


def seasonal_naive(x, b1, b2, j, H):
    """y(t0+h) = x(t0+h-k_h P) by cubic Lagrange, k_h the smallest k with all stencil samples <= t0."""
    xt = torch.tensor(x, dtype=DT, device=DEV)
    t0 = torch.arange(b1[j] + L - 1, b2[j] - H, device=DEV)
    h = torch.arange(1, H + 1, dtype=DT, device=DEV)
    k = torch.ones_like(h)
    while True:
        fl, f, is_int = split_pos(h - k * P_LUNAR)
        need = torch.where(is_int, fl, fl + 2)
        bad = need > 0
        if not bad.any():
            break
        k = torch.where(bad, k + 1, k)
    g = t0[:, None, None] + fl[None, :, None] + torch.arange(-1, 3, device=DEV)[None, None]
    yhat = (lagrange_weights(f)[None, :, :, None] * xt[g]).sum(2)
    err = xt[t0[:, None] + torch.arange(1, H + 1, device=DEV)[None]] - yhat
    return float((err ** 2).mean()), float(err.abs().mean()), k.max().item()


def main():
    tags = sys.argv[1:] or ["M1", "M2", "M3", "M4", "M5"]
    st = json.load(open(f"{ROOT}/data/noaa/stations.json"))
    shas = {f: file_sha(f) for f in ["e4_tides.py", "closed_form.py", "resample.py", "common.py"]}
    for sid in st["chosen"]:
        shutil.copy(f"{ROOT}/data/noaa/{sid}.csv", f"{DATA}/noaa_{sid}.csv")
        ds = f"Tide{sid}"
        data, b1, b2 = load(ds, L)
        raw = np.loadtxt(f"{DATA}/noaa_{sid}.csv", delimiter=",", skiprows=1, usecols=1)
        sd = float(raw[b1[0]:b2[0]].std())
        hfit = harmonic_fit(data[:, 0], b2)[:, None]
        for H in HS:
            runs = [("M1", "linear", None, None)] + [("M2", "rcf", W, None) for W in M2_W] \
                + [("M3", "resampled", Wp, Pr) for Pr, Wp in M3_VARIANTS] + [("M4", "harmonic", None, None),
                                                                              ("M5", "seasonal_naive", None, None)]
            for tag, method, W, P_res in runs:
                if tag not in tags:
                    continue
                cfg = {"exp": "E4", "dataset": ds, "method": method, "tag": tag, "L": L, "H": H, "W": W,
                       "P_res": P_res, "lambdas": LAMBDAS, "revin": 0}
                t0 = time.time()
                try:
                    if method in ("linear", "rcf"):
                        out = fit_closed(data, b1, b2, L, H, method, W)
                    elif method == "resampled":
                        out = ResampledRCF(data, b1, b2, L, H, P_res, W).fit()
                    elif method == "harmonic":
                        out = fit_closed(data - hfit, b1, b2, L, H, "linear")
                    else:
                        vm, va, _ = seasonal_naive(data, b1, b2, 1, H)
                        tm, ta, kmax = seasonal_naive(data, b1, b2, 2, H)
                        out = {"val_mse": vm, "val_mae": va, "test_mse": tm, "test_mae": ta, "k_max": kmax}
                    out["test_rmse_m"] = math.sqrt(out["test_mse"]) * sd
                    status = "ok"
                except Exception as e:
                    out, status = {"error": repr(e)}, "failed"
                rec = dict(cfg, **out, status=status, source="closed_form", config_hash=config_hash(cfg),
                           seconds=round(time.time() - t0, 2), code_sha256=shas, train_sd_m=sd)
                log_run(rec)
                print(ds, H, tag, W, P_res, status,
                      {k: rec.get(k) for k in ["lambda", "val_mse", "test_mse", "test_rmse_m"]}, flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
