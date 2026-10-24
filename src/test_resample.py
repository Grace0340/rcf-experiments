"""Checks of the resampling pipeline at non-integer step (E6 only exercises s = 1)."""
import math

import torch

from closed_form import DEV, DT, fit_closed
from common import load, log_run, file_sha
from resample import ResampledRCF, lagrange_weights, split_pos, to_grid

P = 12.4206
T = 6000
t = torch.arange(T, dtype=DT, device=DEV)
x = torch.cos(2 * math.pi * t / P)[:, None]
report = {"exp": "TEST_RESAMPLE", "code_sha256": {f: file_sha(f) for f in ["resample.py", "test_resample.py"]}}

for Wp in (13, 25):
    Pr = P if Wp == 13 else 2 * P
    s = Pr / Wp
    z, cav, k0 = to_grid(x, s)
    K = z.shape[0]
    tau = torch.arange(K, dtype=DT, device=DEV) * s
    fl, f, is_int = split_pos(tau)
    used = torch.where(is_int, fl, fl + 2)
    valid = torch.arange(K, device=DEV) >= k0
    valid &= ~torch.isnan(z[:, 0])
    # causality: every grid point k <= kl(t0) only uses samples <= t0
    t0 = torch.arange(50, T - 50, device=DEV)
    kl = torch.searchsorted(cav, t0, right=True) - 1
    max_used = torch.cummax(torch.where(valid, used, torch.full_like(used, -1)), 0).values[kl]
    causal = bool((max_used <= t0).all())
    tight = bool(((torch.searchsorted(cav, t0, right=True)) < K).all())
    grid_err = float((z[valid, 0] - torch.cos(2 * math.pi * tau[valid] / P)).abs().max())
    # back-interpolation from exact grid values to integer hours
    zt = torch.cos(2 * math.pi * tau / P)
    u = torch.arange(10, T - 10, dtype=DT, device=DEV)
    ufl, uf, _ = split_pos(u / s)
    vals = zt[ufl[:, None] + torch.arange(-1, 3, device=DEV)[None]]
    back = (lagrange_weights(uf) * vals).sum(1)
    back_err = float((back - torch.cos(2 * math.pi * u / P)).abs().max())
    res = {"Wp": Wp, "P_res": Pr, "s": s, "k0": k0, "n_valid_grid": int(valid.sum()), "K": K,
           "causal": causal, "kl_not_last": tight, "max_grid_interp_err": grid_err, "max_back_interp_err": back_err}
    print(res)
    report[f"W{Wp}"] = res

data, b1, b2 = load("ETTh1", 336)
r = ResampledRCF(data, b1, b2, 336, 96, 24.0, 24).fit()
direct = fit_closed(data, b1, b2, 336, 96, "rcf", 24)
print("ETTh1 s=1: resampled", r["test_mse"], "direct", direct["test_mse"])
report["etth1_s1_test_mse"] = r["test_mse"]
report["etth1_direct_test_mse"] = direct["test_mse"]
report["pass"] = (all(report[f"W{w}"]["causal"] for w in (13, 25))
                  and abs(r["test_mse"] - direct["test_mse"]) / direct["test_mse"] < 5e-3)
log_run(report)
print("pass", report["pass"])
