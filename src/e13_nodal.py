"""E13: harmonic regression with nodal corrections (UTide), fitted on the
training segment only and extrapolated, followed by a closed-form linear model of the residual. L = 336."""
import json
import math
import shutil

import numpy as np
import pandas as pd
import utide

from closed_form import LAMBDAS, fit_closed
from common import DATA, ROOT, config_hash, file_sha, load, log_run

CONSTIT = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "M4", "MS4", "M6", "SA", "SSA"]


def stations():
    out = []
    for f in ["stations.json", "stations_extra.json"]:
        try:
            st = json.load(open(f"{ROOT}/data/noaa/{f}"))
        except FileNotFoundError:
            continue
        lat = {r["id"]: float(r["lat"]) for r in st["log"] if "lat" in r}
        out += [(sid, lat[sid]) for sid in st["chosen"]]
    return out


def main():
    sha = {f: file_sha(f) for f in ["e13_nodal.py", "closed_form.py", "common.py"]}
    for sid, lat in stations():
        shutil.copy(f"{ROOT}/data/noaa/{sid}.csv", f"{DATA}/noaa_{sid}.csv")
        ds = f"Tide{sid}"
        data, b1, b2 = load(ds, 336)
        t = pd.to_datetime(pd.read_csv(f"{DATA}/noaa_{sid}.csv")["date"]).to_numpy()
        u = data[:, 0]
        coef = utide.solve(t[:b2[0]], u[:b2[0]], lat=lat, constit=CONSTIT, nodal=True, trend=False,
                           method="ols", conf_int="none", verbose=False)
        fit = utide.reconstruct(t, coef, verbose=False).h
        amp = {str(n): float(a) for n, a in zip(coef.name, coef.A)}
        raw = np.loadtxt(f"{DATA}/noaa_{sid}.csv", delimiter=",", skiprows=1, usecols=1)
        sd = float(raw[b1[0]:b2[0]].std())
        for H in [24, 96, 336]:
            cfg = {"exp": "E13", "dataset": ds, "method": "harmonic_nodal", "tag": "M4n", "L": 336, "H": H,
                   "lambdas": LAMBDAS, "constit": CONSTIT}
            try:
                out = fit_closed(data - fit[:, None], b1, b2, 336, H, "linear")
                out["test_rmse_m"] = math.sqrt(out["test_mse"]) * sd
                status = "ok"
            except Exception as e:
                out, status = {"error": repr(e)}, "failed"
            log_run(dict(cfg, **out, status=status, source="closed_form", config_hash=config_hash(cfg),
                         code_sha256=sha, utide_version=utide.__version__, amplitudes=amp,
                         train_fit_mse=float(np.mean((u[:b2[0]] - fit[:b2[0]]) ** 2))))
            print(ds, H, status, out.get("test_mse"), flush=True)


if __name__ == "__main__":
    main()
