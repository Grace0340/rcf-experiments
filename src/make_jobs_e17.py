"""Write the E17 job lists: wrapper check (3 runs) and the main grid (384 runs)."""
import json
import os

from common import PERIOD, ROOT
from official_runner import SPEC

TABLES = f"{ROOT}/results/e17_tables"
MLP96 = {"ETTh1": (256, 0.005), "ETTm1": (256, 0.005), "Electricity": (64, 0.005), "Solar": (64, 0.01)}
BACKBONES = [("linear", 336), ("mlp", 96)]
VARIANTS = {0: [("Z", None, 0), ("I", "m", 0), ("F", "m", 1)],
            1: [("Z", None, 0), ("I", "mn", 0), ("F", "mn", 1), ("Im", "m", 0), ("Fm", "m", 1)]}
ORDER = ["Electricity", "Solar", "ETTm1", "ETTh1"]


def job(exp, ds, mtype, L, revin, H, seed, variant, kind, freeze, wrapper=True):
    data, path, C, hp = SPEC[ds]
    bs, lr = hp[L] if mtype == "linear" else MLP96[ds]
    W = PERIOD[ds]
    init = ""
    if kind == "m":
        init = f"{TABLES}/{ds}_L{L}_m.npy"
    elif kind == "mn":
        init = f"{TABLES}/{ds}_L{L}_H{H}_mn.npy"
    tag = variant if wrapper else "direct"
    mid = f"{exp}_{ds}_{mtype}_rev{revin}_L{L}_H{H}_W{W}_{tag}_s{seed}"
    return {"exp": exp, "dataset": ds, "model": "CycleNet", "model_type": mtype, "revin": revin, "L": L, "H": H,
            "W": W, "seed": seed, "data": data, "data_path": path, "enc_in": C, "batch_size": bs, "lr": lr,
            "variant": variant, "init_kind": kind, "init_table": init, "freeze": freeze, "wrapper": wrapper,
            "model_id": mid}


def main():
    check = [job("E17_CHECK", "Electricity", "linear", 336, 0, 96, s, "Z", None, 0) for s in (2024, 2025)]
    check.append(job("E17_CHECK", "Electricity", "linear", 336, 0, 96, 2024, "Z", None, 0, wrapper=False))
    grid = [job("E17", ds, mt, L, r, H, s, v, k, f)
            for ds in ORDER for mt, L in BACKBONES for r in (0, 1) for H in (96, 336)
            for v, k, f in VARIANTS[r] for s in (2024, 2025, 2026)]
    assert len(grid) == 384 and len({j["model_id"] for j in grid}) == 384
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    json.dump(check, open(f"{ROOT}/results/jobs_e17_check.json", "w"), indent=1)
    json.dump(grid, open(f"{ROOT}/results/jobs_e17.json", "w"), indent=1)
    print(len(check), len(grid))


if __name__ == "__main__":
    main()
