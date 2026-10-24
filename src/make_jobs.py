"""Job lists for the official runs of E1 and E2."""
import json
import os

from common import PERIOD, ROOT
from official_runner import job

SEEDS = [2024, 2025, 2026]

e1 = [job("E1", ds, "CycleNet", 0, 336, H, PERIOD[ds], s)
      for ds in ["ETTh1", "ETTm1", "Electricity", "Solar"] for H in [96, 192, 336, 720] for s in SEEDS]

e2 = []
for ds in ["Electricity", "Traffic", "Solar"]:
    for rev in [0, 1]:
        for W in [23, 24, 48, 96, 144, 168]:
            e2 += [job("E2", ds, "CycleNet", rev, 96, 96, W, s) for s in SEEDS]
        e2 += [job("E2", ds, "Linear" if rev == 0 else "RLinear", rev, 96, 96, PERIOD[ds], s) for s in SEEDS]

os.makedirs(f"{ROOT}/results", exist_ok=True)
json.dump(e1, open(f"{ROOT}/results/jobs_e1.json", "w"), indent=1)
json.dump(e2, open(f"{ROOT}/results/jobs_e2.json", "w"), indent=1)
print(len(e1), len(e2))
