"""Job lists for the official runs of E9 (MLP backbone, table lengths W) and E16 (ETTh1, official script settings)."""
import json

from common import PERIOD, ROOT
from official_runner import SPEC, job

SEEDS = [2024, 2025, 2026]
MLP_LR = {"Electricity": 0.005, "Traffic": 0.002, "Solar": 0.01}

e9 = []
for ds in ["Electricity", "Traffic", "Solar"]:
    data, path, C, _ = SPEC[ds]
    spec = (data, path, C, {96: (64, MLP_LR[ds])})
    for rev in [0, 1]:
        for W in [23, 24, 48, 96, 144, 168]:
            e9 += [job("E9", ds, "CycleNet", rev, 96, 96, W, s, spec=spec, model_type="mlp") for s in SEEDS]
    e9 += [job("E9", ds, "RMLP", 1, 96, 96, PERIOD[ds], s, spec=spec, model_type="mlp") for s in SEEDS]

e16 = []
for H in [96, 192, 336, 720]:
    for s in [2024, 2025, 2026, 2027, 2028]:
        e16 += [job("E16", "ETTh1", "CycleNet", rev, 336, H, 24, s) for rev in [1, 0]]
        e16 += [job("E16", "ETTh1", "Linear", 0, 336, H, 24, s)]

# Long jobs first so the pool stays busy.
order = {"Traffic": 0, "Electricity": 1, "Solar": 2}
e9.sort(key=lambda j: order[j["dataset"]])
json.dump(e9 + e16, open(f"{ROOT}/results/jobs_s2.json", "w"), indent=1)
print(len(e9), len(e16))
