"""Official CycleNet cross-check of M2 on the tide stations (no RevIN, W in {12, 25}, 3 seeds).
No official tide script exists; batch size and learning rate follow the official custom-data scripts at L=336."""
import json

from common import ROOT
from official_runner import job

SEEDS = [2024, 2025, 2026]
chosen = json.load(open(f"{ROOT}/data/noaa/stations.json"))["chosen"]
jobs = [job("E4", f"Tide{sid}", "CycleNet", 0, 336, H, W, s, spec=("custom", f"noaa_{sid}.csv", 1, {336: (64, 0.005)}))
        for sid in chosen for H in [24, 96, 336] for W in [12, 25] for s in SEEDS]
json.dump(jobs, open(f"{ROOT}/results/jobs_e4.json", "w"), indent=1)
print(len(jobs))
