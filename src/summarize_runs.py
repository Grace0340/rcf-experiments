"""Group the successful runs in results/runs.jsonl by configuration and write the mean over seeds.

Usage: python summarize_runs.py [EXP ...]      (default: all experiments)
Output: results/summary.csv and a printout.
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np

from common import ROOT

KEYS = ["exp", "source", "dataset", "model", "model_type", "method", "tag", "variant", "revin", "L", "H", "W",
        "P_res", "optimizer", "lr"]
METRICS = ["test_mse", "test_mae", "val_mse", "epochs_run"]


def main():
    want = set(sys.argv[1:])
    groups = defaultdict(list)
    for line in open(f"{ROOT}/results/runs.jsonl"):
        r = json.loads(line)
        if r.get("status") != "ok" or r.get("test_mse") is None or (want and r.get("exp") not in want):
            continue
        groups[tuple(r.get(k) for k in KEYS)].append(r)
    rows = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        row = dict(zip(KEYS, key))
        row["n_runs"] = len(rs)
        row["seeds"] = " ".join(str(s) for s in sorted({r.get("seed") for r in rs if r.get("seed") is not None}))
        for m in METRICS:
            v = [r[m] for r in rs if r.get(m) is not None]
            row[f"{m}_mean"] = float(np.mean(v)) if v else None
            row[f"{m}_std"] = float(np.std(v)) if v else None
        rows.append(row)
    with open(f"{ROOT}/results/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else KEYS)
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        cfg = " ".join(f"{k}={row[k]}" for k in KEYS if row[k] is not None)
        print(f"{cfg} | n={row['n_runs']} test_mse={row['test_mse_mean']:.4f}")
    print(f"{len(rows)} configurations -> {ROOT}/results/summary.csv")


if __name__ == "__main__":
    main()
