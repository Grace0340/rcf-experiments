"""Shared utilities: CycleNet-identical data splits, run logging."""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

ROOT = os.environ.get("RCF_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA = f"{ROOT}/CycleNet/dataset"
RUNS = f"{ROOT}/results/runs.jsonl"
PERIOD = {"ETTh1": 24, "ETTm1": 96, "Electricity": 168, "Traffic": 168, "Solar": 144}
FILE = {"Electricity": "electricity", "Traffic": "traffic"}


def load(name, L):
    """Return z-scored data (T, C) float64 and CycleNet borders (b1, b2) for lookback L."""
    if name.startswith("ETT"):
        raw = pd.read_csv(f"{DATA}/{name}.csv").iloc[:, 1:].to_numpy(np.float64)
        m = 30 * 24 * (1 if name.startswith("ETTh") else 4)
        b1 = [0, 12 * m - L, 16 * m - L]
        b2 = [12 * m, 16 * m, 20 * m]
    elif name == "Solar":
        raw = np.loadtxt(f"{DATA}/solar_AL.txt", delimiter=",")
        n = len(raw)
        ntr, nte, nva = int(n * 0.7), int(n * 0.2), int(n * 0.1)
        b1 = [0, ntr - L, n - nte - L]
        b2 = [ntr, ntr + nva, n]
    elif name in FILE or name.startswith("Tide"):
        path = f"{DATA}/noaa_{name[4:]}.csv" if name.startswith("Tide") else f"{DATA}/{FILE[name]}.csv"
        raw = pd.read_csv(path).iloc[:, 1:].to_numpy(np.float64)
        n = len(raw)
        ntr, nte = int(n * 0.7), int(n * 0.2)
        b1 = [0, ntr - L, n - nte - L]
        b2 = [ntr, n - nte, n]
    else:
        raise ValueError(name)
    return zscore(raw, b1, b2), b1, b2


def zscore(raw, b1, b2):
    tr = raw[b1[0]:b2[0]]
    mu, sd = tr.mean(0), tr.std(0)
    sd[sd == 0] = 1.0
    return (raw - mu) / sd


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def log_run(record, path=RUNS):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = dict(record)
    record.setdefault("logged_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
