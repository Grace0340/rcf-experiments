"""E17: official CycleNet with zero, closed-form-initialised or fixed phase tables; log every run."""
import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from common import ROOT, config_hash, file_sha, log_run
from official_runner import CN, CODE, ENV, NUM_WORKERS, PY, THREADS
FIELDS = ["exp", "dataset", "model", "model_type", "revin", "L", "H", "W", "seed", "batch_size", "lr", "variant"]


def run(j):
    entry = f"{CODE}/run_init.py" if j["wrapper"] else "run.py"
    args = [PY, "-u", entry]
    if j["init_table"]:
        args += ["--init_table", j["init_table"], "--freeze_table", str(j["freeze"])]
    args += ["--is_training", "1", "--root_path", "./dataset/", "--data_path", j["data_path"],
             "--model_id", j["model_id"], "--model", "CycleNet", "--data", j["data"], "--features", "M",
             "--seq_len", str(j["L"]), "--pred_len", str(j["H"]), "--enc_in", str(j["enc_in"]),
             "--cycle", str(j["W"]), "--model_type", j["model_type"], "--use_revin", str(j["revin"]),
             "--train_epochs", "30", "--patience", "5", "--itr", "1", "--batch_size", str(j["batch_size"]),
             "--learning_rate", str(j["lr"]), "--random_seed", str(j["seed"]), "--num_workers", NUM_WORKERS]
    log = f"{ROOT}/logs/e17/{j['model_id']}.log"
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(args, cwd=CN, stdout=fh, stderr=subprocess.STDOUT, env=ENV).returncode
    text = open(log).read()
    m = re.findall(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", text)
    curve = re.findall(r"Train Loss: ([0-9.eE+-]+) Vali Loss: ([0-9.eE+-]+) Test Loss: ([0-9.eE+-]+)", text)
    vals = [float(v) for _, v, _ in curve]
    setting = (f"{j['model_id']}_CycleNet_{j['data']}_ftM_sl{j['L']}_pl{j['H']}_cycle{j['W']}"
               f"_{j['model_type']}_seed{j['seed']}")
    ckpt = f"{CN}/checkpoints/{setting}/checkpoint.pth"
    table = {}
    if rc == 0 and os.path.exists(ckpt):
        q = torch.load(ckpt, map_location="cpu")["cycleQueue.data"].double()
        table["final_table_norm"] = float(q.norm())
        if j["init_table"]:
            q0 = torch.from_numpy(np.load(j["init_table"])).double()
            table["final_minus_init_norm"] = float((q - q0).norm())
            table["init_table_norm"] = float(q0.norm())
    cfg = {k: j[k] for k in FIELDS}
    rec = dict(cfg, source="official", entry=os.path.basename(entry), init_table=j["init_table"],
               freeze_table=j["freeze"], model_id=j["model_id"], returncode=rc,
               status="ok" if (rc == 0 and m) else "failed",
               test_mse=float(m[-1][0]) if m else None, test_mae=float(m[-1][1]) if m else None,
               val_mse=min(vals) if vals else None, best_epoch=int(np.argmin(vals)) + 1 if vals else None,
               epochs_run=len(vals), val_curve=vals, train_curve=[float(a) for a, _, _ in curve],
               test_curve=[float(c) for _, _, c in curve], checkpoint=ckpt, log=log,
               seconds=round(time.time() - t0, 1), config_hash=config_hash(cfg), threads=THREADS,
               num_workers=NUM_WORKERS, cyclenet_commit="d807e51fc2dcd143885ee639d97965a7ab0926f4",
               code_sha256={f: file_sha(f"{CODE}/{f}") for f in ["e17_runner.py", "run_init.py", "official_runner.py"]},
               **table)
    log_run(rec)
    print(j["model_id"], rec["status"], rec["test_mse"], rec["best_epoch"], rec["seconds"], flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
