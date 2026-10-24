"""Run official CycleNet / Linear / RLinear jobs in parallel and log every run."""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from common import PERIOD, ROOT, config_hash, file_sha, log_run

CN = f"{ROOT}/CycleNet"
CODE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get("RCF_PYTHON", sys.executable)
# Several runs share the machine; unbounded torch threads per run stall training.
THREADS = os.environ.get("RCF_THREADS", "2")
NUM_WORKERS = os.environ.get("RCF_NUM_WORKERS", "1")
ENV = dict(os.environ, OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS)
SPEC = {  # data flag, file, channels, (batch, lr) for L=96 and L=336 from the official scripts
    "ETTh1": ("ETTh1", "ETTh1.csv", 7, {96: (256, 0.01), 336: (256, 0.005)}),
    "ETTm1": ("ETTm1", "ETTm1.csv", 7, {96: (256, 0.01), 336: (256, 0.005)}),
    "Electricity": ("custom", "electricity.csv", 321, {96: (64, 0.01), 336: (64, 0.005)}),
    "Traffic": ("custom", "traffic.csv", 862, {96: (64, 0.01), 336: (64, 0.005)}),
    "Solar": ("Solar", "solar_AL.txt", 137, {96: (64, 0.01), 336: (64, 0.005)}),
}


def job(exp, ds, model, revin, L, H, W, seed, spec=None, model_type="linear"):
    data, path, C, hp = spec or SPEC[ds]
    bs, lr = hp[L]
    mid = f"{exp}_{ds}_{model}_{model_type}_rev{revin}_L{L}_H{H}_W{W}_s{seed}"
    return {"exp": exp, "dataset": ds, "model": model, "revin": revin, "L": L, "H": H, "W": W, "seed": seed,
            "data": data, "data_path": path, "enc_in": C, "batch_size": bs, "lr": lr, "model_id": mid,
            "model_type": model_type}


def run(j):
    args = [PY, "-u", "run.py", "--is_training", "1", "--root_path", "./dataset/", "--data_path", j["data_path"],
            "--model_id", j["model_id"], "--model", j["model"], "--data", j["data"], "--features", "M",
            "--seq_len", str(j["L"]), "--pred_len", str(j["H"]), "--enc_in", str(j["enc_in"]),
            "--cycle", str(j["W"]), "--model_type", j.get("model_type", "linear"), "--use_revin", str(j["revin"]),
            "--train_epochs", "30", "--patience", "5", "--itr", "1", "--batch_size", str(j["batch_size"]),
            "--learning_rate", str(j["lr"]), "--random_seed", str(j["seed"]), "--num_workers", NUM_WORKERS]
    log = f"{ROOT}/logs/official/{j['model_id']}.log"
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.run(args, cwd=CN, stdout=fh, stderr=subprocess.STDOUT, env=ENV).returncode
    text = open(log).read()
    m = re.findall(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", text)
    vals = [float(v) for v in re.findall(r"Vali Loss: ([0-9.eE+-]+)", text)]
    mtype = j.get("model_type", "linear")
    setting = (f"{j['model_id']}_{j['model']}_{j['data']}_ftM_sl{j['L']}_pl{j['H']}_cycle{j['W']}"
               f"_{mtype}_seed{j['seed']}")
    cfg = {k: j[k] for k in ["exp", "dataset", "model", "revin", "L", "H", "W", "seed", "batch_size", "lr"]}
    if mtype != "linear":
        cfg["model_type"] = mtype
    rec = dict(cfg, source="official", model_id=j["model_id"], returncode=rc,
               status="ok" if (rc == 0 and m) else "failed",
               test_mse=float(m[-1][0]) if m else None, test_mae=float(m[-1][1]) if m else None,
               val_mse=min(vals) if vals else None, epochs_run=len(vals),
               checkpoint=f"{CN}/checkpoints/{setting}/checkpoint.pth", log=log,
               seconds=round(time.time() - t0, 1), config_hash=config_hash(cfg),
               threads=THREADS, num_workers=NUM_WORKERS,
               code_sha256={"official_runner.py": file_sha(f"{CODE}/official_runner.py")},
               cyclenet_commit="d807e51fc2dcd143885ee639d97965a7ab0926f4")
    log_run(rec)
    print(j["model_id"], rec["status"], rec["test_mse"], rec["seconds"], flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
