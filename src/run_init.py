"""E17 wrapper: run the official CycleNet run.py unchanged after setting its phase table.

Usage (cwd = CycleNet): python run_init.py [--init_table T.npy] [--freeze_table 0|1] <run.py arguments>
"""
import os
import runpy
import sys

import numpy as np
import torch

CN = os.getcwd()
argv = sys.argv[1:]


def pop(flag, default):
    if flag not in argv:
        return default
    i = argv.index(flag)
    value = argv[i + 1]
    del argv[i:i + 2]
    return value


INIT = pop("--init_table", "")
FREEZE = int(pop("--freeze_table", "0"))
sys.argv = [os.path.join(CN, "run.py")] + argv
sys.path.insert(0, CN)

import exp.exp_main as exp_main  # noqa: E402

_build = exp_main.Exp_Main._build_model


def build_model(self):
    model = _build(self)
    if INIT:
        table = model.module.cycleQueue.data if hasattr(model, "module") else model.cycleQueue.data
        q = torch.from_numpy(np.load(INIT)).to(table.dtype)
        assert tuple(q.shape) == tuple(table.shape), (q.shape, table.shape)
        with torch.no_grad():
            table.copy_(q)
        table.requires_grad_(not FREEZE)
        print(f"E17 table: init={INIT} freeze={FREEZE} norm={float(q.norm()):.6f}", flush=True)
    return model


exp_main.Exp_Main._build_model = build_model
runpy.run_path(os.path.join(CN, "run.py"), run_name="__main__")
