"""Build the dataset files expected by the official CycleNet loaders."""
import gzip
import os
import shutil

import numpy as np
import pandas as pd

from common import ROOT

RAW = f"{ROOT}/data/raw"
OUT = f"{ROOT}/CycleNet/dataset"
os.makedirs(OUT, exist_ok=True)

for f in ["ETTh1", "ETTm1"]:
    shutil.copy(f"{RAW}/{f}.csv", f"{OUT}/{f}.csv")

for name in ["electricity", "traffic"]:
    with gzip.open(f"{RAW}/{name}.txt.gz", "rt") as fh:
        a = np.loadtxt(fh, delimiter=",")
    cols = [str(i) for i in range(a.shape[1] - 1)] + ["OT"]
    df = pd.DataFrame(a, columns=cols)
    stamps = pd.date_range("2016-07-01 02:00:00", periods=len(a), freq="h")
    df.insert(0, "date", stamps.strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    print(name, a.shape)

with gzip.open(f"{RAW}/solar_AL.txt.gz", "rb") as fi, open(f"{OUT}/solar_AL.txt", "wb") as fo:
    shutil.copyfileobj(fi, fo)
print("done")
