"""E17: build closed-form initial tables in official index order and check them against the official loader and model.

m   : per-phase mean of the z-scored training segment.
m^n : mean over training windows of the RevIN-normalised inputs, grouped by absolute phase.
Official CycleNet indexes Q[k] at absolute phase (k - L) mod W, so both tables are rolled by L.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

from closed_form import DEV, phase_table
from common import PERIOD, ROOT, file_sha, load, log_run

CN = f"{ROOT}/CycleNet"
sys.path.insert(0, CN)
from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Solar  # noqa: E402
from models import CycleNet  # noqa: E402

OUT = f"{ROOT}/results/e17_tables"
DATASETS = ["ETTh1", "ETTm1", "Electricity", "Solar"]
LS, HS = [336, 96], [96, 336]
OFFICIAL = {"ETTh1": (Dataset_ETT_hour, "ETTh1.csv", "h"), "ETTm1": (Dataset_ETT_minute, "ETTm1.csv", "h"),
            "Electricity": (Dataset_Custom, "electricity.csv", "h"), "Solar": (Dataset_Solar, "solar_AL.txt", "h")}
CHUNK = 256


def table_m(x, n_tr, W, L):
    return torch.roll(phase_table(x, 0, n_tr, W), shifts=L, dims=0)


def table_mn(x, n_tr, W, L, H):
    C = x.shape[1]
    starts = torch.arange(0, n_tr - L - H + 1, device=x.device)
    ar = torch.arange(L, device=x.device)
    acc = torch.zeros(W, C, dtype=x.dtype, device=x.device)
    cnt = torch.zeros(W, dtype=x.dtype, device=x.device)
    for i in range(0, len(starts), CHUNK):
        t = starts[i:i + CHUNK, None] + ar
        X = x[t]
        Z = (X - X.mean(1, keepdim=True)) / torch.sqrt(X.var(1, keepdim=True) + 1e-5)
        ph = (t % W).reshape(-1)
        acc.index_add_(0, ph, Z.reshape(-1, C))
        cnt.index_add_(0, ph, torch.ones_like(ph, dtype=x.dtype))
    return torch.roll(acc / cnt[:, None], shifts=L, dims=0)


def official_set(ds, L, H, W):
    cls, path, freq = OFFICIAL[ds]
    return cls(root_path=f"{CN}/dataset/", flag="train", size=[L, 0, H], features="M", data_path=path,
               target="OT", timeenc=1, freq=freq, cycle=W)


def official_mn(dset, L, W, C):
    """m^n computed from the official loader, official RevIN formula (float32) and official cycle_index."""
    loader = torch.utils.data.DataLoader(dset, batch_size=512, shuffle=False, drop_last=False)
    acc = torch.zeros(W, C, dtype=torch.float64, device=DEV)
    cnt = torch.zeros(W, dtype=torch.float64, device=DEV)
    ar = torch.arange(L, device=DEV)
    for bx, _, _, _, ci in loader:
        x = bx.float().to(DEV)
        x = (x - x.mean(1, keepdim=True)) / torch.sqrt(torch.var(x, dim=1, keepdim=True) + 1e-5)
        k = ((ci.to(DEV).view(-1, 1) + ar) % W).reshape(-1)
        acc.index_add_(0, k, x.reshape(-1, C).double())
        cnt.index_add_(0, k, torch.ones_like(k, dtype=torch.float64))
    return acc / cnt[:, None]


def zero_backbone_check(dset, m_abs, q, L, H, W, C, revin):
    cfg = SimpleNamespace(seq_len=L, pred_len=H, enc_in=C, cycle=W, model_type="linear", d_model=512,
                          use_revin=revin)
    model = CycleNet.Model(cfg).float().to(DEV)
    with torch.no_grad():
        model.model.weight.zero_()
        model.model.bias.zero_()
        model.cycleQueue.data.copy_(q.float())
    n = len(dset)
    err_in = err_out = 0.0
    for s in sorted({0, 1, 777 % n, n // 2, n - 1}):
        bx, _, _, _, ci = dset[s]
        x = torch.tensor(bx, dtype=torch.float32, device=DEV)[None]
        ci = ci.view(1).to(DEV)
        with torch.no_grad():
            y = model(x, ci)[0]
            q_in = model.cycleQueue(ci, L)[0]
        t_in = torch.arange(s, s + L, device=DEV)
        t_out = torch.arange(s + L, s + L + H, device=DEV)
        if revin:
            mu, var = x[0].mean(0), torch.var(x[0], dim=0) + 1e-5
            want_out = q[(t_out + L) % W].float() * torch.sqrt(var) + mu
            err_in = max(err_in, float((q_in - q[(t_in + L) % W].float()).abs().max()))
        else:
            want_out = m_abs[t_out % W].float()
            err_in = max(err_in, float((q_in - m_abs[t_in % W].float()).abs().max()))
        err_out = max(err_out, float((y - want_out).abs().max()))
    return err_in, err_out


def main():
    os.makedirs(OUT, exist_ok=True)
    sha = {"e17_tables.py": file_sha(__file__)}
    ok_all = True
    for ds in DATASETS:
        W = PERIOD[ds]
        for L in LS:
            data, b1, b2 = load(ds, L)
            x = torch.tensor(data, device=DEV)
            n_tr, C = b2[0], x.shape[1]
            m_abs = phase_table(x, 0, n_tr, W)
            qm = table_m(x, n_tr, W, L)
            np.save(f"{OUT}/{ds}_L{L}_m.npy", qm.cpu().numpy().astype(np.float32))
            for H in HS:
                qn = table_mn(x, n_tr, W, L, H)
                np.save(f"{OUT}/{ds}_L{L}_H{H}_mn.npy", qn.cpu().numpy().astype(np.float32))
                dset = official_set(ds, L, H, W)
                d_data = float(np.abs(dset.data_x - data[:n_tr]).max())
                d_mn = float((official_mn(dset, L, W, C) - qn).abs().max())
                e0 = zero_backbone_check(dset, m_abs, qm, L, H, W, C, 0)
                e1 = zero_backbone_check(dset, m_abs, qn, L, H, W, C, 1)
                passed = d_data < 1e-6 and d_mn < 1e-4 and max(e0) < 1e-5 and max(e1) < 1e-4
                ok_all &= passed
                rec = dict(exp="E17_CHECK", check="C1", dataset=ds, L=L, H=H, W=W, n_windows=len(dset),
                           max_abs_data_diff=d_data, max_abs_mn_vs_official_loader=d_mn,
                           rev0_table_in_err=e0[0], rev0_out_err=e0[1], rev1_table_in_err=e1[0],
                           rev1_out_err=e1[1], m_norm=float(qm.norm()), mn_norm=float(qn.norm()),
                           status="pass" if passed else "fail", code_sha256=sha,
                           cyclenet_commit="d807e51fc2dcd143885ee639d97965a7ab0926f4")
                log_run(rec)
                print(ds, L, H, rec["status"], f"data {d_data:.2e} mn {d_mn:.2e} rev0 {e0} rev1 {e1}", flush=True)
    print("C1", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
