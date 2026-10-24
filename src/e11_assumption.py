"""E11: phase means of training windows and targets compared with the shifted closed-form table."""
import torch

from closed_form import DEV, DT, Moments, phase_table
from common import PERIOD, file_sha, load, log_run

L = 336


def main():
    sha = {f: file_sha(f) for f in ["e11_assumption.py", "closed_form.py", "common.py"]}
    for ds in ["ETTh1", "ETTm1", "Electricity", "Solar"]:
        W = PERIOD[ds]
        data, b1, b2 = load(ds, L)
        x = torch.tensor(data, dtype=DT, device=DEV)
        m = phase_table(x, b1[0], b2[0], W)                       # (W, C), absolute phase
        for H in [96, 720]:
            mom = Moments(L, H, x.shape[1], W)
            mom.add(x[b1[0]:b2[0]], b1[0])
            xbar = mom.gx / mom.gn[..., None]                     # (C, W, L), phase = window start mod W
            ybar = mom.gy / mom.gn[..., None]
            ph = torch.arange(W, device=DEV)
            Sm = m[(ph[:, None] + torch.arange(L, device=DEV)[None]) % W].permute(2, 0, 1)
            Fm = m[(ph[:, None] + L + torch.arange(H, device=DEV)[None]) % W].permute(2, 0, 1)
            mu = m.mean(0)[:, None, None]
            ex = float((xbar - Sm).norm() / (Sm - mu).norm())
            ey = float((ybar - Fm).norm() / (Fm - mu).norm())
            log_run({"exp": "E11", "dataset": ds, "L": L, "H": H, "W": W, "eps_x": ex, "eps_y": ey,
                     "n_train": b2[0] - b1[0], "code_sha256": sha, "status": "ok", "source": "analysis"})
            print(ds, H, round(ex, 4), round(ey, 4), flush=True)


if __name__ == "__main__":
    main()
