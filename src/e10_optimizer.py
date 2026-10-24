"""E10: CycleNet/Linear without RevIN trained by our own loop with Adam or plain SGD, recording how the learned
table relates to m along the singular directions of the intercept operator. The table is indexed by absolute phase (window start mod W), so it is compared with m
directly. Training follows the official settings: batch 64, <= 30 epochs, early stopping on validation MSE with
patience 5, 'type3' schedule (lr constant for the first four epochs, then x0.8 per epoch), drop_last for training."""
import copy
import math
import time

import torch

from closed_form import DEV, DT, phase_table
from common import PERIOD, config_hash, file_sha, load, log_run
from e5_nullspace import intercept_operator

DS, L, W = "Electricity", 336, PERIOD["Electricity"]
HS = [96, 336]
SEEDS = [2024, 2025, 2026]
RUNS = [("adam", 0.005)] + [("sgd", lr) for lr in [0.01, 0.03, 0.1, 0.3, 1.0]]
TAUS = [0.05, 0.1, 0.2]


class RCFLinear(torch.nn.Module):
    def __init__(self, L, H, W, C):
        super().__init__()
        self.L, self.H, self.W = L, H, W
        self.Q = torch.nn.Parameter(torch.zeros(W, C))
        self.lin = torch.nn.Linear(L, H)

    def forward(self, x, s):
        L, H, W = self.L, self.H, self.W
        jp = (s[:, None] + torch.arange(L, device=x.device)[None]) % W
        jf = (s[:, None] + L + torch.arange(H, device=x.device)[None]) % W
        r = x - self.Q[jp]
        return self.lin(r.transpose(1, 2)).transpose(1, 2) + self.Q[jf]


def starts(b1, b2, j, H):
    return torch.arange(b1[j], b2[j] - L - H + 1, device=DEV)


def batch(x, s, H):
    idx = s[:, None] + torch.arange(L + H, device=DEV)[None]
    v = x[idx]
    return v[:, :L], v[:, L:]


@torch.no_grad()
def mse(model, x, s_all, H):
    se, cnt = 0.0, 0
    for i in range(0, len(s_all), 256):
        s = s_all[i:i + 256]
        xb, yb = batch(x, s, H)
        e = model(xb, s) - yb
        se += float((e.double() ** 2).sum())
        cnt += e.numel()
    return se / cnt


def train(x, b1, b2, H, opt_name, lr, seed):
    torch.manual_seed(seed)
    model = RCFLinear(L, H, W, x.shape[1]).to(DEV)
    opt = (torch.optim.Adam if opt_name == "adam" else torch.optim.SGD)(model.parameters(), lr=lr)
    s_tr, s_va = starts(b1, b2, 0, H), starts(b1, b2, 1, H)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad, epochs, diverged = math.inf, None, 0, 0, False
    for ep in range(30):
        perm = s_tr[torch.randperm(len(s_tr), generator=g).to(DEV)]
        model.train()
        for i in range(0, len(perm) - 63, 64):
            s = perm[i:i + 64]
            xb, yb = batch(x, s, H)
            loss = ((model(xb, s) - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        epochs = ep + 1
        v = mse(model, x, s_va, H)
        if not math.isfinite(v):
            diverged = True
            break
        if v < best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= 5:
                break
        e = ep + 1
        for pg in opt.param_groups:
            pg["lr"] = lr if e < 3 else lr * 0.8 ** (e - 3)
    if best_state is None:
        return None, best, epochs, True
    model.load_state_dict(best_state)
    return model, best, epochs, diverged


def table_metrics(model, m):
    Q = model.Q.detach().to(DT)
    B = model.lin.weight.detach().to(DT)
    G = intercept_operator(B, L, W)
    _, S, Vh = torch.linalg.svd(G, full_matrices=False)
    mc, Qc = m - m.mean(0), Q - Q.mean(0)
    dQ = Qc - mc
    out = {"rel_err_Q_vs_m": float(dQ.norm() / mc.norm()),
           "rel_err_intercepts": float((G @ dQ).norm() / (G @ mc).norm())}
    for tau in TAUS:
        weak = Vh[S < tau * S[0]]
        out[f"frac_weak_{tau}"] = float(((weak @ dQ) ** 2).sum() / (dQ ** 2).sum()) if len(weak) else 0.0
        out[f"weak_ratio_{tau}"] = (float((weak @ Qc).norm() / (weak @ mc).norm())
                                    if len(weak) and (weak @ mc).norm() > 0 else None)
        out[f"n_weak_{tau}"] = len(weak)
    return out


def main():
    sha = {f: file_sha(f) for f in ["e10_optimizer.py", "e5_nullspace.py", "closed_form.py", "common.py"]}
    data, b1, b2 = load(DS, L)
    x64 = torch.tensor(data, dtype=DT, device=DEV)
    m = phase_table(x64, b1[0], b2[0], W)
    x = x64.float()
    for H in HS:
        s_te = starts(b1, b2, 2, H)
        for seed in SEEDS:
            for opt_name, lr in RUNS:
                cfg = {"exp": "E10", "dataset": DS, "L": L, "H": H, "W": W, "seed": seed, "optimizer": opt_name,
                       "lr": lr, "revin": 0}
                t0 = time.time()
                try:
                    model, val, epochs, diverged = train(x, b1, b2, H, opt_name, lr, seed)
                    rec = dict(cfg, val_mse=val, epochs_run=epochs, diverged=diverged, status="ok")
                    if model is not None:
                        rec.update(test_mse=mse(model, x, s_te, H), **table_metrics(model, m))
                except Exception as e:
                    rec = dict(cfg, status="failed", error=repr(e))
                rec.update(source="own_trainer", config_hash=config_hash(cfg), code_sha256=sha,
                           seconds=round(time.time() - t0, 1))
                log_run(rec)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                       if k in ("H", "seed", "optimizer", "lr", "val_mse", "test_mse", "epochs_run", "diverged",
                                "frac_weak_0.1", "weak_ratio_0.1", "rel_err_Q_vs_m", "status")}, flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
