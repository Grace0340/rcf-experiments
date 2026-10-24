"""Round-trip interpolation error of the resampling pipeline on tide data, and the energy captured by each
phase table on the training segment."""
import json

import torch

from closed_form import DEV, DT, phase_table
from common import ROOT, load, log_run, file_sha
from resample import lagrange_weights, split_pos, to_grid

out = {"exp": "DIAG_E4", "code_sha256": {"diag_e4_interp.py": file_sha("diag_e4_interp.py")}}
for sid in json.load(open(f"{ROOT}/data/noaa/stations.json"))["chosen"]:
    data, b1, b2 = load(f"Tide{sid}", 336)
    x = torch.tensor(data, dtype=DT, device=DEV)
    T = x.shape[0]
    tr = x[b1[0]:b2[0]]
    row = {"train_var": float(tr.var())}
    for P_res, Wp in [(12.4206, 13), (24.8412, 25)]:
        s = P_res / Wp
        z, cav, k0 = to_grid(x, s)
        u = torch.arange(b1[2] + 336, T - 3, dtype=DT, device=DEV)
        fl, f, _ = split_pos(u / s)
        vals = z[fl[:, None] + torch.arange(-1, 3, device=DEV)[None]]
        back = (lagrange_weights(f)[..., None] * vals).sum(1)
        row[f"roundtrip_mse_{Wp}"] = float(((back - x[u.long()]) ** 2).mean())
        k_end = int(torch.searchsorted(cav, torch.tensor(b2[0] - 1, device=DEV), right=True))
        q = phase_table(z, k0, k_end, Wp)
        row[f"phase_table_energy_res_{Wp}"] = float(((q - q.mean(0)) ** 2).mean())
    for W in (12, 24, 25, 50):
        q = phase_table(x, b1[0], b2[0], W)
        row[f"phase_table_energy_int_{W}"] = float(((q - q.mean(0)) ** 2).mean())
    out[sid] = row
    print(sid, {k: round(v, 6) for k, v in row.items()})
log_run(out)
