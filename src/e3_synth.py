"""E3: integer-length phase-mean table on synthetic series with a non-integer period."""
import json
import time

import numpy as np

from common import config_hash, file_sha, log_run

A_TRUE = np.array([1.0, 0.5, 0.25])
RHO, SIGMA = 0.5, 0.5
CASES = [(12.42, 12), (24.5, 24)]
NS = [10, 30, 100, 300]
REPS = 50
LAMBDAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def signal(N, P, rng, burn=200):
    phi = rng.uniform(0, 2 * np.pi, 3)
    t = np.arange(N)
    m = sum(A_TRUE[r] * np.cos(2 * np.pi * (r + 1) * t / P + phi[r]) for r in range(3))
    e = rng.normal(0, SIGMA, N + burn)
    u = np.empty_like(e)
    u[0] = e[0]
    for i in range(1, len(e)):
        u[i] = RHO * u[i - 1] + e[i]
    return m + u[burn:], m, phi


def harmonic_coefs(q, P):
    W = len(q)
    ph = np.arange(W)
    cols = [np.ones(W)]
    for r in range(1, 4):
        cols += [np.cos(2 * np.pi * r * ph / P), np.sin(2 * np.pi * r * ph / P)]
    beta = np.linalg.lstsq(np.stack(cols, 1), q, rcond=None)[0]
    return np.array([beta[2 * r - 1] - 1j * beta[2 * r] for r in range(1, 4)])


def dirichlet_mean(n, theta):
    return np.exp(1j * np.arange(n) * theta).mean()


def part_a():
    rows = []
    for P, W in CASES:
        for n in NS:
            cn = []
            for rep in range(REPS):
                rng = np.random.default_rng(rep)
                x, _, phi = signal(n * W, P, rng)
                q = x.reshape(n, W).mean(0)
                cn.append(harmonic_coefs(q, P) * np.exp(-1j * phi))
            cn = np.array(cn)
            for r in range(3):
                theta = 2 * np.pi * (r + 1) * (W - P) / P
                pred = A_TRUE[r] * dirichlet_mean(n, theta)
                emp = cn[:, r].mean()
                rows.append({"P": P, "W": W, "n": n, "r": r + 1,
                             "pred_abs": abs(pred), "emp_abs_of_mean": abs(emp),
                             "mean_abs": float(np.abs(cn[:, r]).mean()),
                             "err_rel": abs(emp - pred) / A_TRUE[r]})
    return rows


def windows(x, L, H):
    v = np.lib.stride_tricks.sliding_window_view(x, L + H)
    return v[:, :L], v[:, L:]


def ridge_fit(X, Y, lam):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    A = np.linalg.solve(Xc.T @ Xc / len(X) + lam * np.eye(X.shape[1]), Xc.T @ Yc / len(X))
    return A, my - mx @ A


def forecast_mse(series, mean_curve, n_tr, n_va, L, H):
    """series and mean_curve aligned; mean_curve=None means plain Linear."""
    r = series - (mean_curve if mean_curve is not None else 0.0)
    tr, va, te = r[:n_tr], r[n_tr - L:n_tr + n_va], r[n_tr + n_va - L:]
    Xtr, Ytr = windows(tr, L, H)
    best = None
    for lam in LAMBDAS:
        A, b = ridge_fit(Xtr, Ytr, lam)
        Xv, Yv = windows(va, L, H)
        v = np.mean((Yv - Xv @ A - b) ** 2)
        if best is None or v < best[0]:
            best = (v, A, b)
    Xt, Yt = windows(te, L, H)
    return float(np.mean((Yt - Xt @ best[1] - best[2]) ** 2))


def part_b():
    rows = []
    for P, W in CASES:
        L, H = W, 2 * W
        n_va, n_te = 20 * W, 50 * W
        for n in NS:
            res = {"rcf_sa": [], "oracle": [], "linear": []}
            for rep in range(REPS):
                rng = np.random.default_rng(10_000 + rep)
                n_tr = n * W
                x, m, _ = signal(n_tr + n_va + n_te, P, rng)
                ph = np.arange(len(x)) % W
                q = np.bincount(ph[:n_tr], weights=x[:n_tr], minlength=W) / np.bincount(ph[:n_tr], minlength=W)
                res["rcf_sa"].append(forecast_mse(x, q[ph], n_tr, n_va, L, H))
                res["oracle"].append(forecast_mse(x, m, n_tr, n_va, L, H))
                res["linear"].append(forecast_mse(x, None, n_tr, n_va, L, H))
            rows.append({"P": P, "W": W, "n": n, "L": L, "H": H,
                         **{k + "_mse_mean": float(np.mean(v)) for k, v in res.items()},
                         **{k + "_mse_std": float(np.std(v)) for k, v in res.items()}})
    return rows


def main():
    sha = {f: file_sha(f) for f in ["e3_synth.py", "common.py"]}
    cfg = {"exp": "E3", "A": A_TRUE.tolist(), "rho": RHO, "sigma": SIGMA, "cases": CASES, "ns": NS, "reps": REPS}
    t0 = time.time()
    a = part_a()
    worst = float(max(r["err_rel"] for r in a))
    log_run({**cfg, "part": "a", "rows": a, "max_err_rel": worst,
             "config_hash": config_hash(cfg), "code_sha256": sha, "seconds": round(time.time() - t0, 2)})
    print(json.dumps({"part_a_max_err_rel": worst}))
    for r in a:
        print(r)
    t1 = time.time()
    b = part_b()
    log_run({**cfg, "part": "b", "rows": b, "config_hash": config_hash(cfg), "code_sha256": sha,
             "seconds": round(time.time() - t1, 2)})
    for r in b:
        print(r)


if __name__ == "__main__":
    main()
