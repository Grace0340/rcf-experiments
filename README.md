# rcf-experiments

Code for experiments on residual cycle forecasting (RCF), the phase-table mechanism of
[CycleNet](https://github.com/ACAT-SCUT/CycleNet): closed-form linear forecasters with a phase table,
runners for the unmodified official CycleNet code, synthetic experiments, and experiments on NOAA tide-gauge
records.

A citation will be added when the accompanying article is published.

## Requirements

- Linux with a CUDA GPU (the closed forms also run on CPU, slowly).
- Python 3.12, `requirements.txt`. All official CycleNet runs used NumPy 1.26.4.
- Experiment E13 uses UTide, which needs NumPy 2. Run it from a separate environment built from
  `requirements-utide.txt`.

## Setup

```bash
pip install -r requirements.txt   # install a CUDA build of PyTorch first if needed
bash scripts/setup.sh
```

`scripts/setup.sh` does the following:

- clones the official CycleNet code into `CycleNet/` and checks out commit
  `d807e51fc2dcd143885ee639d97965a7ab0926f4`;
- downloads ETTh1, ETTm1, Electricity, Traffic and Solar and verifies them against `scripts/data_sha256.txt`;
- writes the dataset files that the official loaders expect;
- downloads hourly NOAA CO-OPS water levels for 2019–2023.

The official code is never modified. `src/run_init.py` wraps its `run.py` to set the phase table before training.

## Running

```bash
UTIDE_PYTHON=/path/to/utide-env/bin/python bash scripts/run_all.sh
```

The scripts in `src/` must be run with `src/` as the working directory; `scripts/run_all.sh` does this.

- Every run, including failed ones, is appended as one JSON line to `results/runs.jsonl`, with its
  configuration, metrics, run time and SHA-256 hashes of the code that produced it.
- Logs go to `logs/`.
- `python src/summarize_runs.py [EXP ...]` writes the mean over seeds per configuration to
  `results/summary.csv`.
- Set `RCF_ROOT` to place `CycleNet/`, `data/`, `results/` and `logs/` somewhere other than the
  repository root.

Seeds are 2024–2026 for official runs unless stated otherwise. Official runs use the official hyperparameters
with `OMP_NUM_THREADS=2` and `num_workers=1` per run.

## Experiments

| ID | Script | Content |
|---|---|---|
| E1 | `closed_form.py`, `make_jobs.py`, `official_runner.py` | Closed-form Linear and RCF; official CycleNet (linear backbone, no RevIN); $L=336$, $H\in\{96,192,336,720\}$ |
| E2 | same | Table lengths $W\in\{23,24,48,96,144,168\}$, $L=H=96$; official CycleNet with and without RevIN, Linear, RLinear; closed forms |
| E3 | `e3_synth.py` | Synthetic series with non-integer periods |
| E4 | `e4_tides.py`, `make_jobs_e4.py` | Tide gauges, $L=336$: closed-form baselines; official CycleNet cross-check |
| E5, E5b | `e5_nullspace.py`, `e5b_threshold.py` | Learned phase tables of the E1 checkpoints compared with closed-form tables |
| E6 | `e6_harmless.py`, `test_resample.py` | Checks of the resampling pipeline |
| E7 | `closed_form.py` | Closed-form Linear and RCF for look-back $L\in\{48,96,168,336,720\}$ |
| E8 | `e8_tides_short.py` | Tide gauges with $L\in\{24,48,96\}$ |
| E9 | `make_jobs_s2.py` | Official CycleNet with the MLP backbone, table lengths as in E2; RMLP |
| E10, E10x | `e10_optimizer.py`, `e10x_sgd_large_lr.py` | Own training loop (Adam, SGD) for CycleNet with a linear backbone on Electricity |
| E11 | `e11_assumption.py` | Phase means of training windows compared with closed-form tables |
| E13 | `e13_nodal.py` | Harmonic regression with nodal corrections (UTide) |
| E14 | `e14_download.py`, `e14_tides.py` | E4 and E8 on two further tide gauges |
| E15 | `e15_synth.py` | E3 forecasting at small training lengths |
| E16 | `make_jobs_s2.py` | Official ETTh1 settings, seeds 2024–2028 |
| E17 | `e17_tables.py`, `make_jobs_e17.py`, `e17_runner.py`, `run_init.py` | Official CycleNet with the phase table zero-initialised, initialised at the closed-form table, or fixed at it |
| — | `diag_e4_interp.py` | Resampling round-trip error and phase-table energy on the tide gauges |

## Data

- ETTh1 and ETTm1: <https://github.com/zhouhaoyi/ETDataset>.
- Electricity, Traffic and Solar: <https://github.com/laiguokun/multivariate-time-series-data>.
- Tide gauges: NOAA CO-OPS hourly heights (<https://api.tidesandcurrents.noaa.gov/api/prod/>), stations
  9414290, 8518750, 9447130, 8443970 and 9410230.

The data are not redistributed here.

## License

MIT (see `LICENSE`). CycleNet is not included; see its repository for its terms.
