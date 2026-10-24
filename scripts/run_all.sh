#!/bin/bash
# Run every experiment in order. Run from the repository root after scripts/setup.sh.
# All results are appended to results/runs.jsonl; logs go to logs/.
# E13 needs UTide, which requires NumPy >= 2, while the official runs used NumPy 1.26.4;
# set UTIDE_PYTHON to the interpreter of a separate environment built from requirements-utide.txt.
set -eu
cd "$(dirname "$0")/../src"
L=../logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 RCF_THREADS=2 RCF_NUM_WORKERS=1
PY=${RCF_PYTHON:-python}
UTIDE_PY=${UTIDE_PYTHON:-}
$PY -c "import numpy; assert numpy.__version__.startswith('1.'), numpy.__version__"

# Closed forms and synthetic experiments
$PY e3_synth.py > $L/e3.log 2>&1
$PY closed_form.py --exp E1 --datasets ETTh1 ETTm1 Electricity Solar --L 336 --H 96 192 336 720 > $L/e1_closed.log 2>&1
$PY closed_form.py --exp E2 --datasets Electricity Traffic Solar --L 96 --H 96 --methods linear rcf \
    --W 23 24 48 96 144 168 > $L/e2_closed.log 2>&1
for Lb in 48 96 168 336 720; do
  $PY closed_form.py --exp E7 --datasets Electricity Traffic Solar ETTh1 --L $Lb --H 96 --methods linear rcf
done > $L/e7.log 2>&1
$PY e6_harmless.py > $L/e6.log 2>&1
$PY test_resample.py > $L/test_resample.log 2>&1
$PY e11_assumption.py > $L/e11.log 2>&1
$PY e15_synth.py > $L/e15.log 2>&1

# Tide gauges
$PY e4_tides.py > $L/e4_closed.log 2>&1
$PY diag_e4_interp.py > $L/diag_e4.log 2>&1
$PY e8_tides_short.py > $L/e8.log 2>&1
$PY e14_tides.py > $L/e14.log 2>&1
if [ -n "$UTIDE_PY" ]; then $UTIDE_PY e13_nodal.py > $L/e13.log 2>&1; else echo "skip E13: UTIDE_PYTHON not set"; fi

# Official CycleNet runs
$PY make_jobs.py
$PY make_jobs_e4.py
$PY make_jobs_s2.py
$PY official_runner.py ../results/jobs_e1.json --workers 8 > $L/e1_official.log 2>&1
$PY official_runner.py ../results/jobs_e2.json --workers 8 > $L/e2_official.log 2>&1
$PY official_runner.py ../results/jobs_e4.json --workers 8 > $L/e4_official.log 2>&1
$PY official_runner.py ../results/jobs_s2.json --workers 6 > $L/s2_official.log 2>&1

# Learned tables of the E1 checkpoints; own training loop
$PY e5_nullspace.py > $L/e5.log 2>&1
$PY e5b_threshold.py > $L/e5b.log 2>&1
$PY e10_optimizer.py > $L/e10.log 2>&1
$PY e10x_sgd_large_lr.py > $L/e10x.log 2>&1

# Closed-form table as initial or fixed table of official CycleNet
$PY e17_tables.py > $L/e17_tables.log 2>&1
$PY make_jobs_e17.py
$PY e17_runner.py ../results/jobs_e17_check.json --workers 3 > $L/e17_check.log 2>&1
$PY e17_runner.py ../results/jobs_e17.json --workers 6 > $L/e17.log 2>&1

$PY summarize_runs.py > $L/summary.log 2>&1
echo ALL_DONE
