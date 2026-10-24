#!/bin/bash
# Clone the official CycleNet code at the pinned commit, download the public datasets and build the files
# expected by the official loaders. Run from the repository root.
set -eu
ROOT=$(pwd)
COMMIT=d807e51fc2dcd143885ee639d97965a7ab0926f4

if [ ! -d CycleNet/.git ]; then
  git clone https://github.com/ACAT-SCUT/CycleNet.git
fi
(cd CycleNet && git checkout -q $COMMIT && git log -1 --format="CycleNet commit %H")

mkdir -p data/raw results logs
cd data/raw
B=https://raw.githubusercontent.com
get() { [ -s "$2" ] || curl -sL --retry 3 -o "$2" "$1"; }
get $B/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv ETTh1.csv
get $B/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv ETTm1.csv
get $B/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz electricity.txt.gz
get $B/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz traffic.txt.gz
get $B/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz solar_AL.txt.gz
sha256sum -c "$ROOT/scripts/data_sha256.txt"

cd "$ROOT/src"
python prep_data.py
python noaa_download.py
python e14_download.py
