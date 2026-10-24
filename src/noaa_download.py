"""Download NOAA CO-OPS verified hourly heights (MSL, metric, GMT), 2019-2023, one year per request."""
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

from common import ROOT

OUT = f"{ROOT}/data/noaa"
PRIMARY = ["9414290", "8518750", "9447130"]
RESERVE = ["8443970", "9410230"]
YEARS = range(2019, 2024)
API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
META = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{}.json"


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rcf-experiments"})
            return urllib.request.urlopen(req, timeout=120).read().decode()
        except Exception as e:
            err = e
            time.sleep(5 * (i + 1))
    raise err


def station(sid):
    meta = json.loads(get(META.format(sid)))
    st = meta["stations"][0]
    frames = []
    for y in YEARS:
        url = (f"{API}?begin_date={y}0101&end_date={y}1231 23:00&station={sid}&product=hourly_height"
               f"&datum=MSL&units=metric&time_zone=gmt&format=json&application=rcf_experiments").replace(" ", "%20")
        js = json.loads(get(url))
        if "data" not in js:
            raise RuntimeError(f"{sid} {y}: {js.get('error')}")
        frames.append(pd.DataFrame(js["data"]))
    df = pd.concat(frames)
    df["t"] = pd.to_datetime(df["t"])
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    full = pd.date_range("2019-01-01 00:00", "2023-12-31 23:00", freq="h")
    s = df.drop_duplicates("t").set_index("t")["v"].reindex(full)
    miss = float(s.isna().mean())
    longest = int(s.isna().astype(int).groupby(s.notna().cumsum()).sum().max())
    filled = s.interpolate(limit_direction="both")
    out = pd.DataFrame({"date": full.strftime("%Y-%m-%d %H:%M:%S"), "OT": filled.to_numpy()})
    out.to_csv(f"{OUT}/{sid}.csv", index=False)
    info = {"id": sid, "name": st.get("name"), "state": st.get("state"), "lat": st.get("lat"), "lng": st.get("lng"),
            "n": len(full), "missing_frac": miss, "longest_gap_h": longest}
    return info


def main():
    os.makedirs(OUT, exist_ok=True)
    chosen, log = [], []
    for sid in PRIMARY + RESERVE:
        if len(chosen) == 3:
            break
        try:
            info = station(sid)
        except Exception as e:
            log.append({"id": sid, "error": repr(e)})
            print(sid, "ERROR", e, flush=True)
            continue
        ok = info["missing_frac"] <= 0.02
        info["accepted"] = ok
        log.append(info)
        print(info, flush=True)
        if ok:
            chosen.append(sid)
    json.dump({"chosen": chosen, "log": log}, open(f"{OUT}/stations.json", "w"), indent=1)
    print("chosen", chosen)


if __name__ == "__main__":
    main()
