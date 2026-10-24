"""E14: download the two reserve stations.
Writes data/noaa/stations_extra.json; stations.json (used by E4) is not touched."""
import json

from noaa_download import OUT, RESERVE, station


def main():
    chosen, log = [], []
    for sid in RESERVE:
        try:
            info = station(sid)
        except Exception as e:
            log.append({"id": sid, "error": repr(e)})
            print(sid, "ERROR", e, flush=True)
            continue
        info["accepted"] = info["missing_frac"] <= 0.02
        log.append(info)
        print(info, flush=True)
        if info["accepted"]:
            chosen.append(sid)
    json.dump({"chosen": chosen, "log": log}, open(f"{OUT}/stations_extra.json", "w"), indent=1)
    print("chosen", chosen)


if __name__ == "__main__":
    main()
