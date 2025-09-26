import os
import time
import requests
import pandas as pd

# -------- CONFIG --------
OUTPUT_DIR = "usgs_data"
START_DATE = "2024-07-01"   # change as needed
END_DATE   = "2024-09-15"

SITES = [
    # Upstream gauges
    {"name": "Walnut_Creek_at_Buck_Jones_Road",    "site": "02087337",   "type": "streamflow"},   # 00060
    {"name": "Walnut_Creek_at_Trailwood_Drive",    "site": "0208734210", "type": "streamflow"},   # 00060
    {"name": "Walnut_Creek_at_South_Wilmington_St","site": "0208734795", "type": "streamflow"},   # 00060
    {"name": "Walnut_Creek_at_South_State_Street", "site": "0208735460", "type": "gage_height"},  # 00065 (🎯 target)

    # Rain ( Lake Johnson)
    {"name": "Rain_Gage_at_Lake_Johnson",          "site": "354546078422045", "type": "precip"},  # 00045
]

PARAM = {"streamflow": "00060", "gage_height": "00065", "precip": "00045"}

def fetch_usgs_iv(site: str, param: str, start: str, end: str) -> pd.DataFrame | None:
    url = (
        "https://waterservices.usgs.gov/nwis/iv/"
        f"?sites={site}&parameterCd={param}&startDT={start}&endDT={end}&format=json"
    )
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print(f" HTTP {r.status_code} for {site} {param}")
        return None
    try:
        ts = r.json()["value"]["timeSeries"]
        if not ts: 
            print(f" No timeSeries for {site} {param}")
            return None
        vals = ts[0]["values"][0]["value"]
        if not vals:
            print(f" Empty values for {site} {param}")
            return None
        df = pd.DataFrame([(v["dateTime"], v["value"]) for v in vals], columns=["datetime", "value"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df
    except Exception as e:
        print(f" Parse issue for {site} {param}: {e}")
        return None

def safe_filename(name: str, site: str) -> str:
    return f"{name}_{site}.csv".replace(" ", "_").replace(",", "")

# -------- 1) DOWNLOAD --------
os.makedirs(OUTPUT_DIR, exist_ok=True)
seen = set()

for s in SITES:
    site_id = s["site"].strip()
    if site_id in seen:
        print(f"  Skipping duplicate site {site_id}")
        continue
    seen.add(site_id)

    param = PARAM[s["type"]]
    name  = s["name"].replace(" ", "_")
    print(f"  Fetching {name} ({site_id}) param {param} ...")
    df = fetch_usgs_iv(site_id, param, START_DATE, END_DATE)
    if df is None:
        print(f" Skipped {name} ({site_id})")
        continue
    fp = os.path.join(OUTPUT_DIR, safe_filename(name, site_id))
    if os.path.exists(fp):
        try: os.remove(fp)
        except PermissionError: print(f" Close {fp} and re-run.")
    df.to_csv(fp, index=False)
    print(f" Saved {fp}  ({len(df):,} rows)")
    time.sleep(0.2)

# -------- 2) MERGE → HOURLY --------
frames = []
for fname in os.listdir(OUTPUT_DIR):
    if not fname.endswith(".csv"): 
        continue
    path = os.path.join(OUTPUT_DIR, fname)
    df = pd.read_csv(path, parse_dates=["datetime"])
    if "value" not in df.columns:
        print(f" No 'value' column in {fname}; skipping")
        continue
    colname = fname.replace(".csv","")    # unique by filename
    frames.append(df.set_index("datetime").sort_index().rename(columns={"value": colname}))

if not frames:
    raise SystemExit(" No data frames to merge. Check downloads.")

combined = pd.concat(frames, axis=1)
combined = combined.loc[:, ~combined.columns.duplicated()]           # drop dup columns if any
combined_hourly = combined.resample("1H").mean()                      # hourly mean
combined_hourly = combined_hourly.dropna(axis=1, how="all")           # drop dead cols

combined_hourly.to_csv("combined_usgs_hourly.csv")
print(f"\n combined_usgs_hourly.csv  (rows={len(combined_hourly):,}, cols={combined_hourly.shape[1]})")
print("Columns:")
for c in combined_hourly.columns: print(" -", c)
