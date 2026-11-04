"""
Rebuild the surface from the raw CDC JSON: writes
data/raw/cdc_brfss_obesity_total_us_states.csv, then runs 02_build_surface.py.
"""

import json
import pandas as pd
import runpy
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / "data" / "raw"
OUT_DIR = PROJECT / "outputs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load raw JSON ──
raw_path = RAW_DIR / "cdc_brfss_raw.json"
with open(raw_path) as f:
    rows = json.load(f)

df = pd.DataFrame(rows)
df["yearstart"] = df["yearstart"].astype(int)
df["data_value"] = pd.to_numeric(df.get("data_value"), errors="coerce")
df["p"] = df["data_value"] / 100.0
df = df.dropna(subset=["p"])
df = df[(df["p"] > 0) & (df["p"] < 1)]

csv_path = RAW_DIR / "cdc_brfss_obesity_total_us_states.csv"
df.sort_values(["locationabbr", "yearstart"]).to_csv(csv_path, index=False)
print(f"Saved CSV: {csv_path} ({len(df)} rows)")
print(f"Years: {df['yearstart'].min()}-{df['yearstart'].max()}")
print(f"Locations: {df['locationabbr'].nunique()}")

runpy.run_path(str(PROJECT / "scripts" / "02_build_surface.py"), run_name="__main__")
