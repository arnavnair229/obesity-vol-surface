"""
Download CDC BRFSS Obesity Prevalence Data (US + States, Annual)

Source: CDC Nutrition, Physical Activity, and Obesity - BRFSS
Dataset ID: hn4x-zwk7 (Socrata)
"""

import pandas as pd
import requests
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

DATASET = "hn4x-zwk7"
BASE = f"https://data.cdc.gov/resource/{DATASET}.json"

QUESTION = "Percent of adults aged 18 years and older who have obesity"
CLASS = "Obesity / Weight Status"

params = {
    "$select": ",".join([
        "yearstart",
        "locationabbr",
        "locationdesc",
        "class",
        "question",
        "data_value",
        "data_value_unit",
        "stratificationcategory1",
        "stratification1",
    ]),
    "$where": (
        f"class = '{CLASS}' AND "
        f"question = '{QUESTION}' AND "
        "stratificationcategory1 = 'Total' AND "
        "stratification1 = 'Total'"
    ),
    "$limit": 50000,
}

print("Requesting:", BASE)
r = requests.get(BASE, params=params, timeout=60)
r.raise_for_status()
rows = r.json()
df = pd.DataFrame(rows)

print(f"Raw rows received: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Basic cleanup
df["yearstart"] = df["yearstart"].astype(int)
df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")

# data_value is in percent (0-100). Convert to fraction (0-1).
df["p"] = df["data_value"] / 100.0

# Keep only usable rows
df = df.dropna(subset=["p"])
df = df[(df["p"] > 0) & (df["p"] < 1)]

# Save
out_path = OUT / "cdc_brfss_obesity_total_us_states.csv"
df.sort_values(["locationabbr", "yearstart"]).to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Rows: {len(df)}")
print(f"Years: {df['yearstart'].min()} - {df['yearstart'].max()}")
print(f"Locations: {df['locationabbr'].nunique()}")
print(f"\nSample:")
print(df.head(10).to_string(index=False))
