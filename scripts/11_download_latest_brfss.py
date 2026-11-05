"""
Download the latest CDC BRFSS obesity data, overall and by age, with confidence limits.

This is a separate file on purpose. The surface and 08_backtest.py stay frozen on data through 2024,
so 2025 can be scored as a year none of them saw.

Output: data/raw/cdc_brfss_latest.csv
"""

from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "cdc_brfss_latest.csv"
BASE = "https://data.cdc.gov/resource/hn4x-zwk7.json"
QUESTION = "Percent of adults aged 18 years and older who have obesity"
FIELDS = "yearstart,locationabbr,stratificationcategory1,stratification1,data_value,low_confidence_limit,high_confidence_limit,sample_size"

params = {
    "$select": FIELDS,
    "$where": f"question = '{QUESTION}' AND stratificationcategory1 in ('Total', 'Age (years)')",
    "$limit": 50000,
}
response = requests.get(BASE, params=params, timeout=120, headers={"User-Agent": "obesity-vol-surface data download"})
response.raise_for_status()
df = pd.DataFrame(response.json()).rename(columns={
    "yearstart": "year", "locationabbr": "location", "stratification1": "group", "data_value": "percent",
    "low_confidence_limit": "low", "high_confidence_limit": "high"})
df["group"] = df.group.str.replace(" ", "", regex=False).replace({"65orolder": "65+"})
df = df.drop(columns="stratificationcategory1")
for column in ("year", "percent", "low", "high", "sample_size"):
    df[column] = pd.to_numeric(df[column], errors="coerce")
df = df.dropna(subset=["percent"]).sort_values(["location", "group", "year"])
df.to_csv(OUT, index=False)
print(f"{len(df)} rows, years {df.year.min()}-{df.year.max()}")
