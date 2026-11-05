"""
Download the health context for 16_options_lens.py.

Outputs:
  - data/raw/cdc_brfss_behaviors.csv: fruit, vegetables, inactivity and overweight by state
  - data/raw/census_adults_by_age.csv: 2024 civilian adults by state and CDC age band
"""

import io
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "obesity-vol-surface data download"}
QUESTIONS = {
    "Percent of adults who report consuming fruit less than one time daily": "low_fruit",
    "Percent of adults who report consuming vegetables less than one time daily": "low_vegetables",
    "Percent of adults who engage in no leisure-time physical activity": "inactive",
    "Percent of adults aged 18 years and older who have an overweight classification": "overweight",
}
CENSUS = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/asrh/sc-est2024-agesex-civ.csv"
BANDS = [(18, 24, "18-24"), (25, 34, "25-34"), (35, 44, "35-44"), (45, 54, "45-54"), (55, 64, "55-64"), (65, 85, "65+")]
STATE_CODES = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA",
               "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA",
               "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY",
               "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX",
               "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"}


def behaviors():
    params = {
        "$select": "yearstart,locationabbr,question,data_value",
        "$where": "stratificationcategory1 = 'Total' AND question in (" + ", ".join(f"'{q}'" for q in QUESTIONS) + ")",
        "$limit": 50000,
    }
    response = requests.get("https://data.cdc.gov/resource/hn4x-zwk7.json", params=params, timeout=120, headers=HEADERS)
    response.raise_for_status()
    d = pd.DataFrame(response.json())
    d["measure"] = d.question.map(QUESTIONS)
    d["percent"] = pd.to_numeric(d.data_value, errors="coerce")
    d["year"] = d.yearstart.astype(int)
    d = d.dropna(subset=["percent"]).rename(columns={"locationabbr": "location"})
    return d[["year", "location", "measure", "percent"]].sort_values(["measure", "location", "year"])


def adults():
    response = requests.get(CENSUS, timeout=180, headers=HEADERS)
    response.raise_for_status()
    d = pd.read_csv(io.BytesIO(response.content), dtype={"STATE": str})
    d = d[(d.SUMLEV == 40) & (d.SEX == 0) & d.AGE.between(18, 85)]
    d["state"] = d.STATE.str.zfill(2).map(STATE_CODES)
    d = d.dropna(subset=["state"])
    d["group"] = pd.cut(d.AGE, [b[0] - 1 for b in BANDS] + [85], labels=[b[2] for b in BANDS])
    out = d.groupby(["state", "group"], observed=True).POPEST2024_CIV.sum().rename("adults").reset_index()
    return out


if __name__ == "__main__":
    b = behaviors()
    a = adults()
    b.to_csv(RAW / "cdc_brfss_behaviors.csv", index=False)
    a.to_csv(RAW / "census_adults_by_age.csv", index=False)
    print(f"{len(b)} behavior rows, {len(a)} adult rows, {a.adults.sum() / 1e6:.1f}M adults")
