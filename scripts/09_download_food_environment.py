"""
Download state-level food establishment counts and population from the Census Bureau.

County Business Patterns counts by state, all legal forms of organization:
  722513 limited-service restaurants, 722511 full-service restaurants, 445110 grocery.
The flat files are used because the CBP API was unreliable when this was written.

Outputs:
  - data/raw/census_cbp_food_establishments.csv
  - data/raw/census_state_population.csv
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
CBP = "https://www2.census.gov/programs-surveys/cbp/datasets/{y}/cbp{yy}st.zip"
POP = {
    "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv": range(2012, 2020),
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/totals/NST-EST2024-ALLDATA.csv": range(2020, 2024),
}
KINDS = {"722513": "limited", "722511": "fullservice", "445110": "grocery"}
YEARS = range(2012, 2024)


def fetch(url):
    response = requests.get(url, timeout=180, headers={"User-Agent": "obesity-vol-surface data download"})
    response.raise_for_status()
    return response.content


def establishments():
    rows = []
    for y in YEARS:
        with zipfile.ZipFile(io.BytesIO(fetch(CBP.format(y=y, yy=str(y)[2:])))) as archive:
            data = pd.read_csv(archive.open(archive.namelist()[0]), dtype=str,
                               usecols=lambda c: c.lower() in ("fipstate", "naics", "lfo", "est"))
        data.columns = [c.lower() for c in data.columns]
        data = data[data.naics.isin(KINDS) & (data.lfo == "-")]
        rows.append(pd.DataFrame({"year": y, "fips": data.fipstate.str.zfill(2),
                                  "kind": data.naics.map(KINDS), "establishments": data.est.astype(int)}))
    return pd.concat(rows, ignore_index=True)


def population():
    rows = []
    for url, years in POP.items():
        data = pd.read_csv(io.BytesIO(fetch(url)), dtype={"STATE": str})
        data = data[data.SUMLEV == 40]
        for y in years:
            rows.append(pd.DataFrame({"year": y, "fips": data.STATE.str.zfill(2), "population": data[f"POPESTIMATE{y}"]}))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    est = establishments()
    pop = population()
    est.sort_values(["year", "fips", "kind"]).to_csv(RAW / "census_cbp_food_establishments.csv", index=False)
    pop.sort_values(["year", "fips"]).to_csv(RAW / "census_state_population.csv", index=False)
    print(f"{len(est)} establishment rows, {len(pop)} population rows")
