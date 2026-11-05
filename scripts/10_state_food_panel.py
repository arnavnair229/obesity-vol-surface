"""
Does a food-environment measure that varies by state track obesity once national trends are removed?

The exposure index in 06_panel_regression.py is one national number per year, so year fixed effects
would absorb it. Limited-service restaurant density differs by state, so it can be tested with them.

Coefficients are percentage points of obesity per log unit of density. Errors are clustered by state.
Lag, lead and contemporaneous specifications are also run on one common sample, so a difference
between them can't come from different years entering the fit.

Output: outputs/state_food_panel_results.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "outputs"

FIPS = {"AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "DC": "11",
        "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
        "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30",
        "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
        "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
        "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"}
FE = "C(state) + C(year)"


def load():
    obesity = pd.read_csv(RAW / "cdc_brfss_obesity_total_us_states.csv")[["yearstart", "locationabbr", "p"]]
    obesity = obesity.rename(columns={"yearstart": "year", "locationabbr": "state"})
    obesity = obesity[obesity.state.isin(FIPS)]
    obesity["fips"] = obesity.state.map(FIPS)
    obesity["obesity"] = obesity.p * 100

    est = pd.read_csv(RAW / "census_cbp_food_establishments.csv", dtype={"fips": str})
    est = est.pivot_table(index=["year", "fips"], columns="kind", values="establishments").reset_index()
    pop = pd.read_csv(RAW / "census_state_population.csv", dtype={"fips": str})
    income = pd.read_csv(RAW / "census_state_income.csv").rename(columns={"locationabbr": "state"})
    other = pd.read_csv(RAW / "cdc_brfss_state_panel.csv").rename(columns={"yearstart": "year", "locationabbr": "state"})

    d = (obesity.merge(est, on=["year", "fips"]).merge(pop, on=["year", "fips"])
         .merge(income, on=["year", "state"], how="left")
         .merge(other[["year", "state", "pct_inactive"]], on=["year", "state"], how="left"))
    for kind in ("limited", "fullservice", "grocery"):
        d[kind] = np.log(d[kind] / d.population * 1e5)
    d["income"] = np.log(d.median_hh_income)

    # Shift on a complete state-year grid so a missing survey year can't turn into a wrong lag.
    grid = pd.MultiIndex.from_product([d.state.unique(), range(d.year.min(), d.year.max() + 1)], names=["state", "year"])
    lim = d.set_index(["state", "year"]).limited.reindex(grid)
    for name, shift in (("lag2", 2), ("lag4", 4), ("lead2", -2)):
        d[name] = d.set_index(["state", "year"]).index.map(lim.groupby(level="state").shift(shift))
    return d


def fit(label, formula, data, sample="full"):
    variables = [v for v in ("obesity", "limited", "fullservice", "grocery", "income", "pct_inactive", "lag2", "lag4", "lead2")
                 if v in formula]
    frame = data.dropna(subset=variables)
    model = smf.ols(formula, frame).fit(cov_type="cluster", cov_kwds={"groups": frame.state.astype("category").cat.codes})
    return [{"spec": label, "sample": sample, "term": term, "coef": model.params[term], "t": model.tvalues[term],
             "p": model.pvalues[term], "n": int(model.nobs), "states": frame.state.nunique()}
            for term in model.params.index if not term.startswith(("C(", "Intercept"))]


def long_difference(d):
    first, last = d.year.min(), d.year.max()
    a, z = d[d.year == first].set_index("state"), d[d.year == last].set_index("state")
    ld = pd.DataFrame({"obesity": z.obesity - a.obesity, "limited": z.limited - a.limited, "income": z.income - a.income}).dropna()
    model = smf.ols("obesity ~ limited + income", ld).fit(cov_type="HC1")
    return [{"spec": f"long difference {first}-{last}, robust errors", "sample": "full", "term": term, "coef": model.params[term],
             "t": model.tvalues[term], "p": model.pvalues[term], "n": int(model.nobs), "states": len(ld)}
            for term in ("limited", "income")]


def main():
    d = load()
    rows = []
    rows += fit("pooled, no fixed effects", "obesity ~ limited", d)
    rows += fit("state effects only", "obesity ~ limited + C(state)", d)
    rows += fit("state and year effects", f"obesity ~ limited + {FE}", d)
    rows += fit("with full-service and grocery", f"obesity ~ limited + fullservice + grocery + {FE}", d)
    rows += fit("with income and inactivity", f"obesity ~ limited + income + pct_inactive + {FE}", d)
    rows += fit("density 2y earlier", f"obesity ~ lag2 + {FE}", d)
    rows += fit("density 4y earlier", f"obesity ~ lag4 + {FE}", d)
    rows += fit("placebo: density 2y later", f"obesity ~ lead2 + {FE}", d)

    common = d.dropna(subset=["lag2", "lead2"])
    for label, term in (("density 2y earlier", "lag2"), ("same year", "limited"), ("placebo: density 2y later", "lead2")):
        rows += fit(label, f"obesity ~ {term} + {FE}", common, sample="common")
    rows += fit("all three together", f"obesity ~ lag2 + limited + lead2 + {FE}", common, sample="common")
    rows += long_difference(d)

    results = pd.DataFrame(rows)
    results.round(4).to_csv(OUT / "state_food_panel_results.csv", index=False)
    print(f"{len(d)} state-years, {d.state.nunique()} states, {d.year.min()}-{d.year.max()}")
    print(f"Common sample: {common.year.min()}-{common.year.max()}, n={len(common)}")
    print(results.round(3).to_string(index=False))
    a, z = d[d.year == d.year.min()].set_index("state"), d[d.year == d.year.max()].set_index("state")
    print("corr(change in obesity, change in density):", round((z.obesity - a.obesity).corr(z.limited - a.limited), 3))


if __name__ == "__main__":
    main()
