"""
Score the frozen models on 2025, the first outcome year nothing here was built or fixed with.

Every fit uses the same data and code as 08_backtest.py, and only pairs known by the forecast date.
All of these forecasts share one outcome year, so one national shock can move every score. The interval
resamples states and doesn't capture that.

Output: outputs/holdout_2025.csv
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs"
spec = importlib.util.spec_from_file_location("backtest", PROJECT / "scripts" / "08_backtest.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)
EXCLUDE = ["US", "GU", "PR", "VI"]
SHOW = ["logistic_old", "logistic_fixed", "random_walk", "persistence"]


def main():
    frozen = pd.read_csv(PROJECT / "data" / "raw" / "cdc_brfss_obesity_total_us_states.csv")
    frozen = frozen[~frozen.locationabbr.isin(EXCLUDE)].pivot(index="locationabbr", columns="yearstart", values="p")
    latest = pd.read_csv(PROJECT / "data" / "raw" / "cdc_brfss_latest.csv")
    latest = latest[(latest.group == "Total") & ~latest.location.isin(EXCLUDE)].pivot(index="location", columns="year", values="percent") / 100

    shared = latest.reindex(index=frozen.index, columns=frozen.columns)
    if not np.allclose(frozen, shared, equal_nan=True):
        raise SystemExit("CDC revised earlier years, so the frozen data no longer matches")

    rows, rng = [], np.random.default_rng(7)
    for T in bt.HORIZONS:
        origin = 2025 - T
        train = bt.training_pairs(frozen, T, origin)
        test = pd.concat([frozen[origin].rename("start"), latest[2025].rename("end")], axis=1).dropna()
        preds = bt.forecast(train, test.start.to_numpy())
        change = bt.logit(train[:, 2]) - bt.logit(train[:, 1])
        level = np.exp(bt.logit(test.start) + change.mean())
        level = level / (1 + level)
        hit = np.column_stack([(test.end >= k).astype(float) for k in bt.STRIKES])
        near = np.abs(test.start.to_numpy()[:, None] - np.array(bt.STRIKES)) <= bt.NEAR_MONEY
        row = {"T": T, "origin": origin, "states": len(test), "cells": hit.size, "contested_cells": int(near.sum()),
               "drift_forecast_mean_pct": level.mean() * 100, "actual_mean_pct": test.end.mean() * 100}
        for name in SHOW:
            err = (preds[name] - hit) ** 2
            row[f"brier_{name}"] = err.mean()
            row[f"brier_contested_{name}"] = err[near].mean()
        row["contested_actual"] = hit[near].mean()
        row["contested_forecast_fixed"] = preds["logistic_fixed"][near].mean()
        row["contested_forecast_rw"] = preds["random_walk"][near].mean()
        for label, name in (("old", "logistic_old"), ("fixed", "logistic_fixed")):
            by_state = ((preds[name] - hit) ** 2 - (preds["random_walk"] - hit) ** 2).mean(axis=1)
            draws = [by_state[rng.integers(0, len(by_state), len(by_state))].mean() for _ in range(5000)]
            row[f"{label}_minus_rw_low"], row[f"{label}_minus_rw_high"] = np.percentile(draws, [2.5, 97.5])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(OUT / "holdout_2025.csv", index=False)
    print(out.round(3).T.to_string())


if __name__ == "__main__":
    main()
