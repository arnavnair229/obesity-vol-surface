"""
Did the rise in self-reported obesity break, and does the pattern match the GLP-1 story?

CDC's national figure changes with which states report: California, Mississippi and Nevada are absent in 2025.
So the main series is the mean of the states present in every year. Year effects from a state fixed-effects
fit on all states serve as a check, since they tolerate missing states.

The break date is searched over, not assumed, because 2021 was chosen after seeing the data. With 15 annual
points this test has little power, and the 2020-2021 jump overlaps the pandemic.

Outputs:
  - outputs/trend_break.csv
  - outputs/trend_break_checks.csv
  - outputs/age_changes.csv
  - outputs/three_measures.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "outputs"
EXCLUDE = ["US", "GU", "PR", "VI"]
KNOTS = range(2014, 2023)  # at least 3 years on each side
AGES = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
MIDDLE = ["35-44", "45-54", "55-64"]
WINDOWS = {"2022-2025": (2022, 2025), "2021-2025": (2021, 2025), "placebo 2018-2021": (2018, 2021)}


def hinge_f(years, y):
    """F statistic per knot for adding a change in slope to a straight line. Knots keep 3 years on each side."""
    X0 = np.column_stack([np.ones_like(years, dtype=float), years])
    rss0 = np.sum((y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]) ** 2)
    out = {}
    for knot in (k for k in KNOTS if k <= years.max() - 3):
        X1 = np.column_stack([X0, np.maximum(years - knot, 0)])
        rss1 = np.sum((y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]) ** 2)
        out[knot] = (rss0 - rss1) / (rss1 / (len(y) - 3))
    return out


def sup_f_test(years, y, n=10000, seed=7):
    """Compare the largest F with its distribution under a straight line, resampling residuals."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones_like(years, dtype=float), years])
    fit = X @ np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - fit
    fs = hinge_f(years, y)
    observed = max(fs.values())
    draws = np.array([max(hinge_f(years, fit + rng.choice(resid, len(y))).values()) for _ in range(n)])
    knot = max(fs, key=fs.get)
    X1 = np.column_stack([X, np.maximum(years - knot, 0)])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    return {"sup_f": observed, "p": float((draws >= observed).mean()), "best_knot": knot, "slope_before": beta[1], "slope_after": beta[1] + beta[2]}


def slope_after_jump(years, y, knot=2021):
    """Let the level jump at the knot, as it did in 2021, then test whether the slope still changes."""
    after = years >= knot
    X = np.column_stack([np.ones_like(years, dtype=float), years, after.astype(float), np.where(after, years - knot, 0.0)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = np.sum(resid ** 2) / dof * np.linalg.inv(X.T @ X)
    t = beta[3] / np.sqrt(cov[3, 3])
    return {"jump": beta[2], "slope_before": beta[1], "slope_after": beta[1] + beta[3], "t_slope_change": t, "p_slope_change": 2 * stats.t.sf(abs(t), dof)}


def extrapolation(years, y, last_train=2020):
    train = years <= last_train
    fit = stats.linregress(years[train], y[train])
    t_crit = stats.t.ppf(0.975, train.sum() - 2)
    resid = y[train] - (fit.intercept + fit.slope * years[train])
    s = np.sqrt(np.sum(resid ** 2) / (train.sum() - 2))
    rows = []
    for year, actual in zip(years[~train], y[~train]):
        se = s * np.sqrt(1 + 1 / train.sum() + (year - years[train].mean()) ** 2 / np.sum((years[train] - years[train].mean()) ** 2))
        trend = fit.intercept + fit.slope * year
        rows.append({"year": year, "trend": trend, "actual": actual, "low": trend - t_crit * se, "high": trend + t_crit * se})
    return pd.DataFrame(rows), fit


def age_changes(latest):
    states = latest[~latest.location.isin(EXCLUDE) & latest.group.isin(AGES)]
    cube = states.pivot_table(index="location", columns=["group", "year"], values="percent")
    cube = cube.dropna()
    rng = np.random.default_rng(7)
    rows = []
    for label, (a, b) in WINDOWS.items():
        delta = pd.DataFrame({age: cube[age][b] - cube[age][a] for age in AGES})
        delta["middle_minus_others"] = delta[MIDDLE].mean(axis=1) - delta[[g for g in AGES if g not in MIDDLE]].mean(axis=1)
        for col in delta.columns:
            values = delta[col].to_numpy()
            draws = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(5000)]
            low, high = np.percentile(draws, [2.5, 97.5])
            rows.append({"window": label, "group": col, "change_points": values.mean(), "low": low, "high": high, "states": len(values)})
    return pd.DataFrame(rows)


def figure(series, gallup, nhanes, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(series.index, series.values, color="#0f6cbd", lw=2, marker="o", ms=5, label="BRFSS, self-report, phone (fixed set of states)")
    ax.errorbar((nhanes.period_start + nhanes.period_end) / 2, nhanes.percent, yerr=1.96 * nhanes.se, color="#c2410c", marker="s", ms=8, lw=2, capsize=3,
                ls="none", label="NHANES, measured (age 20+, age-adjusted)")
    ax.errorbar(gallup.period_start, gallup.percent, yerr=gallup.moe_95, color="#7c3aed", marker="^", ms=8, lw=2, capsize=3, label="Gallup, self-report, web")
    ax.set(xlabel="Year", ylabel="Adult obesity (%)", ylim=(25, 46), title="Three measures of US adult obesity disagree on level and direction")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e7eb", lw=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    latest = pd.read_csv(RAW / "cdc_brfss_latest.csv")
    total = latest[latest.group == "Total"]
    states = total[~total.location.isin(EXCLUDE)]
    wide = states.pivot(index="location", columns="year", values="percent")
    balanced = wide.dropna()
    series = balanced.mean()
    effects = smf.ols("percent ~ C(location) + C(year)", states).fit().params.filter(like="C(year)")
    all_states = pd.Series([0.0] + list(effects.to_numpy()), index=sorted(states.year.unique())) + wide.loc[balanced.index, sorted(states.year.unique())[0]].mean()
    us = total[total.location == "US"].set_index("year").percent

    rows = []
    for label, s in (("fixed set of states", series), ("all states, year effects", all_states), ("CDC national figure", us)):
        years, y = s.index.to_numpy(dtype=float), s.to_numpy()
        result = sup_f_test(years, y)
        path, fit = extrapolation(years, y)
        last = path.iloc[-1]
        rows.append({"series": label, **result, "trend_2011_2020_per_year": fit.slope, "trend_for_2025": last.trend, "low_2025": last.low, "high_2025": last.high,
                     "actual_2025": last.actual, "first_above": s.index[s.index >= 2021][0], "change_2021_2025": s[2025] - s[2021], "change_2022_2025": s[2025] - s[2022]})
    results = pd.DataFrame(rows)
    results.round(4).to_csv(OUT / "trend_break.csv", index=False)

    years, y = series.index.to_numpy(dtype=float), series.to_numpy()
    no_2025 = series.index < 2025
    no_jump = series.index != 2021
    checks = {"through 2024": sup_f_test(years[no_2025], y[no_2025]), "without 2021": sup_f_test(years[no_jump], y[no_jump])}
    checks = pd.DataFrame(checks).T.rename_axis("check").reset_index()
    jump = slope_after_jump(years, y)
    pd.concat([checks, pd.DataFrame([{"check": "level jump allowed at 2021", **jump}])]).round(4).to_csv(OUT / "trend_break_checks.csv", index=False)

    path, _ = extrapolation(series.index.to_numpy(dtype=float), series.to_numpy())
    ages = age_changes(latest)
    ages.round(3).to_csv(OUT / "age_changes.csv", index=False)

    bench = pd.read_csv(RAW / "external_obesity_benchmarks.csv")
    figure(series, bench[bench.source == "Gallup"], bench[bench.source == "NHANES"], OUT / "three_measures.png")

    change4 = series.diff(4).dropna()
    slope = lambda col, a, b: np.polyfit(range(a, b + 1), col.loc[a:b].to_numpy(), 1)[0]
    pre = balanced.apply(lambda c: slope(c, 2011, 2020), axis=1)
    post = balanced.apply(lambda c: slope(c, 2021, 2025), axis=1)

    pd.set_option("display.width", 220)
    print(f"Fixed set: {len(balanced)} states; left out: {sorted(set(wide.index) - set(balanced.index))}")
    print(series.round(2).to_dict())
    print(results.round(3).T.to_string())
    print(checks.round(3).to_string(index=False))
    print("Level jump allowed:", {k: round(float(v), 3) for k, v in jump.items()})
    print(path.round(2).to_string(index=False))
    print("F by knot", {k: round(v, 1) for k, v in hinge_f(series.index.to_numpy(dtype=float), series.to_numpy()).items()})
    print("4-year changes", change4.round(2).to_dict())
    print(f"States with a flatter slope after 2021: {(post < pre).sum()} of {len(pre)}; mean before {pre.mean():.2f}, after {post.mean():.2f}")
    print(ages.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
