"""
Would the surface's crossing probabilities have come true?

Walk-forward test: at each forecast origin, fit only on (t, t+T) pairs whose outcome was already
known, then forecast each state's crossing of each strike T years later. Nothing is tuned.

logistic_old keeps the C=1 penalty the surface used to have, and logistic_fixed is the unpenalized version
of it, so the size of each error stays on record. gaussian is the one-curve model the surface uses now.

Forecasts in one origin share national shocks and overlap across origins, so the row count overstates
the evidence. Intervals resample origins and states independently (a pigeonhole bootstrap).

Outputs:
  - outputs/backtest_scores.csv
  - outputs/backtest_calibration.csv
  - outputs/backtest_by_origin.csv
  - outputs/backtest_national.csv
  - outputs/model_comparison_today.csv
  - outputs/backtest_calibration.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs"

HORIZONS = [1, 2, 3, 5]
STRIKES = [0.25, 0.28, 0.30, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.40]  # the surface grid up to 40%
MIN_TRAIN_YEARS = 2
NEAR_MONEY = 0.03  # a cell is contested when the state starts within 3 points of the strike
BINS = [0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1]
MODELS = ["logistic_old", "logistic_fixed", "gaussian", "random_walk", "random_walk_recent", "persistence", "climatology"]


def logit(p):
    return np.log(p / (1 - p))


def monotone(probs):
    """Crossing probability must fall as the strike rises, as in 02_build_surface.py."""
    return np.minimum.accumulate(probs, axis=1)


def fit_logistic(x_train, y_train, x_test, c):
    if y_train.min() == y_train.max():
        return np.full(len(x_test), float(y_train[0]))
    model = LogisticRegression(solver="lbfgs", max_iter=5000, C=c)
    return model.fit(x_train.reshape(-1, 1), y_train).predict_proba(x_test.reshape(-1, 1))[:, 1]


def forecast(train, level, strikes=STRIKES):
    """Every model's P(crossing) for each starting level and strike. train = rows of (year, p_t, p_t+T)."""
    start, end = train[:, 1], train[:, 2]
    change = logit(end) - logit(start)
    mu, sd = change.mean(), change.std(ddof=1)
    recent = train[:, 0] >= train[:, 0].max() - 2
    mu_recent = change[recent].mean()
    # One curve per horizon: logit(p_t+T) = a + b logit(p_t) + normal error.
    b, a = np.polyfit(logit(start), logit(end), 1)
    s = (logit(end) - (a + b * logit(start))).std(ddof=2)

    out = {name: np.zeros((len(level), len(strikes))) for name in MODELS}
    for j, k in enumerate(strikes):
        hit = (end >= k).astype(int)
        out["logistic_old"][:, j] = fit_logistic(logit(start), hit, logit(level), 1.0)
        out["logistic_fixed"][:, j] = fit_logistic(logit(start), hit, logit(level), 1e6)
        out["gaussian"][:, j] = norm.cdf((a + b * logit(level) - logit(k)) / s)
        out["random_walk"][:, j] = norm.cdf((logit(level) + mu - logit(k)) / sd)
        out["random_walk_recent"][:, j] = norm.cdf((logit(level) + mu_recent - logit(k)) / sd)
        out["persistence"][:, j] = (level >= k).astype(float)
        out["climatology"][:, j] = hit.mean()
    out["logistic_old"] = monotone(out["logistic_old"])
    out["logistic_fixed"] = monotone(out["logistic_fixed"])
    return out


def training_pairs(wide, T, origin):
    rows = []
    for year in wide.columns:
        if year + T <= origin and year + T in wide.columns:
            pair = wide[[year, year + T]].dropna().to_numpy()
            rows.append(np.column_stack([np.full(len(pair), year), pair]))
    return np.vstack(rows) if rows else np.empty((0, 3))


def run_state_backtest(wide):
    rows = []
    for T in HORIZONS:
        for origin in wide.columns:
            if origin + T not in wide.columns:
                continue
            train = training_pairs(wide, T, origin)
            if len(np.unique(train[:, 0])) < MIN_TRAIN_YEARS:
                continue
            test = wide[[origin, origin + T]].dropna()
            level, later = test[origin].to_numpy(), test[origin + T].to_numpy()
            preds = forecast(train, level)
            for j, k in enumerate(STRIKES):
                frame = pd.DataFrame({"T": T, "origin": origin, "K": k, "state": test.index,
                                      "start": level, "hit": (later >= k).astype(float)})
                for name in MODELS:
                    frame[name] = preds[name][:, j]
                rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def pigeonhole_ci(frame, a, b, n=2000, seed=7):
    """Interval for mean Brier(a) - Brier(b), resampling origins and states independently."""
    rng = np.random.default_rng(seed)
    diff = (frame[a] - frame.hit) ** 2 - (frame[b] - frame.hit) ** 2
    cube = frame.assign(d=diff).groupby(["origin", "state"]).d.mean().unstack()
    values = cube.to_numpy()
    no, ns = values.shape
    draws = []
    for _ in range(n):
        sample = values[rng.integers(0, no, no)][:, rng.integers(0, ns, ns)]
        draws.append(np.nanmean(sample))
    return np.percentile(draws, [2.5, 97.5])


def score_table(fc):
    rows = []
    for subset, frame in (("all", fc), ("contested", fc[(fc.start - fc.K).abs() <= NEAR_MONEY])):
        for T, g in frame.groupby("T"):
            row = {"subset": subset, "T": T, "n": len(g), "origins": g.origin.nunique(),
                   "first_origin": g.origin.min(), "last_origin": g.origin.max(), "base_rate": g.hit.mean()}
            for name in MODELS:
                row[f"brier_{name}"] = ((g[name] - g.hit) ** 2).mean()
            for label, model in (("old", "logistic_old"), ("fixed", "logistic_fixed"), ("gaussian", "gaussian")):
                low, high = pigeonhole_ci(g, model, "random_walk")
                row[f"{label}_minus_rw_low"], row[f"{label}_minus_rw_high"] = low, high
                by_origin = g.assign(d=(g[model] - g.hit) ** 2 - (g.random_walk - g.hit) ** 2).groupby("origin").d.mean()
                row[f"origins_rw_beats_{label}"] = int((by_origin > 0).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(fc, names=("logistic_old", "logistic_fixed", "gaussian", "random_walk")):
    rows = []
    for name in names:
        bins = pd.cut(fc[name], BINS, include_lowest=True)
        g = fc.groupby(bins, observed=True).agg(n=("hit", "size"), forecast=(name, "mean"), actual=("hit", "mean")).reset_index(drop=True)
        g.insert(0, "model", name)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def by_origin_table(fc):
    frame = fc[(fc.start - fc.K).abs() <= NEAR_MONEY]
    rows = []
    for (T, origin), g in frame.groupby(["T", "origin"]):
        row = {"T": T, "origin": origin, "n": len(g), "actual": g.hit.mean()}
        for name in ("logistic_old", "logistic_fixed", "gaussian", "random_walk", "random_walk_recent"):
            row[f"mean_{name}"] = g[name].mean()
            row[f"brier_{name}"] = ((g[name] - g.hit) ** 2).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def national_backtest(wide, us):
    """The surface is read at the national level, so also score the US series. Very few independent outcomes."""
    rows = []
    for T in HORIZONS:
        for origin in wide.columns:
            if origin + T not in us.index or origin not in us.index:
                continue
            train = training_pairs(wide, T, origin)
            if len(np.unique(train[:, 0])) < MIN_TRAIN_YEARS:
                continue
            preds = forecast(train, np.array([us[origin]]))
            for j, k in enumerate(STRIKES):
                row = {"T": T, "origin": origin, "K": k, "start": us[origin], "hit": float(us[origin + T] >= k)}
                row.update({name: preds[name][0, j] for name in MODELS})
                rows.append(row)
    return pd.DataFrame(rows)


def today_table(wide, us):
    """Both surviving models at today's national level. Agreement here is not a test, only a comparison."""
    origin = int(us.index.max())
    rows = []
    for T in HORIZONS:
        train = training_pairs(wide, T, origin)
        preds = forecast(train, np.array([us[origin]]))
        for j, k in enumerate(STRIKES):
            rows.append({"T": T, "K": k, "logistic_fixed": preds["logistic_fixed"][0, j], "gaussian": preds["gaussian"][0, j], "random_walk": preds["random_walk"][0, j]})
    return pd.DataFrame(rows)


def calibration_figure(cal, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    style = {"logistic_old": ("#c2410c", "o", "Logistic, old default penalty"),
             "logistic_fixed": ("#7c3aed", "s", "Logistic, unpenalized"),
             "gaussian": ("#00897b", "^", "One normal curve per horizon (current)")}
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1, ls="--", label="Perfect calibration")
    for name, (color, marker, label) in style.items():
        g = cal[cal.model == name]
        ax.plot(g.forecast, g.actual, color=color, marker=marker, ms=7, lw=2, label=label)
    ax.set(xlabel="Forecast probability", ylabel="Share that actually crossed", xlim=(0, 1), ylim=(0, 1),
           title="Out-of-sample calibration, state crossings, T = 1 to 5 years")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e7eb", lw=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    df = pd.read_csv(PROJECT / "data" / "raw" / "cdc_brfss_obesity_total_us_states.csv")
    states = df[~df.locationabbr.isin(["US", "GU", "PR", "VI"])]
    wide = states.pivot(index="locationabbr", columns="yearstart", values="p")
    us = df[df.locationabbr == "US"].set_index("yearstart").p.sort_index()
    if us.empty:
        raise SystemExit("No US rows in the CDC file")

    fc = run_state_backtest(wide)
    scores = score_table(fc)
    scores.round(4).to_csv(OUT / "backtest_scores.csv", index=False)
    cal = calibration_table(fc)
    cal.round(3).to_csv(OUT / "backtest_calibration.csv", index=False)
    origin = by_origin_table(fc)
    origin.round(4).to_csv(OUT / "backtest_by_origin.csv", index=False)
    calibration_figure(cal, OUT / "backtest_calibration.png")

    nat = national_backtest(wide, us)
    nat.round(4).to_csv(OUT / "backtest_national.csv", index=False)
    today = today_table(wide, us)
    today.round(4).to_csv(OUT / "model_comparison_today.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"{len(fc)} forecasts, origins {fc.origin.min()}-{fc.origin.max()}")
    print(scores.drop(columns=["first_origin", "last_origin"]).round(4).T.to_string())
    print("\nCalibration\n", cal.round(3).to_string(index=False))
    print("\nContested cells by origin\n", origin.round(3).to_string(index=False))
    print("\nNational Brier by horizon")
    print(nat.groupby("T").apply(lambda g: pd.Series({"n": len(g), **{m: ((g[m] - g.hit) ** 2).mean() for m in MODELS}}), include_groups=False).round(4).to_string())
    print("\nToday, by model\n", today.pivot(index="K", columns="T", values=["logistic_fixed", "gaussian", "random_walk"]).round(3).to_string())


if __name__ == "__main__":
    main()
