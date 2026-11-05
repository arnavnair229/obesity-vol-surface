"""
Build the obesity threshold-crossing surface (Model 2: one normal curve per horizon).

For each horizon T, fit logit(p_t+T) = a + b logit(p_t) + normal error on the state panel. Then
    forward m_T = a + b S0
    P(p_T >= K) = Phi((m_T - logit(K)) / s_T)
    implied sigma = s_T / sqrt(T)

Earlier versions fit a separate logistic regression at every strike and inverted the results, which
left noise near the forward (sigma of 0.027 beside 0.042). One curve per horizon removes that, gives
closed-form Greeks, and scored as well or better in 08_backtest.py. Its sigma is flat across strikes
by construction, and the residuals are close to normal, so the data doesn't call for a smile.

The spot adjusts for missing states. CDC's 2025 national figure excludes California, Mississippi and
Nevada, so it isn't comparable with 2024 as published.

Outputs:
  - outputs/health_vol_surface_model1.csv
  - outputs/health-surface-data.json (for the site)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TS = [1, 2, 3, 5, 7, 10]
KS = [0.25, 0.28, 0.30, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.40, 0.42, 0.45]
FIPS_COLUMNS = ["year", "fips", "population"]
EXCLUDE = ["US", "GU", "PR", "VI"]


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def load_states():
    d = pd.read_csv(RAW / "cdc_brfss_latest.csv")
    return d[d.group == "Total"]


def adjusted_spot(total):
    """The latest US figure, restated for the states missing from it, weighting states by adult population."""
    us = total[total.location == "US"].set_index("year").percent
    latest = int(us.index.max())
    adults = pd.read_csv(RAW / "census_adults_by_age.csv").groupby("state").adults.sum()
    wide = total[~total.location.isin(EXCLUDE)].pivot(index="location", columns="year", values="percent")
    previous = wide[latest - 1].dropna()
    reporting = previous.index.intersection(wide[latest].dropna().index)
    weight = adults.reindex(previous.index)
    everyone = (previous * weight).sum() / weight.sum()
    without = (previous[reporting] * weight[reporting]).sum() / weight[reporting].sum()
    return {"year": latest, "reported": float(us[latest]), "composition_effect": float(without - everyone),
            "adjusted": float(us[latest] - (without - everyone)), "missing": sorted(set(previous.index) - set(reporting))}


def fit_curves(total, TS=TS):
    wide = total[~total.location.isin(EXCLUDE)].pivot(index="location", columns="year", values="percent") / 100
    curves = {}
    for T in TS:
        pairs = np.vstack([np.column_stack([logit(wide[y]), logit(wide[y + T])]) for y in wide.columns if y + T in wide.columns])
        pairs = pairs[~np.isnan(pairs).any(axis=1)]
        b, a = np.polyfit(pairs[:, 0], pairs[:, 1], 1)
        s = (pairs[:, 1] - a - b * pairs[:, 0]).std(ddof=2)
        curves[T] = {"a": float(a), "b": float(b), "s": float(s), "n": len(pairs)}
    return curves


def main():
    total = load_states()
    spot = adjusted_spot(total)
    p0 = spot["adjusted"] / 100
    S0 = float(logit(p0))
    print(f"Spot {spot['year']}: reported {spot['reported']}, adjusted {spot['adjusted']:.2f}, missing {spot['missing']}")
    curves = fit_curves(total)

    rows = []
    for T, c in curves.items():
        m = c["a"] + c["b"] * S0
        print(f"  T={T:2d}: n={c['n']}, b={c['b']:.3f}, s={c['s']:.4f}, forward={expit(m) * 100:.1f}%")
        for K in KS:
            k = float(logit(K))
            rows.append({
                "asof_year": spot["year"], "spot_p": round(p0, 4), "spot_S": round(S0, 4), "T_years": T,
                "K_obesity_threshold": K, "K_logit": round(k, 4), "P_cross_at_spot": round(float(norm.cdf((m - k) / c["s"])), 4),
                "implied_forward_logit": round(m, 4), "implied_forward_pct": round(float(expit(m) * 100), 2),
                "implied_sigma_logit": round(c["s"] / np.sqrt(T), 4), "n_obs_total": c["n"], "note": "normal_curve",
            })
    surface = pd.DataFrame(rows)
    surface.to_csv(OUT / "health_vol_surface_model1.csv", index=False)

    sigma = surface.pivot(index="T_years", columns="K_obesity_threshold", values="implied_sigma_logit")
    crossing = surface.pivot(index="T_years", columns="K_obesity_threshold", values="P_cross_at_spot")
    forwards = surface.groupby("T_years").implied_forward_pct.first()
    web = {
        "metadata": {
            "title": "US Obesity Threshold Surface",
            "subtitle": "Chance of crossing each obesity level by horizon, from CDC BRFSS state data",
            "model": "Model 2: one normal curve per horizon",
            "source": "CDC BRFSS Nutrition, Physical Activity, and Obesity",
            "asof_year": spot["year"],
            "spot_obesity_pct": round(spot["adjusted"], 1),
            "reported_obesity_pct": spot["reported"],
            "states_missing_from_reported": spot["missing"],
            "spot_logit": round(S0, 4),
            "note": "Historical frequencies, not risk-neutral prices. For a state at the national level.",
        },
        "axes": {
            "x": {"label": "Obesity Threshold (Strike K %)", "values": [round(k * 100, 0) for k in sigma.columns]},
            "y": {"label": "Horizon (Maturity T, years)", "values": [int(t) for t in sigma.index]},
        },
        "implied_forward_pct": {str(int(t)): float(v) for t, v in forwards.items()},
        "surface": {"z_label": "Implied Sigma (logit scale)", "grid": [[float(v) for v in sigma.loc[t]] for t in sigma.index]},
        "crossing": {"z_label": "Chance of crossing", "grid": [[float(v) for v in crossing.loc[t]] for t in crossing.index]},
    }
    (OUT / "health-surface-data.json").write_text(json.dumps(web, indent=2))
    print((crossing * 100).round(1).to_string())


if __name__ == "__main__":
    main()
