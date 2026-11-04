"""
Build the obesity threshold-crossing surface (Model 1 v3: implied forward).

For each horizon T, a logistic regression gives P(p_{t+T} >= K | S_t = S0) on a fine strike grid.
The implied forward m_T is the logit strike where that probability is 50%, the same way an option
smile implies its own forward. Since drift now comes from the same curve as the probabilities,
sigma(T, K) = (K_logit - m_T) / (z(K) * sqrt(T)) stays finite at the money, where numerator and
denominator reach zero together. There, sigma is the local slope of the curve.

Earlier versions took drift from a separate bucket average. The numerator and denominator then
had different roots, which produced spikes at the 50% point (0.30 next to 0.03).

Cells with no usable probability are left empty, in both outputs. Nothing is filled in.

Outputs:
  - outputs/health_vol_surface_model1.csv
  - outputs/health-surface-data.json (for the site)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

PROJECT = Path(__file__).resolve().parent.parent
INP = PROJECT / "data" / "raw" / "cdc_brfss_obesity_total_us_states.csv"
OUT = PROJECT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TS = [1, 2, 3, 5, 7, 10]
KS = [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45]
FINE_KS = np.round(np.arange(0.22, 0.4801, 0.005), 3)
P_MIN, P_MAX = 0.01, 0.99  # outside this range the inversion is not meaningful
MIN_EVENT_RATE = 0.01
ATM_Z = 0.15  # inside this band, use the local slope instead of the ratio


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def crossing_prob(x, fwd_p, k, s0):
    """Logistic P(forward prevalence >= k | S = s0), or NaN if the event is too rare or too common."""
    y = (fwd_p >= k).astype(int)
    rate = y.mean()
    if rate < MIN_EVENT_RATE or rate > 1 - MIN_EVENT_RATE:
        return np.nan
    model = LogisticRegression(solver="lbfgs", max_iter=1000).fit(x, y)
    return float(model.predict_proba([[s0]])[0, 1])


def implied_forward(fine_k_logit, fine_z):
    """Root of z(K), by linear interpolation. Returns (m_T, dK/dz at the root), or None without a sign change."""
    ok = ~np.isnan(fine_z)
    k, z = fine_k_logit[ok], fine_z[ok]
    for i in range(len(z) - 1):
        if z[i] <= 0 <= z[i + 1] and z[i + 1] > z[i]:
            slope = (k[i + 1] - k[i]) / (z[i + 1] - z[i])
            return float(k[i] - z[i] * slope), float(slope)
    return None


df = pd.read_csv(INP)[["yearstart", "locationabbr", "locationdesc", "p"]].copy()
df["S"] = logit(df["p"].values)

us = df[df["locationabbr"] == "US"].sort_values("yearstart")
if us.empty:
    print("No 'US' rows found, computing national average from states...")
    us = df[~df["locationabbr"].isin(["US", "GU", "PR", "VI", "DC"])].copy()
    us = us.groupby("yearstart").agg(p=("p", "mean")).reset_index()
    us["S"] = logit(us["p"].values)

S0, p0, y0 = float(us.iloc[-1]["S"]), float(us.iloc[-1]["p"]), int(us.iloc[-1]["yearstart"])
print(f"Spot (US, {y0}): p0={p0:.4f}, S0={S0:.4f}")

panel = df[~df["locationabbr"].isin(["US", "GU", "PR", "VI"])].sort_values(["locationabbr", "yearstart"]).copy()
for T in TS:
    panel[f"p_fwd_{T}"] = panel.groupby("locationabbr")["p"].shift(-T)

rows = []
for T in TS:
    valid = panel.dropna(subset=[f"p_fwd_{T}"])
    x = valid[["S"]].values
    fwd = valid[f"p_fwd_{T}"].values

    fine_p = np.array([crossing_prob(x, fwd, k, S0) for k in FINE_KS])
    fine_p[(fine_p <= P_MIN) | (fine_p >= P_MAX)] = np.nan
    # Crossing probability must fall as the strike rises. Fits are separate, so enforce it.
    mono = fine_p.copy()
    seen = np.inf
    for i, v in enumerate(mono):
        if not np.isnan(v):
            seen = min(seen, v)
            mono[i] = seen
    fine_z = norm.ppf(1.0 - mono)
    fine_k_logit = logit(FINE_KS)

    root = implied_forward(fine_k_logit, fine_z)
    m_T, atm_slope = root if root else (np.nan, np.nan)
    print(f"  T={T:2d}: n={len(valid)}, implied forward="
          f"{'n/a' if not root else f'{np.exp(m_T)/(1+np.exp(m_T))*100:.1f}%'}")

    for K in KS:
        k_l = float(logit(np.array([K]))[0])
        P = float(np.interp(k_l, fine_k_logit[~np.isnan(mono)], mono[~np.isnan(mono)])) \
            if K in FINE_KS and not np.isnan(mono[list(FINE_KS).index(K)]) else np.nan
        sigma, note = np.nan, "no_usable_probability"
        if not np.isnan(P) and root:
            z = float(norm.ppf(1.0 - P))
            if abs(z) < ATM_Z:
                sigma, note = atm_slope / np.sqrt(T), "atm_local_slope"
            else:
                sigma, note = (k_l - m_T) / (z * np.sqrt(T)), "ok"
        rows.append({
            "asof_year": y0, "spot_p": round(p0, 4), "spot_S": round(S0, 4),
            "T_years": T, "K_obesity_threshold": K, "K_logit": round(k_l, 4),
            "P_cross_at_spot": round(P, 4) if not np.isnan(P) else np.nan,
            "implied_forward_logit": round(m_T, 4) if root else np.nan,
            "implied_forward_pct": round(float(np.exp(m_T) / (1 + np.exp(m_T)) * 100), 2) if root else np.nan,
            "implied_sigma_logit": round(float(sigma), 4) if not np.isnan(sigma) else np.nan,
            "n_obs_total": int(len(valid)), "note": note,
        })

surface = pd.DataFrame(rows)
surface.to_csv(OUT / "health_vol_surface_model1.csv", index=False)
print(surface[["T_years", "K_obesity_threshold", "P_cross_at_spot", "implied_forward_pct", "implied_sigma_logit", "note"]].to_string(index=False))

pivot = surface.pivot(index="T_years", columns="K_obesity_threshold", values="implied_sigma_logit")
forwards = surface.groupby("T_years")["implied_forward_pct"].first()
web = {
    "metadata": {
        "title": "US Obesity Threshold Surface",
        "subtitle": "Implied uncertainty of threshold crossings, from CDC BRFSS state data",
        "model": "Model 1 v3: logistic crossing curve with implied forward",
        "source": "CDC BRFSS Nutrition, Physical Activity, and Obesity",
        "asof_year": y0,
        "spot_obesity_pct": round(p0 * 100, 1),
        "spot_logit": round(S0, 4),
        "note": "Physical-measure, not risk-neutral. Null cells had no usable probability and are not interpolated.",
    },
    "axes": {
        "x": {"label": "Obesity Threshold (Strike K %)", "values": [round(k * 100, 0) for k in pivot.columns]},
        "y": {"label": "Horizon (Maturity T, years)", "values": [int(t) for t in pivot.index]},
    },
    "implied_forward_pct": {str(int(t)): (None if pd.isna(v) else float(v)) for t, v in forwards.items()},
    "surface": {
        "z_label": "Implied Sigma (logit scale)",
        "grid": [[None if pd.isna(v) else round(float(v), 4) for v in pivot.loc[t]] for t in pivot.index],
    },
    "raw_points": [
        {"T": int(r.T_years), "K": float(r.K_obesity_threshold), "sigma": float(r.implied_sigma_logit),
         "P_cross": float(r.P_cross_at_spot), "note": r.note}
        for r in surface.itertuples() if pd.notna(r.implied_sigma_logit)
    ],
}
(OUT / "health-surface-data.json").write_text(json.dumps(web, indent=2))
print("\nSurface grid (T rows x K cols, -- = no data):")
for t, row in zip(pivot.index, web["surface"]["grid"]):
    print(f"  T={t:2d}y: " + " ".join("  --  " if v is None else f"{v:.4f}" for v in row))
