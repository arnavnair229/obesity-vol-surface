"""
Build Health Volatility Surface (Model 1 v2: Conditional + Logistic Smoothing)

Fixes over v1:
  1. Conditional probability via logistic regression: Pr(p_{t+T} >= K | S_t)
     evaluated at S_t = S0 (US spot) — eliminates unconditional mixing
  2. Conditional drift from ATM bucket — keeps moneyness analogy clean
  3. NaN for extreme probabilities — no fake clamped values
  4. Signed sigma preserved for diagnostics

Outputs:
  - outputs/health_vol_surface_model1.csv (full surface data)
  - outputs/health-surface-data.json (for Three.js frontend)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

PROJECT = Path(__file__).resolve().parent.parent
INP = PROJECT / "data" / "raw" / "cdc_brfss_obesity_total_us_states.csv"
OUT = PROJECT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


# ── 1. Load data ──
df = pd.read_csv(INP)
df = df[["yearstart", "locationabbr", "locationdesc", "p"]].copy()
df["S"] = logit(df["p"].values)

# US national spot
us = df[df["locationabbr"] == "US"].sort_values("yearstart")
if us.empty:
    print("No 'US' rows found, computing national average from states...")
    us = df[~df["locationabbr"].isin(["US", "GU", "PR", "VI", "DC"])].copy()
    us = us.groupby("yearstart").agg(p=("p", "mean")).reset_index()
    us["S"] = logit(us["p"].values)
    us["locationabbr"] = "US"

S0 = float(us.iloc[-1]["S"])
p0 = float(us.iloc[-1]["p"])
y0 = int(us.iloc[-1]["yearstart"])
print(f"Spot (US, {y0}): p0={p0:.4f} ({p0*100:.1f}%), S0={S0:.4f}")
print(f"Data range: {df['yearstart'].min()}-{df['yearstart'].max()}")

# ── 2. State panel ──
states = df[~df["locationabbr"].isin(["US", "GU", "PR", "VI"])].copy()
panel = states.sort_values(["locationabbr", "yearstart"]).copy()

Ts = [1, 2, 3, 5, 7, 10]
Ks = [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45]

# Forward columns
panel = panel.set_index(["locationabbr", "yearstart"]).sort_index()
for T in Ts:
    panel[f"p_fwd_{T}"] = panel.groupby(level=0)["p"].shift(-T)
    panel[f"S_fwd_{T}"] = panel.groupby(level=0)["S"].shift(-T)
panel = panel.reset_index()

# ATM bucket width (logit units) for conditional drift
DELTA = 0.25

# ── 3. Build surface ──
surface_rows = []

for T in Ts:
    valid = panel.dropna(subset=[f"S_fwd_{T}", f"p_fwd_{T}"]).copy()
    if len(valid) == 0:
        print(f"  T={T}: no forward data, skipping")
        continue

    # Conditional drift: mean logit change for states near current spot
    bucket = valid[(valid["S"] >= S0 - DELTA) & (valid["S"] <= S0 + DELTA)]
    if len(bucket) >= 30:
        mu_T = float(np.mean(bucket[f"S_fwd_{T}"] - bucket["S"]))
        mu_n = int(len(bucket))
        mu_method = "bucket"
    else:
        mu_T = float(np.mean(valid[f"S_fwd_{T}"] - valid["S"]))
        mu_n = int(len(valid))
        mu_method = "global"

    print(f"  T={T}: n={len(valid)}, bucket_n={len(bucket)}, mu_T={mu_T:.4f} ({mu_method})")

    # Feature matrix for logistic regression
    X = valid[["S"]].values

    for K in Ks:
        K_logit = float(logit(np.array([K]))[0])

        # Event: did forward prevalence cross threshold K?
        y = (valid[f"p_fwd_{T}"].values >= K).astype(int)
        event_rate = float(y.mean())

        # If event is too rare or too common, mark as NaN (can't estimate)
        if event_rate < 0.01 or event_rate > 0.99:
            surface_rows.append({
                "asof_year": y0,
                "spot_p": round(p0, 4),
                "spot_S": round(S0, 4),
                "T_years": T,
                "K_obesity_threshold": K,
                "K_logit": round(K_logit, 4),
                "P_cross_at_spot": np.nan,
                "mu_T_logit": round(mu_T, 4),
                "mu_method": mu_method,
                "mu_n_obs": mu_n,
                "event_rate_global": round(event_rate, 4),
                "implied_sigma_logit": np.nan,
                "implied_sigma_signed": np.nan,
                "n_obs_total": int(len(valid)),
                "note": "insufficient_variation",
            })
            continue

        # Logistic regression: Pr(crossing K | S_t)
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(X, y)
        P0 = float(model.predict_proba([[S0]])[0, 1])

        # If conditional probability at spot is extreme, mark as NaN
        if P0 <= 0.01 or P0 >= 0.99:
            surface_rows.append({
                "asof_year": y0,
                "spot_p": round(p0, 4),
                "spot_S": round(S0, 4),
                "T_years": T,
                "K_obesity_threshold": K,
                "K_logit": round(K_logit, 4),
                "P_cross_at_spot": round(P0, 4),
                "mu_T_logit": round(mu_T, 4),
                "mu_method": mu_method,
                "mu_n_obs": mu_n,
                "event_rate_global": round(event_rate, 4),
                "implied_sigma_logit": np.nan,
                "implied_sigma_signed": np.nan,
                "n_obs_total": int(len(valid)),
                "note": "extreme_conditional_prob",
            })
            continue

        # Invert for sigma using normal approximation on logit scale
        z = norm.ppf(1.0 - P0)
        denom = z * np.sqrt(T)

        if abs(denom) < 1e-8:
            sigma = np.nan
            sigma_signed = np.nan
            note = "near_atm_singularity"
        else:
            sigma_signed = (K_logit - (S0 + mu_T)) / denom
            sigma = abs(sigma_signed)
            note = "ok"

        surface_rows.append({
            "asof_year": y0,
            "spot_p": round(p0, 4),
            "spot_S": round(S0, 4),
            "T_years": T,
            "K_obesity_threshold": K,
            "K_logit": round(K_logit, 4),
            "P_cross_at_spot": round(P0, 4),
            "mu_T_logit": round(mu_T, 4),
            "mu_method": mu_method,
            "mu_n_obs": mu_n,
            "event_rate_global": round(event_rate, 4),
            "implied_sigma_logit": round(sigma, 4) if not np.isnan(sigma) else np.nan,
            "implied_sigma_signed": round(sigma_signed, 4) if not np.isnan(sigma_signed) else np.nan,
            "n_obs_total": int(len(valid)),
            "note": note,
        })

surface = pd.DataFrame(surface_rows)

# ── 4. Save CSV ──
csv_path = OUT / "health_vol_surface_model1.csv"
surface.to_csv(csv_path, index=False)
print(f"\nSaved surface CSV: {csv_path}")
print(f"Shape: {surface.shape}")
print(surface.to_string(index=False))

# ── 5. Generate Three.js JSON ──
pivot = surface.pivot(
    index="T_years", columns="K_obesity_threshold", values="implied_sigma_logit"
)
# Interpolate NaN for rendering (but mark them in raw_points)
pivot = pivot.interpolate(method="linear", axis=1).interpolate(method="linear", axis=0)
pivot = pivot.bfill(axis=1).ffill(axis=1).bfill(axis=0).ffill(axis=0)

web_data = {
    "metadata": {
        "title": "US Health Volatility Surface",
        "subtitle": "Conditional implied uncertainty of obesity threshold crossings",
        "model": "Model 1 v2: Logistic-smoothed conditional threshold crossing",
        "source": "CDC BRFSS Nutrition, Physical Activity, and Obesity",
        "asof_year": y0,
        "spot_obesity_pct": round(p0 * 100, 1),
        "spot_logit": round(S0, 4),
        "note": "Physical-measure risk surface. Probabilities conditional on S_t = S0 via logistic regression.",
    },
    "axes": {
        "x": {
            "label": "Obesity Threshold (Strike K %)",
            "values": [round(k * 100, 0) for k in pivot.columns.tolist()],
        },
        "y": {
            "label": "Horizon (Maturity T, years)",
            "values": pivot.index.tolist(),
        },
    },
    "surface": {
        "z_label": "Implied Sigma (logit scale)",
        "grid": [],
    },
    "raw_points": [],
}

for T in pivot.index:
    row = []
    for K in pivot.columns:
        val = pivot.loc[T, K]
        row.append(round(float(val), 4) if pd.notna(val) else 0)
    web_data["surface"]["grid"].append(row)

for _, r in surface.iterrows():
    if pd.notna(r.get("implied_sigma_logit")):
        web_data["raw_points"].append({
            "T": int(r["T_years"]),
            "K": float(r["K_obesity_threshold"]),
            "sigma": round(float(r["implied_sigma_logit"]), 4),
            "P_cross": round(float(r["P_cross_at_spot"]), 4),
            "note": r["note"],
        })

json_path = OUT / "health-surface-data.json"
with open(json_path, "w") as f:
    json.dump(web_data, f, indent=2)

print(f"\nFrontend JSON: {json_path}")
print(f"Grid: {len(web_data['surface']['grid'])} x {len(web_data['surface']['grid'][0])}")
print(f"Points: {len(web_data['raw_points'])}")
print("\nSurface grid (T rows x K cols):")
for i, T in enumerate(pivot.index):
    vals = " ".join(f"{v:.4f}" for v in web_data["surface"]["grid"][i])
    print(f"  T={T:2d}y: {vals}")
