"""
Build Health Volatility Surface (Model 1: Threshold Digital Option)

Maps (T, K) -> implied sigma using:
  - Underlying: S_t = logit(p_t) where p_t = obesity prevalence
  - Strike K: obesity threshold (e.g., 30%, 35%, 40%)
  - Maturity T: forecasting horizon in years
  - P_hat: empirical probability of crossing threshold K at horizon T
  - Inversion: sigma(T,K) = (K_logit - (S0 + mu_T)) / (Phi^-1(1 - P_hat) * sqrt(T))

Outputs:
  - outputs/health_vol_surface_model1.csv (full surface data)
  - ../data/health-surface-data.json (for Three.js frontend)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

PROJECT = Path(__file__).resolve().parent.parent
INP = PROJECT / "data_raw" / "cdc_brfss_obesity_total_us_states.csv"
OUT = PROJECT / "outputs"
WEB_DATA = PROJECT.parent / "data"
OUT.mkdir(parents=True, exist_ok=True)
WEB_DATA.mkdir(parents=True, exist_ok=True)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def clamp_prob(x, lo=1e-4, hi=1 - 1e-4):
    return float(min(max(x, lo), hi))


# Load data
df = pd.read_csv(INP)
df = df[["yearstart", "locationabbr", "locationdesc", "p"]].copy()
df["S"] = logit(df["p"].values)

# US national series for "spot" S0
us = df[df["locationabbr"] == "US"].sort_values("yearstart").copy()
if us.empty:
    # Fall back: compute national average from states
    print("No 'US' rows found, computing national average from states...")
    us = df[~df["locationabbr"].isin(["US", "GU", "PR", "VI", "DC"])].copy()
    us = us.groupby("yearstart").agg(p=("p", "mean")).reset_index()
    us["S"] = logit(us["p"].values)
    us["locationabbr"] = "US"

S0 = float(us.iloc[-1]["S"])
p0 = float(us.iloc[-1]["p"])
y0 = int(us.iloc[-1]["yearstart"])
print(f"Spot (US, {y0}): p0={p0:.4f}, S0={S0:.4f}")

# Panel: use state-level data for richer cross-section
states = df[~df["locationabbr"].isin(["US", "GU", "PR", "VI"])].copy()
panel = states.sort_values(["locationabbr", "yearstart"]).copy()

# Horizons and strikes
Ts = [1, 2, 3, 5, 7, 10]
Ks = [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45]

# Create forward columns
panel = panel.set_index(["locationabbr", "yearstart"]).sort_index()
for T in Ts:
    panel[f"p_fwd_{T}"] = panel.groupby(level=0)["p"].shift(-T)
    panel[f"S_fwd_{T}"] = panel.groupby(level=0)["S"].shift(-T)
panel = panel.reset_index()

surface_rows = []

for T in Ts:
    valid = panel.dropna(subset=[f"S_fwd_{T}"]).copy()
    if len(valid) == 0:
        continue

    dS = valid[f"S_fwd_{T}"].values - valid["S"].values
    mu_T = float(np.nanmean(dS))

    for K in Ks:
        K_logit = float(logit(np.array([K]))[0])

        # Empirical crossing probability
        event = (valid[f"p_fwd_{T}"].values >= K).astype(float)
        P_hat = float(np.mean(event))
        P_hat = clamp_prob(P_hat)

        # Invert to implied sigma
        z = norm.ppf(1.0 - P_hat)
        denom = z * np.sqrt(T)

        if abs(denom) < 1e-8:
            sigma = np.nan
        else:
            sigma = (K_logit - (S0 + mu_T)) / denom

        sigma = float(abs(sigma)) if sigma is not None and not np.isnan(sigma) else np.nan

        surface_rows.append({
            "asof_year": y0,
            "spot_p": round(p0, 4),
            "spot_S": round(S0, 4),
            "T_years": T,
            "K_obesity_threshold": K,
            "K_logit": round(K_logit, 4),
            "P_hat_cross": round(P_hat, 4),
            "mu_T_logit": round(mu_T, 4),
            "implied_sigma_logit": round(sigma, 4) if not np.isnan(sigma) else None,
            "n_obs": int(len(valid)),
        })

surface = pd.DataFrame(surface_rows)

# Save CSV
csv_path = OUT / "health_vol_surface_model1.csv"
surface.to_csv(csv_path, index=False)
print(f"\nSaved surface CSV: {csv_path}")
print(f"Shape: {surface.shape}")
print(surface.to_string(index=False))

# --------------------------------------------------
# Generate JSON for Three.js frontend visualization
# --------------------------------------------------
# Build a grid: T (rows) x K (cols) -> sigma
pivot = surface.pivot(index="T_years", columns="K_obesity_threshold", values="implied_sigma_logit")

# Interpolate any NaN values
pivot = pivot.interpolate(method="linear", axis=1).interpolate(method="linear", axis=0)
pivot = pivot.fillna(method="bfill", axis=1).fillna(method="ffill", axis=1)
pivot = pivot.fillna(method="bfill", axis=0).fillna(method="ffill", axis=0)

# Build JSON structure for Three.js
web_data = {
    "metadata": {
        "title": "US Health Volatility Surface",
        "subtitle": "Implied uncertainty of obesity threshold crossings",
        "model": "Model 1: Threshold Digital Option on Logit(Obesity Prevalence)",
        "source": "CDC BRFSS Nutrition, Physical Activity, and Obesity",
        "asof_year": y0,
        "spot_obesity": round(p0, 4),
        "spot_logit": round(S0, 4),
    },
    "axes": {
        "x": {
            "label": "Obesity Threshold (Strike K)",
            "values": [round(k, 2) for k in pivot.columns.tolist()],
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

# Grid for surface mesh (row = T, col = K)
for t_idx, T in enumerate(pivot.index):
    row = []
    for k_idx, K in enumerate(pivot.columns):
        val = pivot.loc[T, K]
        row.append(round(float(val), 4) if pd.notna(val) else 0)
    web_data["surface"]["grid"].append(row)

# Raw points for scatter/markers
for _, r in surface.iterrows():
    if r["implied_sigma_logit"] is not None and not np.isnan(r["implied_sigma_logit"]):
        web_data["raw_points"].append({
            "T": r["T_years"],
            "K": r["K_obesity_threshold"],
            "sigma": round(r["implied_sigma_logit"], 4),
            "P_hat": round(r["P_hat_cross"], 4),
        })

json_path = WEB_DATA / "health-surface-data.json"
with open(json_path, "w") as f:
    json.dump(web_data, f, indent=2)

print(f"\nSaved frontend JSON: {json_path}")
print(f"Grid dimensions: {len(web_data['surface']['grid'])} x {len(web_data['surface']['grid'][0])}")
print(f"Raw points: {len(web_data['raw_points'])}")
