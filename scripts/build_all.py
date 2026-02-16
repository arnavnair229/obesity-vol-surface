"""
Build Health Volatility Surface from CDC BRFSS Obesity Data
Model 1: Threshold Digital Option on Logit(Obesity Prevalence)

Reads raw CDC JSON, computes implied sigma surface, outputs:
  - data/raw/cdc_brfss_obesity_total_us_states.csv
  - outputs/health_vol_surface_model1.csv
  - outputs/health-surface-data.json (for Three.js frontend)
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / "data" / "raw"
OUT_DIR = PROJECT / "outputs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load raw JSON ──
raw_path = RAW_DIR / "cdc_brfss_raw.json"
with open(raw_path) as f:
    rows = json.load(f)

df = pd.DataFrame(rows)
df["yearstart"] = df["yearstart"].astype(int)
df["data_value"] = pd.to_numeric(df.get("data_value"), errors="coerce")
df["p"] = df["data_value"] / 100.0
df = df.dropna(subset=["p"])
df = df[(df["p"] > 0) & (df["p"] < 1)]

csv_path = RAW_DIR / "cdc_brfss_obesity_total_us_states.csv"
df.sort_values(["locationabbr", "yearstart"]).to_csv(csv_path, index=False)
print(f"Saved CSV: {csv_path} ({len(df)} rows)")
print(f"Years: {df['yearstart'].min()}-{df['yearstart'].max()}")
print(f"Locations: {df['locationabbr'].nunique()}")

# ── 2. Compute surface ──
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))

def clamp_prob(x, lo=1e-4, hi=1 - 1e-4):
    return float(min(max(x, lo), hi))

df_work = df[["yearstart", "locationabbr", "locationdesc", "p"]].copy()
df_work["S"] = logit(df_work["p"].values)

# US national spot
us = df_work[df_work["locationabbr"] == "US"].sort_values("yearstart")
if us.empty:
    print("No US rows, computing national avg from states")
    us = df_work[~df_work["locationabbr"].isin(["US", "GU", "PR", "VI", "DC"])].copy()
    us = us.groupby("yearstart").agg(p=("p", "mean")).reset_index()
    us["S"] = logit(us["p"].values)
    us["locationabbr"] = "US"

S0 = float(us.iloc[-1]["S"])
p0 = float(us.iloc[-1]["p"])
y0 = int(us.iloc[-1]["yearstart"])
print(f"\nSpot (US, {y0}): p0={p0:.4f} ({p0*100:.1f}%), S0={S0:.4f}")

# State panel for cross-sectional estimation
states = df_work[~df_work["locationabbr"].isin(["US", "GU", "PR", "VI"])].copy()
panel = states.sort_values(["locationabbr", "yearstart"]).copy()

# Horizons and strikes
Ts = [1, 2, 3, 5, 7, 10]
Ks = [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45]

# Forward columns
panel = panel.set_index(["locationabbr", "yearstart"]).sort_index()
for T in Ts:
    panel[f"p_fwd_{T}"] = panel.groupby(level=0)["p"].shift(-T)
    panel[f"S_fwd_{T}"] = panel.groupby(level=0)["S"].shift(-T)
panel = panel.reset_index()

surface_rows = []
for T in Ts:
    valid = panel.dropna(subset=[f"S_fwd_{T}"]).copy()
    if len(valid) == 0:
        print(f"  T={T}: no forward data, skipping")
        continue

    dS = valid[f"S_fwd_{T}"].values - valid["S"].values
    mu_T = float(np.nanmean(dS))

    for K in Ks:
        K_logit = float(logit(np.array([K]))[0])
        event = (valid[f"p_fwd_{T}"].values >= K).astype(float)
        P_hat = float(np.mean(event))
        P_hat = clamp_prob(P_hat)

        z = norm.ppf(1.0 - P_hat)
        denom = z * np.sqrt(T)

        if abs(denom) < 1e-8:
            sigma = np.nan
        else:
            sigma = (K_logit - (S0 + mu_T)) / denom

        sigma = float(abs(sigma)) if not np.isnan(sigma) else np.nan

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
csv_out = OUT_DIR / "health_vol_surface_model1.csv"
surface.to_csv(csv_out, index=False)
print(f"\nSurface CSV: {csv_out}")
print(surface.to_string(index=False))

# ── 3. Generate Three.js JSON ──
pivot = surface.pivot(index="T_years", columns="K_obesity_threshold", values="implied_sigma_logit")
pivot = pivot.interpolate(method="linear", axis=1).interpolate(method="linear", axis=0)
pivot = pivot.bfill(axis=1).ffill(axis=1).bfill(axis=0).ffill(axis=0)

web_data = {
    "metadata": {
        "title": "US Health Volatility Surface",
        "subtitle": "Implied uncertainty of obesity threshold crossings",
        "model": "Model 1: Threshold Digital Option on Logit(Obesity Prevalence)",
        "source": "CDC BRFSS Nutrition, Physical Activity, and Obesity",
        "asof_year": y0,
        "spot_obesity_pct": round(p0 * 100, 1),
        "spot_logit": round(S0, 4),
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
    if r["implied_sigma_logit"] is not None:
        web_data["raw_points"].append({
            "T": r["T_years"],
            "K": r["K_obesity_threshold"],
            "sigma": round(r["implied_sigma_logit"], 4),
            "P_hat": round(r["P_hat_cross"], 4),
        })

json_path = OUT_DIR / "health-surface-data.json"
with open(json_path, "w") as f:
    json.dump(web_data, f, indent=2)

print(f"\nFrontend JSON: {json_path}")
print(f"Grid: {len(web_data['surface']['grid'])} x {len(web_data['surface']['grid'][0])}")
print(f"Points: {len(web_data['raw_points'])}")
print("\nSurface grid (T rows x K cols):")
for i, T in enumerate(pivot.index):
    vals = " ".join(f"{v:.4f}" for v in web_data["surface"]["grid"][i])
    print(f"  T={T:2d}y: {vals}")
