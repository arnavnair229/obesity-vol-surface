"""
Exposure-Outcome Analysis: Additive Intensity vs Obesity Prevalence

Merges CDC obesity time series with Open Food Facts additive metrics to test:
  H1: Food formulation complexity (additive/ingredient intensity) has increased
  H2: Increases in additive intensity are associated with worsening obesity

Methods:
  - Composite Additive Intensity Index (AII) from OFF data
  - Pearson/Spearman correlation (levels and first-differences)
  - OLS regression: Delta_S_t = alpha + beta * Delta_AII_t + epsilon_t
  - Lead-lag cross-correlation analysis
  - Dual-axis time series visualization

Outputs:
  - outputs/exposure_outcome_analysis.png  (multi-panel figure)
  - outputs/exposure_regression_results.csv (regression table)
  - outputs/merged_timeseries.csv          (merged annual data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load both datasets ──

# CDC obesity (national)
cdc = pd.read_csv(PROJECT / "data" / "raw" / "cdc_brfss_obesity_total_us_states.csv")
us = cdc[cdc["locationabbr"] == "US"][["yearstart", "p"]].copy()
us = us.rename(columns={"yearstart": "year", "p": "obesity_pct"})
us = us.sort_values("year").reset_index(drop=True)

# Open Food Facts additive metrics
off = pd.read_csv(OUT / "additive_load_by_year.csv")

print("CDC obesity years:", sorted(us["year"].unique()))
print("OFF additive years:", sorted(off["year"].unique()))

# ── 2. Merge on year ──
merged = us.merge(off, on="year", how="inner")
print(f"\nOverlapping years: {merged['year'].min()}-{merged['year'].max()} ({len(merged)} obs)")

# ── 3. Build Composite Additive Intensity Index (AII) ──
# Normalize each component to [0, 1] then average
components = ["mean_additives", "mean_ingredients", "nova4_share"]
for col in components:
    cmin, cmax = merged[col].min(), merged[col].max()
    if cmax > cmin:
        merged[f"{col}_norm"] = (merged[col] - cmin) / (cmax - cmin)
    else:
        merged[f"{col}_norm"] = 0.5

merged["AII"] = merged[[f"{c}_norm" for c in components]].mean(axis=1)

# Logit-transform obesity for consistency with vol surface
merged["S"] = np.log(merged["obesity_pct"] / (1 - merged["obesity_pct"]))

# First differences (for non-spurious correlation on trending series)
merged["d_obesity"] = merged["obesity_pct"].diff()
merged["d_S"] = merged["S"].diff()
merged["d_AII"] = merged["AII"].diff()
merged["d_mean_additives"] = merged["mean_additives"].diff()
merged["d_mean_ingredients"] = merged["mean_ingredients"].diff()
merged["d_nova4"] = merged["nova4_share"].diff()

# Save merged data
merged.to_csv(OUT / "merged_timeseries.csv", index=False)

# ── 4. Correlation Analysis ──
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

# Levels
print("\n--- Levels (Pearson) ---")
level_pairs = [
    ("obesity_pct", "AII", "Obesity vs AII"),
    ("obesity_pct", "mean_additives", "Obesity vs Mean Additives"),
    ("obesity_pct", "mean_ingredients", "Obesity vs Mean Ingredients"),
    ("obesity_pct", "nova4_share", "Obesity vs NOVA 4 Share"),
]
level_results = []
for x, y, label in level_pairs:
    mask = merged[[x, y]].notna().all(axis=1)
    r, p = stats.pearsonr(merged.loc[mask, x], merged.loc[mask, y])
    rho, p_s = stats.spearmanr(merged.loc[mask, x], merged.loc[mask, y])
    print(f"  {label:35s}  r={r:+.3f} (p={p:.3f})  rho={rho:+.3f} (p={p_s:.3f})")
    level_results.append({"pair": label, "pearson_r": r, "pearson_p": p,
                          "spearman_rho": rho, "spearman_p": p_s, "type": "levels"})

# First differences (addresses common-trend spurious correlation)
print("\n--- First Differences (Pearson) ---")
diff_pairs = [
    ("d_obesity", "d_AII", "dObesity vs dAII"),
    ("d_obesity", "d_mean_additives", "dObesity vs dMean Additives"),
    ("d_obesity", "d_mean_ingredients", "dObesity vs dMean Ingredients"),
    ("d_obesity", "d_nova4", "dObesity vs dNOVA 4 Share"),
]
diff_results = []
for x, y, label in diff_pairs:
    mask = merged[[x, y]].notna().all(axis=1)
    if mask.sum() < 4:
        print(f"  {label:35s}  insufficient obs ({mask.sum()})")
        continue
    r, p = stats.pearsonr(merged.loc[mask, x], merged.loc[mask, y])
    rho, p_s = stats.spearmanr(merged.loc[mask, x], merged.loc[mask, y])
    print(f"  {label:35s}  r={r:+.3f} (p={p:.3f})  rho={rho:+.3f} (p={p_s:.3f})")
    diff_results.append({"pair": label, "pearson_r": r, "pearson_p": p,
                         "spearman_rho": rho, "spearman_p": p_s, "type": "first_diff"})

# ── 5. OLS Regression ──
print("\n" + "=" * 70)
print("OLS REGRESSION: d(Obesity) = alpha + beta * d(AII) + epsilon")
print("=" * 70)

reg_data = merged[["d_obesity", "d_AII", "d_mean_ingredients", "d_nova4"]].dropna()
n = len(reg_data)

regression_results = []

# Model 1: d(Obesity) ~ d(AII)
if n >= 4:
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        reg_data["d_AII"], reg_data["d_obesity"]
    )
    print(f"\n  Model 1: d(Obesity) ~ d(AII)")
    print(f"    alpha  = {intercept:+.5f}")
    print(f"    beta   = {slope:+.5f}  (SE={std_err:.5f}, p={p_value:.4f})")
    print(f"    R^2    = {r_value**2:.4f}")
    print(f"    n      = {n}")
    regression_results.append({
        "model": "d(Obesity) ~ d(AII)",
        "alpha": intercept, "beta": slope, "se": std_err,
        "p_value": p_value, "r_squared": r_value**2, "n": n,
    })

# Model 2: d(Obesity) ~ d(Mean Ingredients)
if n >= 4:
    slope2, intercept2, r2, p2, se2 = stats.linregress(
        reg_data["d_mean_ingredients"], reg_data["d_obesity"]
    )
    print(f"\n  Model 2: d(Obesity) ~ d(Mean Ingredients)")
    print(f"    alpha  = {intercept2:+.5f}")
    print(f"    beta   = {slope2:+.5f}  (SE={se2:.5f}, p={p2:.4f})")
    print(f"    R^2    = {r2**2:.4f}")
    print(f"    n      = {n}")
    regression_results.append({
        "model": "d(Obesity) ~ d(Mean Ingredients)",
        "alpha": intercept2, "beta": slope2, "se": se2,
        "p_value": p2, "r_squared": r2**2, "n": n,
    })

# Model 3: d(Obesity) ~ d(NOVA 4 Share)
if n >= 4:
    slope3, intercept3, r3, p3, se3 = stats.linregress(
        reg_data["d_nova4"], reg_data["d_obesity"]
    )
    print(f"\n  Model 3: d(Obesity) ~ d(NOVA 4 Share)")
    print(f"    alpha  = {intercept3:+.5f}")
    print(f"    beta   = {slope3:+.5f}  (SE={se3:.5f}, p={p3:.4f})")
    print(f"    R^2    = {r3**2:.4f}")
    print(f"    n      = {n}")
    regression_results.append({
        "model": "d(Obesity) ~ d(NOVA 4 Share)",
        "alpha": intercept3, "beta": slope3, "se": se3,
        "p_value": p3, "r_squared": r3**2, "n": n,
    })

# Save regression results
reg_df = pd.DataFrame(level_results + diff_results + regression_results)
reg_df.to_csv(OUT / "exposure_regression_results.csv", index=False)

# ── 6. Lead-Lag Cross-Correlation ──
print("\n" + "=" * 70)
print("LEAD-LAG CROSS-CORRELATION: d(Obesity) vs d(AII)")
print("=" * 70)

cc_data = merged[["d_obesity", "d_AII"]].dropna()
max_lag = min(4, len(cc_data) // 2 - 1)
lag_corrs = []
for lag in range(-max_lag, max_lag + 1):
    if lag >= 0:
        x = cc_data["d_AII"].iloc[:len(cc_data) - lag].values
        y = cc_data["d_obesity"].iloc[lag:].values
    else:
        x = cc_data["d_AII"].iloc[-lag:].values
        y = cc_data["d_obesity"].iloc[:len(cc_data) + lag].values
    if len(x) >= 4:
        r, p = stats.pearsonr(x, y)
        lag_corrs.append({"lag": lag, "corr": r, "p_value": p, "n": len(x)})
        direction = "AII leads" if lag > 0 else ("Obesity leads" if lag < 0 else "contemporaneous")
        print(f"  lag={lag:+d} ({direction:20s}):  r={r:+.3f}  p={p:.3f}  n={len(x)}")

# ── 7. Trend Significance Tests ──
print("\n" + "=" * 70)
print("TREND TESTS")
print("=" * 70)

# Obesity trend
ob_slope, ob_int, ob_r, ob_p, ob_se = stats.linregress(us["year"], us["obesity_pct"])
print(f"\n  Obesity linear trend: +{ob_slope*100:.2f} pp/year (p={ob_p:.6f}, R^2={ob_r**2:.3f})")

# AII trend (on overlapping years with sufficient data)
aii_valid = merged[merged["n_products"] >= 15]
if len(aii_valid) >= 5:
    aii_slope, aii_int, aii_r, aii_p, aii_se = stats.linregress(
        aii_valid["year"], aii_valid["AII"]
    )
    print(f"  AII linear trend:    +{aii_slope:.4f}/year (p={aii_p:.4f}, R^2={aii_r**2:.3f})")

# Ingredient count trend
if len(aii_valid) >= 5:
    ing_slope, _, ing_r, ing_p, _ = stats.linregress(
        aii_valid["year"], aii_valid["mean_ingredients"]
    )
    print(f"  Ingredients trend:   +{ing_slope:.2f}/year (p={ing_p:.4f}, R^2={ing_r**2:.3f})")

# ── 8. Generate Research Figure ──
print("\n" + "=" * 70)
print("GENERATING FIGURE")
print("=" * 70)

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.35)

# Panel A: Dual-axis time series (Obesity + AII)
ax1 = fig.add_subplot(gs[0, :2])
color1 = "crimson"
color2 = "steelblue"
ln1 = ax1.plot(merged["year"], merged["obesity_pct"] * 100, "o-",
               color=color1, linewidth=2, markersize=6, label="Obesity (%)")
ax1.set_xlabel("Year")
ax1.set_ylabel("Obesity Prevalence (%)", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_ylim(25, 36)

ax1b = ax1.twinx()
ln2 = ax1b.plot(merged["year"], merged["AII"], "s--",
                color=color2, linewidth=2, markersize=6, label="AII")
ax1b.set_ylabel("Additive Intensity Index", color=color2)
ax1b.tick_params(axis="y", labelcolor=color2)

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="upper left")
ax1.set_title("A. Obesity Prevalence vs Additive Intensity Index")
ax1.grid(True, alpha=0.2)

# Panel B: Scatter (levels) — Obesity vs AII
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(merged["AII"], merged["obesity_pct"] * 100, c=merged["year"],
            cmap="viridis", s=80, edgecolors="black", linewidth=0.5, zorder=3)
# Fit line
mask = merged[["AII", "obesity_pct"]].notna().all(axis=1)
if mask.sum() >= 4:
    z = np.polyfit(merged.loc[mask, "AII"], merged.loc[mask, "obesity_pct"] * 100, 1)
    xfit = np.linspace(merged["AII"].min(), merged["AII"].max(), 50)
    ax2.plot(xfit, np.polyval(z, xfit), "--", color="gray", alpha=0.7)
for _, row in merged.iterrows():
    ax2.annotate(str(int(row["year"])), (row["AII"], row["obesity_pct"] * 100),
                 fontsize=7, ha="center", va="bottom")
ax2.set_xlabel("Additive Intensity Index")
ax2.set_ylabel("Obesity Prevalence (%)")
ax2.set_title("B. Cross-Sectional: Obesity vs AII")
ax2.grid(True, alpha=0.2)

# Panel C: Scatter (first differences) — dObesity vs dAII
ax3 = fig.add_subplot(gs[1, 0])
diff_valid = merged[["d_obesity", "d_AII", "year"]].dropna()
if len(diff_valid) >= 3:
    ax3.scatter(diff_valid["d_AII"], diff_valid["d_obesity"] * 100,
                c=diff_valid["year"], cmap="viridis", s=80,
                edgecolors="black", linewidth=0.5, zorder=3)
    ax3.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax3.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    if len(diff_valid) >= 4:
        z2 = np.polyfit(diff_valid["d_AII"], diff_valid["d_obesity"] * 100, 1)
        xfit2 = np.linspace(diff_valid["d_AII"].min(), diff_valid["d_AII"].max(), 50)
        ax3.plot(xfit2, np.polyval(z2, xfit2), "--", color="gray", alpha=0.7)
    for _, row in diff_valid.iterrows():
        ax3.annotate(str(int(row["year"])), (row["d_AII"], row["d_obesity"] * 100),
                     fontsize=7, ha="center", va="bottom")
ax3.set_xlabel("$\\Delta$ AII")
ax3.set_ylabel("$\\Delta$ Obesity (pp)")
ax3.set_title("C. First Differences: $\\Delta$Obesity vs $\\Delta$AII")
ax3.grid(True, alpha=0.2)

# Panel D: Component breakdown — ingredients over time
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(merged["year"], merged["mean_ingredients"], color="teal", alpha=0.7)
ax4.set_xlabel("Year")
ax4.set_ylabel("Mean Ingredient Count")
ax4.set_title("D. Ingredient Complexity Over Time")
ax4.grid(True, alpha=0.2)

# Panel E: NOVA 4 share over time
ax5 = fig.add_subplot(gs[1, 2])
ax5.bar(merged["year"], merged["nova4_share"] * 100, color="firebrick", alpha=0.7)
ax5.set_xlabel("Year")
ax5.set_ylabel("NOVA 4 Share (%)")
ax5.set_title("E. Ultra-Processed Product Share")
ax5.grid(True, alpha=0.2)

# Panel F: Lead-lag cross-correlation
ax6 = fig.add_subplot(gs[2, 0])
if lag_corrs:
    lags = [lc["lag"] for lc in lag_corrs]
    corrs = [lc["corr"] for lc in lag_corrs]
    colors = ["darkorange" if lc["p_value"] < 0.10 else "lightgray" for lc in lag_corrs]
    ax6.bar(lags, corrs, color=colors, edgecolor="black", linewidth=0.5)
    ax6.axhline(0, color="black", linewidth=0.5)
    ax6.set_xlabel("Lag (years, positive = AII leads obesity)")
    ax6.set_ylabel("Correlation")
    ax6.set_title("F. Lead-Lag Cross-Correlation")
    ax6.grid(True, alpha=0.2)

# Panel G: Crossing probability term structure
surface = pd.read_csv(OUT / "health_vol_surface_model1.csv")
ax7 = fig.add_subplot(gs[2, 1])
for K in [0.30, 0.32, 0.35, 0.38, 0.40]:
    sub = surface[surface["K_obesity_threshold"] == K]
    sub = sub.dropna(subset=["P_cross_at_spot"])
    if len(sub) > 0:
        ax7.plot(sub["T_years"], sub["P_cross_at_spot"],
                 "o-", linewidth=2, markersize=5, label=f"K={K*100:.0f}%")
ax7.set_xlabel("Horizon T (years)")
ax7.set_ylabel("Pr(Obesity $\\geq$ K | S$_0$)")
ax7.set_title("G. Threshold-Crossing Probability Term Structure")
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.2)

# Panel H: Implied sigma surface (heatmap)
ax8 = fig.add_subplot(gs[2, 2])
pivot = surface.pivot(index="T_years", columns="K_obesity_threshold",
                      values="implied_sigma_logit")
im = ax8.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                origin="lower", interpolation="nearest")
ax8.set_xticks(range(len(pivot.columns)))
ax8.set_xticklabels([f"{k*100:.0f}%" for k in pivot.columns], fontsize=8)
ax8.set_yticks(range(len(pivot.index)))
ax8.set_yticklabels([f"{t}y" for t in pivot.index], fontsize=8)
ax8.set_xlabel("Obesity Threshold (Strike K)")
ax8.set_ylabel("Horizon (Maturity T)")
ax8.set_title("H. Implied Sigma Surface (logit)")
plt.colorbar(im, ax=ax8, shrink=0.8, label="$\\sigma$")

fig.suptitle("Profit Incentives vs Public Health: Obesity Risk Surface & Additive Exposure",
             fontsize=14, fontweight="bold", y=0.98)
plt.savefig(OUT / "exposure_outcome_analysis.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT / 'exposure_outcome_analysis.png'}")
plt.close()

# ── 9. Print Summary ──
print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY")
print("=" * 70)
print(f"""
Data:
  CDC BRFSS obesity prevalence: {us['year'].min()}-{us['year'].max()} (US national)
  Open Food Facts additive metrics: {off['year'].min()}-{off['year'].max()} ({off['n_products'].sum()} products)
  Overlapping analysis period: {merged['year'].min()}-{merged['year'].max()} ({len(merged)} annual obs)

Trends:
  Obesity: +{ob_slope*100:.2f} pp/year (p={ob_p:.2e}), from {us['obesity_pct'].iloc[0]*100:.1f}% to {us['obesity_pct'].iloc[-1]*100:.1f}%
  Ingredient complexity: +{ing_slope:.1f} ingredients/year (p={ing_p:.3f})

Correlation (levels):
  Obesity vs AII: r={level_results[0]['pearson_r']:+.3f} (p={level_results[0]['pearson_p']:.3f})

Correlation (first differences, spurious-trend-robust):
  dObesity vs dAII: r={diff_results[0]['pearson_r']:+.3f} (p={diff_results[0]['pearson_p']:.3f}) [n={len(reg_data)}]

Regression:
  d(Obesity) = {regression_results[0]['alpha']:+.5f} + {regression_results[0]['beta']:+.5f} * d(AII)
  R^2 = {regression_results[0]['r_squared']:.4f}, p = {regression_results[0]['p_value']:.4f}

Caveats:
  - Small sample (n={len(merged)} annual obs); low statistical power
  - OFF data reflects database additions, not market launches (selection bias)
  - Correlation is not causation; common confounders (income, urbanization) not controlled
  - Additive metrics based on {off['n_products'].sum()} sampled products (not full market)
""")
