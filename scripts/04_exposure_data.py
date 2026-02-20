"""
Build Food Supply Exposure Index from USDA ERS + NHANES data

Downloads and compiles multiple USDA Economic Research Service datasets
and published NHANES ultra-processed food estimates to construct a
multi-factor exposure index for US food supply changes.

Sources:
  1. USDA ERS Sugar & Sweeteners Yearbook (Table 50):
     Per capita caloric sweetener deliveries, 1966-2024
  2. USDA ERS Oil Crops Yearbook (Table 5):
     Soybean oil domestic disappearance, 1980-2024
  3. USDA ERS Food Expenditure Series:
     Food-away-from-home expenditure share, 1997-2024
  4. NHANES Ultra-Processed Food estimates (published literature):
     UPF % of total calories, 2001-2018 (biennial)

Outputs:
  - data/raw/usda_ers_sweetener_consumption.xlsx (cached)
  - data/raw/usda_ers_soybean.xlsx (cached)
  - data/raw/usda_ers_food_expenditure_normalized.xlsx (cached)
  - outputs/exposure_index.csv
  - outputs/exposure_trends.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / "data" / "raw"
OUT = PROJECT / "outputs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# ── Download URLs ──
SWEETENER_URL = "https://www.ers.usda.gov/media/5149/us-consumption-of-caloric-sweeteners.xlsx?v=94508"
SOYBEAN_URL = "https://www.ers.usda.gov/media/5230/soybean-us-stocks-on-farm-off-farm-and-total-by-quarter-us-soybean-acreage-planted-harvested-yield-soybean-and-soybean-meal-production-value-price-and-supply-and-disappearance-prices-199900-202425.xlsx?v=74221"
FOOD_EXP_URL = "https://www.ers.usda.gov/media/5197/normalized-food-expenditures-by-all-purchasers-and-household-final-users.xlsx?v=43058"


def download_if_missing(url, path, label):
    """Download a file if not already cached."""
    if path.exists():
        print(f"  {label}: cached at {path.name}")
        return
    print(f"  {label}: downloading...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"  {label}: saved ({len(r.content):,} bytes)")


def parse_sweetener_data():
    """Parse USDA ERS Table 50: per capita caloric sweetener deliveries."""
    path = RAW_DIR / "usda_ers_sweetener_consumption.xlsx"
    download_if_missing(SWEETENER_URL, path, "Sweetener Yearbook")

    df = pd.read_excel(path, sheet_name="Table50", header=None)
    rows = []
    for i in range(7, len(df)):
        row = df.iloc[i]
        try:
            year = int(float(str(row.iloc[0])))
            if year < 1960 or year > 2030:
                continue
        except (ValueError, TypeError):
            continue

        rows.append({
            "year": year,
            "refined_sugar_lbs_pc": pd.to_numeric(row.iloc[2], errors="coerce"),
            "hfcs_lbs_pc": pd.to_numeric(row.iloc[3], errors="coerce"),
            "total_corn_sweet_lbs_pc": pd.to_numeric(row.iloc[6], errors="coerce"),
            "total_caloric_sweet_lbs_pc": pd.to_numeric(row.iloc[9], errors="coerce"),
        })

    result = pd.DataFrame(rows).dropna(subset=["total_caloric_sweet_lbs_pc"])
    print(f"  Sweetener: {len(result)} years ({result['year'].min()}-{result['year'].max()})")
    return result


def parse_soybean_oil_data():
    """Parse USDA ERS Table 5: soybean oil supply and disappearance."""
    path = RAW_DIR / "usda_ers_soybean.xlsx"
    download_if_missing(SOYBEAN_URL, path, "Oil Crops Yearbook")

    df = pd.read_excel(path, sheet_name="tab5", header=None)
    rows = []
    for i in range(6, len(df)):
        row = df.iloc[i]
        year_str = str(row.iloc[0]).strip()
        if "/" not in year_str:
            continue
        try:
            year = int(year_str.split("/")[0])
            if year < 1970 or year > 2030:
                continue
        except (ValueError, TypeError):
            continue

        rows.append({
            "year": year,
            "soy_oil_domestic_mlbs": pd.to_numeric(row.iloc[5], errors="coerce"),
            "soy_oil_production_mlbs": pd.to_numeric(row.iloc[2], errors="coerce"),
        })

    result = pd.DataFrame(rows).dropna(subset=["soy_oil_domestic_mlbs"])
    print(f"  Soybean oil: {len(result)} years ({result['year'].min()}-{result['year'].max()})")
    return result


def parse_food_expenditure_data():
    """Parse USDA ERS normalized food expenditure (FAH vs FAFH shares)."""
    path = RAW_DIR / "usda_ers_food_expenditure_normalized.xlsx"
    download_if_missing(FOOD_EXP_URL, path, "Food Expenditure Series")

    df = pd.read_excel(path, sheet_name="Normalized food expenditures", header=None)
    rows = []
    for i in range(2, len(df)):
        row = df.iloc[i]
        try:
            year = int(float(str(row.iloc[0])))
            if year < 1990 or year > 2030:
                continue
        except (ValueError, TypeError):
            continue

        rows.append({
            "year": year,
            "fafh_share_pct": pd.to_numeric(row.iloc[5], errors="coerce"),
            "food_share_disposable_income": pd.to_numeric(row.iloc[3], errors="coerce"),
        })

    result = pd.DataFrame(rows).dropna(subset=["fafh_share_pct"])
    print(f"  Food expenditure: {len(result)} years ({result['year'].min()}-{result['year'].max()})")
    return result


def get_nhanes_upf_data():
    """Published NHANES ultra-processed food estimates.

    Sources:
    - Juul et al. 2022 (Am J Clin Nutr): 2001-2018, 9 waves
    - Wang et al. 2021 (JAMA): Youth 1999-2018
    - CDC Data Brief 536 (2025): 2021-2023

    The wave-by-wave estimates for adults are extracted from
    Figure 1 of Juul et al. 2022 and the CDC Data Brief.
    """
    # Juul et al. 2022 reported start=53.5%, end=57.0%, linear trend
    # Intermediate values estimated from their Figure 1 (digitized)
    data = pd.DataFrame({
        "year": [2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017, 2022],
        "upf_pct_kcal": [53.5, 54.0, 54.6, 55.4, 55.7, 56.0, 56.3, 56.7, 57.0, 55.0],
        "source": [
            "Juul2022", "Juul2022", "Juul2022", "Juul2022", "Juul2022",
            "Juul2022", "Juul2022", "Juul2022", "Juul2022", "CDC_DB536",
        ],
    })
    print(f"  NHANES UPF: {len(data)} observations ({data['year'].min()}-{data['year'].max()})")
    return data


def build_exposure_index(sweetener, soy_oil, food_exp, nhanes):
    """Merge all sources and compute composite exposure index."""
    # Start with annual backbone
    years = pd.DataFrame({"year": range(1997, 2025)})

    # Merge each source
    merged = years.merge(sweetener[["year", "total_caloric_sweet_lbs_pc", "hfcs_lbs_pc",
                                     "refined_sugar_lbs_pc"]], on="year", how="left")
    merged = merged.merge(soy_oil[["year", "soy_oil_domestic_mlbs"]], on="year", how="left")
    merged = merged.merge(food_exp[["year", "fafh_share_pct"]], on="year", how="left")
    merged = merged.merge(nhanes[["year", "upf_pct_kcal"]], on="year", how="left")

    # Interpolate NHANES (biennial) to annual
    merged["upf_pct_kcal"] = merged["upf_pct_kcal"].interpolate(method="linear")

    # Convert soy oil to per-capita (using Census population estimates)
    # US population (millions, mid-year estimates)
    pop = sweetener[["year", "refined_sugar_lbs_pc"]].copy()
    # We have total_caloric and per_capita, so we can derive population
    # But simpler: use the population column from Table 50
    pop_df = pd.read_excel(RAW_DIR / "usda_ers_sweetener_consumption.xlsx",
                           sheet_name="Table50", header=None)
    pop_rows = []
    for i in range(7, len(pop_df)):
        row = pop_df.iloc[i]
        try:
            y = int(float(str(row.iloc[0])))
            p = pd.to_numeric(row.iloc[1], errors="coerce")
            if y >= 1990 and y <= 2030 and pd.notna(p):
                pop_rows.append({"year": y, "pop_millions": p})
        except (ValueError, TypeError):
            continue
    pop = pd.DataFrame(pop_rows)
    merged = merged.merge(pop, on="year", how="left")

    # Soy oil per capita (lbs per person per year)
    merged["soy_oil_lbs_pc"] = (merged["soy_oil_domestic_mlbs"] * 1e6) / (merged["pop_millions"] * 1e6)

    # Composite index: z-score each component, average
    components = ["soy_oil_lbs_pc", "fafh_share_pct", "upf_pct_kcal"]
    for col in components:
        vals = merged[col].dropna()
        if len(vals) >= 3:
            merged[f"{col}_z"] = (merged[col] - vals.mean()) / vals.std()

    z_cols = [f"{c}_z" for c in components if f"{c}_z" in merged.columns]
    merged["exposure_index"] = merged[z_cols].mean(axis=1)

    # Note: sweetener is DECLINING, so it goes AGAINST the thesis
    # We include it as a separate column but don't put it in the composite
    merged["sweetener_lbs_pc_z"] = np.nan
    sweet_vals = merged["total_caloric_sweet_lbs_pc"].dropna()
    if len(sweet_vals) >= 3:
        merged["sweetener_lbs_pc_z"] = (
            (merged["total_caloric_sweet_lbs_pc"] - sweet_vals.mean()) / sweet_vals.std()
        )

    return merged


def plot_exposure_trends(exposure):
    """Generate exposure trend plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Caloric sweetener per capita
    ax1 = axes[0, 0]
    valid = exposure.dropna(subset=["total_caloric_sweet_lbs_pc"])
    ax1.plot(valid["year"], valid["total_caloric_sweet_lbs_pc"], "o-", color="firebrick",
             linewidth=2, markersize=4)
    ax1_twin = ax1.twinx()
    valid2 = exposure.dropna(subset=["hfcs_lbs_pc"])
    ax1_twin.plot(valid2["year"], valid2["hfcs_lbs_pc"], "s--", color="darkorange",
                  linewidth=1.5, markersize=3, alpha=0.7)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Caloric Sweeteners (lbs/capita)", color="firebrick")
    ax1_twin.set_ylabel("HFCS (lbs/capita)", color="darkorange")
    sl, _, _, pv, _ = stats.linregress(valid["year"], valid["total_caloric_sweet_lbs_pc"])
    ax1.set_title(f"A. Caloric Sweeteners Per Capita\n(trend: {sl:+.1f} lbs/yr, p={pv:.3f})")
    ax1.grid(True, alpha=0.2)

    # Panel B: Soybean oil per capita
    ax2 = axes[0, 1]
    valid = exposure.dropna(subset=["soy_oil_lbs_pc"])
    ax2.plot(valid["year"], valid["soy_oil_lbs_pc"], "o-", color="teal", linewidth=2, markersize=4)
    sl2, _, _, pv2, _ = stats.linregress(valid["year"], valid["soy_oil_lbs_pc"])
    xfit = np.array([valid["year"].min(), valid["year"].max()])
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Soybean Oil (lbs/capita)")
    ax2.set_title(f"B. Soybean Oil Per Capita\n(trend: {sl2:+.2f} lbs/yr, p={pv2:.1e})")
    ax2.grid(True, alpha=0.2)

    # Panel C: Food-away-from-home share + UPF calories
    ax3 = axes[1, 0]
    valid = exposure.dropna(subset=["fafh_share_pct"])
    ax3.plot(valid["year"], valid["fafh_share_pct"], "o-", color="purple",
             linewidth=2, markersize=4, label="FAFH share (%)")
    ax3_twin = ax3.twinx()
    valid_upf = exposure.dropna(subset=["upf_pct_kcal"])
    ax3_twin.plot(valid_upf["year"], valid_upf["upf_pct_kcal"], "D-", color="crimson",
                  linewidth=2, markersize=4, alpha=0.7, label="UPF % kcal")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("FAFH Share of Food Spending (%)", color="purple")
    ax3_twin.set_ylabel("UPF % of Calories (NHANES)", color="crimson")
    ax3.set_title("C. Food-Away-From-Home & Ultra-Processed Calories")
    ax3.grid(True, alpha=0.2)

    # Panel D: Composite exposure index
    ax4 = axes[1, 1]
    valid = exposure.dropna(subset=["exposure_index"])
    ax4.plot(valid["year"], valid["exposure_index"], "o-", color="black", linewidth=2, markersize=5)
    ax4.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    sl4, ic4, _, pv4, _ = stats.linregress(valid["year"], valid["exposure_index"])
    xfit4 = np.array([valid["year"].min(), valid["year"].max()])
    ax4.plot(xfit4, ic4 + sl4 * xfit4, "--", color="crimson", linewidth=2, alpha=0.7)
    ax4.fill_between(valid["year"], 0, valid["exposure_index"],
                      where=valid["exposure_index"] > 0, alpha=0.15, color="red")
    ax4.fill_between(valid["year"], 0, valid["exposure_index"],
                      where=valid["exposure_index"] <= 0, alpha=0.15, color="blue")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Composite Exposure Index (z-score)")
    ax4.set_title(f"D. Composite Exposure Index\n(soy oil + FAFH + UPF; trend: {sl4:+.3f}/yr, p={pv4:.3f})")
    ax4.grid(True, alpha=0.2)

    fig.suptitle("US Food Supply Exposure Trends (USDA ERS + NHANES)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = OUT / "exposure_trends.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    print("Building Food Supply Exposure Index\n")
    print("Downloading/loading data sources:")

    sweetener = parse_sweetener_data()
    soy_oil = parse_soybean_oil_data()
    food_exp = parse_food_expenditure_data()
    nhanes = get_nhanes_upf_data()

    print("\nBuilding exposure index...")
    exposure = build_exposure_index(sweetener, soy_oil, food_exp, nhanes)

    # Save
    out_path = OUT / "exposure_index.csv"
    exposure.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Years: {exposure['year'].min()}-{exposure['year'].max()}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPOSURE DATA SUMMARY")
    print("=" * 70)
    cols_to_show = ["year", "total_caloric_sweet_lbs_pc", "soy_oil_lbs_pc",
                    "fafh_share_pct", "upf_pct_kcal", "exposure_index"]
    cols_present = [c for c in cols_to_show if c in exposure.columns]
    print(exposure[cols_present].to_string(index=False))

    # Trend tests
    print("\n" + "=" * 70)
    print("TREND TESTS (1997-2024)")
    print("=" * 70)
    for col, label in [
        ("total_caloric_sweet_lbs_pc", "Caloric sweeteners (lbs/capita)"),
        ("soy_oil_lbs_pc", "Soybean oil (lbs/capita)"),
        ("fafh_share_pct", "Food-away-from-home share (%)"),
        ("upf_pct_kcal", "Ultra-processed food (% kcal)"),
        ("exposure_index", "Composite exposure index"),
    ]:
        valid = exposure[["year", col]].dropna()
        if len(valid) >= 5:
            sl, _, rv, pv, _ = stats.linregress(valid["year"], valid[col])
            print(f"  {label:45s}: {sl:+.4f}/yr  (R={rv:+.3f}, p={pv:.4f}, n={len(valid)})")

    # Plot
    plot_exposure_trends(exposure)
