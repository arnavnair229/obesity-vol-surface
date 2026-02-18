"""
Build Additive Load Index from Open Food Facts (US products)

Downloads US product data via the Open Food Facts API and computes:
  - Mean additive count by year
  - Mean ingredient count by year
  - NOVA ultra-processed (group 4) share by year

Outputs:
  - data/raw/openfoodfacts_us_sample.csv (cached API results)
  - outputs/additive_load_by_year.csv
  - outputs/additive_trends.png

Note: "year" = year the product was added to the Open Food Facts database
(proxy for market availability). This is a selection-bias caveat — see README.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / "data" / "raw"
OUT = PROJECT / "outputs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

CACHE_PATH = RAW_DIR / "openfoodfacts_us_sample.csv"

FIELDS = ",".join([
    "code",
    "product_name",
    "created_t",
    "additives_n",
    "ingredients_n",
    "nova_group",
    "nutrition_grades",
    "additives_tags",
])

MAX_PAGES = 200       # up to 20,000 products
PAGE_SIZE = 100
RATE_LIMIT_S = 0.2    # seconds between requests


def download_off_data():
    """Download US products from Open Food Facts API using popularity sort
    for a diverse cross-year sample."""
    print("Downloading US products from Open Food Facts...")
    print(f"  Max pages: {MAX_PAGES}, page size: {PAGE_SIZE}")

    # Use the classic search endpoint which gives better cross-year coverage
    # when sorted by popularity (popular products span many creation years)
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    all_products = []

    for page in range(1, MAX_PAGES + 1):
        params = {
            "action": "process",
            "tagtype_0": "countries",
            "tag_contains_0": "contains",
            "tag_0": "united-states",
            "sort_by": "unique_scans_n",
            "page_size": PAGE_SIZE,
            "page": page,
            "json": 1,
            "fields": FIELDS,
        }

        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.RequestException as e:
            print(f"  Page {page}: request error ({e}), stopping")
            break
        except ValueError:
            print(f"  Page {page}: invalid JSON response, stopping")
            break

        products = data.get("products", [])
        if not products:
            print(f"  Page {page}: no more products, done")
            break

        all_products.extend(products)

        if page % 25 == 0:
            print(f"  Page {page}: {len(all_products)} products fetched")

        time.sleep(RATE_LIMIT_S)

    print(f"  Total products downloaded: {len(all_products)}")

    df = pd.DataFrame(all_products)
    df.to_csv(CACHE_PATH, index=False)
    print(f"  Cached to: {CACHE_PATH}")
    return df


def process_data(df):
    """Process raw OFF data into yearly additive metrics."""
    if "created_t" not in df.columns:
        print("ERROR: no created_t column in data")
        return pd.DataFrame()

    df["created_t"] = pd.to_numeric(df["created_t"], errors="coerce")
    df = df.dropna(subset=["created_t"])
    df = df[df["created_t"] > 0]
    df["year"] = df["created_t"].apply(lambda t: datetime.fromtimestamp(t).year)

    # Convert numeric fields
    for col in ["additives_n", "ingredients_n", "nova_group"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to reasonable year range
    df = df[(df["year"] >= 2012) & (df["year"] <= 2026)]

    # Drop rows with no usable data
    has_data = df[["additives_n", "ingredients_n", "nova_group"]].notna().any(axis=1)
    df = df[has_data]

    print(f"\nProcessed products with data: {len(df)}")
    if len(df) == 0:
        return pd.DataFrame()

    print(f"Year range: {df['year'].min()}-{df['year'].max()}")
    print(f"Products with additives_n: {df['additives_n'].notna().sum()}")
    print(f"Products with ingredients_n: {df['ingredients_n'].notna().sum()}")
    print(f"Products with nova_group: {df['nova_group'].notna().sum()}")

    # Aggregate by year
    yearly = df.groupby("year").agg(
        n_products=("code", "count"),
        mean_additives=("additives_n", "mean"),
        median_additives=("additives_n", "median"),
        pct_with_additives=("additives_n", lambda x: (x > 0).mean()),
        mean_ingredients=("ingredients_n", "mean"),
        median_ingredients=("ingredients_n", "median"),
        nova4_share=("nova_group", lambda x: (x == 4).mean()),
        mean_nova=("nova_group", "mean"),
    ).reset_index()

    return yearly


def plot_trends(yearly):
    """Generate additive trend plots."""
    yearly = yearly[yearly["n_products"] >= 20].copy()

    if len(yearly) < 3:
        print("Not enough yearly data for meaningful plots (need >= 3 years with >= 20 products)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Additive count
    ax1 = axes[0]
    ax1.plot(yearly["year"], yearly["mean_additives"], "o-", color="darkorange", linewidth=2)
    ax1.set_xlabel("Year (product added to database)")
    ax1.set_ylabel("Mean Additive Count")
    ax1.set_title("Additives per Product Over Time")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Ingredient count
    ax2 = axes[1]
    ax2.plot(yearly["year"], yearly["mean_ingredients"], "s-", color="teal", linewidth=2)
    ax2.set_xlabel("Year (product added to database)")
    ax2.set_ylabel("Mean Ingredient Count")
    ax2.set_title("Ingredients per Product Over Time")
    ax2.grid(True, alpha=0.3)

    # Plot 3: NOVA 4 share
    ax3 = axes[2]
    ax3.bar(yearly["year"], yearly["nova4_share"] * 100, color="firebrick", alpha=0.7)
    ax3.set_xlabel("Year (product added to database)")
    ax3.set_ylabel("NOVA 4 Share (%)")
    ax3.set_title("Ultra-Processed (NOVA 4) Share")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT / "additive_trends.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    # Check for cached data
    if CACHE_PATH.exists():
        print(f"Loading cached data from: {CACHE_PATH}")
        df = pd.read_csv(CACHE_PATH)
    else:
        df = download_off_data()

    yearly = process_data(df)

    if yearly.empty:
        print("\nNo processable data found. Try deleting the cache and re-downloading:")
        print(f"  rm {CACHE_PATH}")
        print("  python scripts/04_additive_index.py")
    else:
        # Save
        out_path = OUT / "additive_load_by_year.csv"
        yearly.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")
        print(yearly.to_string(index=False))

        # Plot
        plot_trends(yearly)
