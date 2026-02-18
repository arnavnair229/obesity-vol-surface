"""
Plot US Obesity Prevalence Trends

Generates:
  - US national prevalence by year (2011-2024)
  - State median + IQR over time

Outputs:
  - outputs/obesity_trends.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
INP = PROJECT / "data" / "raw" / "cdc_brfss_obesity_total_us_states.csv"
OUT = PROJECT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INP)

# US national series
us = df[df["locationabbr"] == "US"].sort_values("yearstart")

# State panel (exclude territories and national aggregate)
states = df[~df["locationabbr"].isin(["US", "GU", "PR", "VI"])].copy()
state_stats = states.groupby("yearstart")["p"].agg(
    median="median",
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: US national
ax1 = axes[0]
ax1.plot(us["yearstart"], us["p"] * 100, "o-", color="crimson", linewidth=2, markersize=6)
ax1.set_xlabel("Year")
ax1.set_ylabel("Obesity Prevalence (%)")
ax1.set_title("US National Adult Obesity Prevalence")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(20, 40)

# Plot 2: State distribution
ax2 = axes[1]
ax2.fill_between(
    state_stats["yearstart"],
    state_stats["q25"] * 100,
    state_stats["q75"] * 100,
    alpha=0.3, color="steelblue", label="State IQR (25th-75th)",
)
ax2.plot(
    state_stats["yearstart"], state_stats["median"] * 100,
    "o-", color="steelblue", linewidth=2, markersize=6, label="State Median",
)
ax2.set_xlabel("Year")
ax2.set_ylabel("Obesity Prevalence (%)")
ax2.set_title("State-Level Obesity Distribution Over Time")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(20, 45)

plt.tight_layout()
out_path = OUT / "obesity_trends.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()

# Print summary
print(f"\nUS National Obesity Prevalence ({us['yearstart'].min()}-{us['yearstart'].max()}):")
for _, row in us.iterrows():
    print(f"  {int(row['yearstart'])}: {row['p']*100:.1f}%")
