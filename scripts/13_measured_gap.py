"""
How far is self-reported BRFSS obesity from measured NHANES obesity, and what does that do to the strikes?

The gap mixes self-report bias with other differences: NHANES is age 20+ and age-adjusted, BRFSS is age 18+
and unadjusted, and the two surveys differ in mode and response. It is national, so it can't correct a state.
NHANES ends in August 2023, so it says nothing about the years since.

Outputs:
  - outputs/measured_gap.csv
  - outputs/measured_strikes.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "outputs"
MEASURED_STRIKES = [0.40, 0.42, 0.44, 0.46, 0.48]
HORIZONS = [1, 3, 5]


def logit(p):
    return np.log(p / (1 - p))


def gap_table():
    bench = pd.read_csv(RAW / "external_obesity_benchmarks.csv")
    bench = bench[bench.source == "NHANES"]
    us = pd.read_csv(RAW / "cdc_brfss_latest.csv")
    us = us[(us.group == "Total") & (us.location == "US")].set_index("year").percent
    rows = []
    for r in bench.itertuples():
        brfss = us.loc[r.period_start:r.period_end].mean()
        # The delta method carries the NHANES standard error onto the logit scale. BRFSS's is far smaller.
        se_logit = (r.se / 100) / (r.percent / 100 * (1 - r.percent / 100))
        rows.append({"cycle": r.period, "nhanes_pct": r.percent, "nhanes_se": r.se, "brfss_pct": brfss, "gap_points": r.percent - brfss,
                     "gap_logit": logit(r.percent / 100) - logit(brfss / 100), "gap_logit_se": se_logit})
    return pd.DataFrame(rows)


def main():
    gaps = gap_table()
    weight = 1 / gaps.gap_logit_se ** 2
    pooled = float((weight * gaps.gap_logit).sum() / weight.sum())
    pooled_se = float(np.sqrt(1 / weight.sum()))
    q = float((weight * (gaps.gap_logit - pooled) ** 2).sum())
    gaps["pooled_gap_logit"], gaps["pooled_se"], gaps["constant_gap_p"] = pooled, pooled_se, chi2.sf(q, len(gaps) - 1)
    gaps.round(4).to_csv(OUT / "measured_gap.csv", index=False)

    frozen = pd.read_csv(RAW / "cdc_brfss_obesity_total_us_states.csv")
    us = frozen[frozen.locationabbr == "US"].set_index("yearstart").p
    wide = frozen[~frozen.locationabbr.isin(["US", "GU", "PR", "VI"])].pivot(index="locationabbr", columns="yearstart", values="p")
    spot = logit(us[2024])
    rows = []
    for T in HORIZONS:
        change = np.concatenate([(logit(wide[y + T]) - logit(wide[y])).dropna().to_numpy() for y in wide.columns if y + T in wide.columns])
        for k in MEASURED_STRIKES:
            row = {"T": T, "measured_strike": k}
            equivalent = 1 / (1 + np.exp(-(logit(k) - pooled)))
            row["self_report_equivalent"] = equivalent
            row["p_cross"] = norm.cdf((spot + change.mean() + pooled - logit(k)) / np.sqrt(change.var(ddof=1) + pooled_se ** 2))
            # The latest gap alone is the low end of what the cycles show.
            latest = gaps.gap_logit.iloc[-1]
            row["p_cross_latest_gap"] = norm.cdf((spot + change.mean() + latest - logit(k)) / np.sqrt(change.var(ddof=1) + gaps.gap_logit_se.iloc[-1] ** 2))
            rows.append(row)
    strikes = pd.DataFrame(rows)
    strikes.round(4).to_csv(OUT / "measured_strikes.csv", index=False)
    print(gaps.round(3).to_string(index=False))
    print(f"Spot, self-report 2024: {us[2024]:.3f}; on the measured basis: {1 / (1 + np.exp(-(spot + pooled))):.3f}")
    print(strikes.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
