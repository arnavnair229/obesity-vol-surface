"""
Stress tests on the national exposure result. Co-trending series correlate strongly whether or not
they are related, so these checks remove the shared trend and respect the time ordering.
Output: outputs/robustness_results.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
m = pd.read_csv(ROOT / "outputs" / "merged_timeseries.csv").dropna(subset=["obesity_pct", "cum_exposure_index"])
y, x, t = m["obesity_pct"].to_numpy(), m["cum_exposure_index"].to_numpy(), m["year"].to_numpy()


def detrend(a):
    return a - np.polyval(np.polyfit(t, a, 1), t)


def block_ci(a, b, block=3, n=10000, seed=7):
    rng = np.random.default_rng(seed)
    k = int(np.ceil(len(a) / block))
    out = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, len(a) - block + 1, k)])[: len(a)]
        if np.std(a[idx]) > 0 and np.std(b[idx]) > 0:
            out.append(np.corrcoef(a[idx], b[idx])[0, 1])
    return np.percentile(out, [2.5, 97.5])


def trend_only():
    """How high r gets when the 'exposure' is unrelated noise, cumulated the same way."""
    rng = np.random.default_rng(7)
    rs = [abs(np.corrcoef(y, np.cumsum(rng.normal(size=len(y))))[0, 1]) for _ in range(10000)]
    return float(np.mean(np.array(rs) >= abs(np.corrcoef(y, x)[0, 1])))


rows = []
r, p = stats.pearsonr(y, x)
lo, hi = block_ci(y, x)
rows.append(("Levels, as in the headline", r, p, f"block bootstrap 95% CI [{lo:.2f}, {hi:.2f}]"))
r, p = stats.pearsonr(t, y)
rows.append(("Obesity vs calendar year alone", r, p, "a plain time trend, no food data"))
r, p = stats.pearsonr(detrend(y), detrend(x))
rows.append(("Both series linearly detrended", r, p, "removes the shared trend"))
d = m.dropna(subset=["d_obesity_pct", "d_exposure_index"])
r, p = stats.pearsonr(d["d_obesity_pct"], d["d_exposure_index"])
rows.append(("Year-over-year changes", r, p, "annual obesity change vs annual exposure change"))
share = trend_only()
rows.append(("Cumulated random noise placebo", np.nan, share, "share of 10,000 random walks matching the headline |r|"))

out = pd.DataFrame(rows, columns=["test", "pearson_r", "p_value", "note"])
out.to_csv(ROOT / "outputs" / "robustness_results.csv", index=False)
print(out.to_string(index=False))
