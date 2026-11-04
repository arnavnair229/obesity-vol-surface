# US Obesity Threshold Surface

Option-pricing tools applied to public health data: a calibrated surface of how likely US obesity prevalence is to cross a given level over a given horizon, built from CDC state data.

The surface is the main result. A second analysis of food-supply data is included as a hypothesis, along with the tests that show why it can't support a causal claim.

## Run it

```bash
pip install -r requirements.txt
python scripts/01_download_cdc_obesity.py   # optional, the raw data is already in data/raw
python scripts/build_all.py                 # surface
python scripts/03_obesity_trends.py         # trend plots
python scripts/04_exposure_data.py          # exposure index
python scripts/05_exposure_regression.py    # national analysis
python scripts/06_panel_regression.py       # state panel
python scripts/07_robustness.py             # stress tests on the national result
```

## The surface

US adult obesity rose from 27.4% in 2011 to 33.1% in 2024 (CDC BRFSS).

| Concept | Options analogy | Here |
|---|---|---|
| Underlying $S_t$ | Stock price | $\text{logit}(\text{prevalence})$ |
| Strike $K$ | Option strike | Obesity threshold, 25% to 45% |
| Maturity $T$ | Time to expiry | Horizon, 1 to 10 years |
| Implied forward $m_T$ | Forward price | The strike with a 50% crossing probability |
| Implied $\sigma$ | Implied vol | Uncertainty implied by the crossing curve |

**Construction:**

1. For each $(T, K)$, fit a logistic regression for $\Pr(p_{t+T} \geq K \mid S_t)$ on the state panel and evaluate it at the 2024 national level. Probabilities are forced to fall as the strike rises.
2. Read the implied forward $m_T$ off that curve, at the strike where the probability is 50%.
3. Invert: $\sigma(T, K) = \dfrac{K_{\text{logit}} - m_T}{\Phi^{-1}(1 - \hat P)\,\sqrt{T}}$. Near the money, where numerator and denominator both approach zero, use the curve's local slope.
4. Leave a cell empty when $\hat P$ is outside 1% to 99%. K=42% and K=45% are empty at every horizon, and nothing is interpolated into them.

**Why the implied forward:** earlier versions took drift from a separate average of nearby states. The numerator and denominator then hit zero at different strikes, which put a false spike in the surface (0.30 at K=35%, T=5y, next to values around 0.03) and a false "ATM ridge". Using one curve for both removes it.

**Implied sigma (logit scale):**

| T \ K | 25% | 28% | 30% | 32% | 35% | 38% | 40% |
|---|---|---|---|---|---|---|---|
| 1y | 0.191 | 0.153 | 0.135 | 0.151 | 0.127 | 0.133 | 0.141 |
| 3y | 0.120 | 0.097 | 0.083 | 0.082 | 0.072 | 0.069 | 0.073 |
| 5y | 0.100 | 0.081 | 0.071 | 0.062 | 0.044 | 0.049 | 0.051 |
| 10y | 0.079 | 0.068 | 0.057 | 0.050 | 0.036 | 0.033 | 0.034 |

The shape is a smile: lowest near the implied forward, higher in the wings, and falling with maturity.

**Crossing probabilities, for a state at today's national level:**

| K | T = 1y | T = 3y | T = 5y | T = 10y |
|---|---|---|---|---|
| 30% | 87.2% | 91.5% | 93.1% | 94.5% |
| 32% | 65.3% | 76.7% | 84.4% | 89.2% |
| 35% | 27.7% | 40.0% | 52.5% | 70.2% |
| 38% | 6.2% | 8.8% | 12.8% | 24.9% |
| 40% | 2.0% | 2.6% | 3.5% | 7.6% |

The implied forward rises from 33.3% at 1 year to 36.4% at 10 years.

**How far to trust it:**

- These are historical-frequency probabilities, not risk-neutral ones. Nothing here trades.
- They describe a state at today's national level, not the national rate itself.
- State-years overlap and share national shocks, so the effective sample is much smaller than the row count. The 10-year column rests on 199 overlapping observations from only 14 years of data.
- The fit extrapolates recent history. It has no mechanism for a break in trend, such as wider use of GLP-1 drugs.

## Food supply: a hypothesis, not a finding

I built an exposure index from USDA supply data (soybean oil per capita, food-away-from-home share) and published NHANES estimates of ultra-processed calories, then compared its running total with obesity.

| Check | r | p |
|---|---|---|
| Cumulative exposure vs obesity, in levels | +0.964 | < 0.001 |
| **Calendar year alone vs obesity** | **+0.985** | **< 0.001** |
| Both series detrended | -0.187 | 0.52 |
| Year-over-year changes | -0.110 | 0.72 |

**A plain time trend fits better than the food data, and the relationship disappears once the shared trend is removed.** Both series rise steadily, so the high level correlation says little. In a placebo test, about 1.9% of cumulated random-noise series matched the headline correlation, so cumulating adds some apparent fit as well (`outputs/robustness_results.csv`).

The state panel (state fixed effects, clustered errors, 608 observations) gives $t = 10.0$ on cumulative exposure. It doesn't resolve this. Exposure is a single national number per year, so it can't be separated from anything else that trended nationally, and year fixed effects would absorb it entirely. Clustering by state also doesn't account for national shocks common to every state in a year.

Other limits on this analysis:

- Sweeteners were left out of the composite because they are falling, a choice made after seeing the data.
- Physical inactivity losing significance in the panel doesn't show that diet matters more than exercise. It may only mean the trend term absorbs everything.
- Falling sugar supply alongside rising obesity doesn't rule out a role for sugar.

Testing this properly needs treatment that varies by state, or a natural experiment.

## Data sources

| Source | Coverage | Used for |
|---|---|---|
| [CDC BRFSS](https://data.cdc.gov/resource/hn4x-zwk7) | 50 states + DC, 2011-2024 | Obesity prevalence, physical inactivity |
| [Census ACS](https://api.census.gov/) | 50 states + DC, 2011-2023 | Median household income |
| [USDA ERS Sugar & Sweeteners](https://www.ers.usda.gov/data-products/sugar-and-sweeteners-yearbook-tables/) | 1966-2024 | Caloric sweeteners |
| [USDA ERS Oil Crops](https://www.ers.usda.gov/data-products/oil-crops-yearbook/) | 1999-2024 | Soybean oil |
| [USDA ERS Food Expenditure](https://www.ers.usda.gov/data-products/food-expenditure-series/) | 1997-2024 | Food-away-from-home share |
| Juul et al. 2022 (*Am J Clin Nutr*), CDC Data Brief 536 | 2001-2022 | Ultra-processed food share |

## Outputs

| File | Contents |
|---|---|
| `outputs/health_vol_surface_model1.csv` | Full surface with probabilities, forwards and notes |
| `outputs/health-surface-data.json` | Data behind the wireframe on [my site](https://arnavnair229.github.io) |
| `outputs/robustness_results.csv` | Stress tests on the national result |
| `outputs/exposure_regression_results.csv`, `outputs/panel_regression_results.csv` | Regression output |
| `outputs/*.png` | Figures |
