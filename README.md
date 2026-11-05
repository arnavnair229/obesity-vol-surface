# US Obesity Threshold Surface

Option-pricing tools applied to public health data: a surface of how likely US obesity prevalence is to cross a given level over a given horizon, built from CDC state data and backtested out of sample.

The backtest found that a default setting had been distorting the surface. That is fixed here, and the old numbers stay on record below. Food data is included as a hypothesis, along with the tests it failed.

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
python scripts/08_backtest.py               # out-of-sample test of the surface
python scripts/09_download_food_environment.py  # optional, the Census data is already in data/raw
python scripts/10_state_food_panel.py       # restaurant density by state
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

1. For each $(T, K)$, fit an unpenalized logistic regression for $\Pr(p_{t+T} \geq K \mid S_t)$ on the state panel and evaluate it at the 2024 national level. Probabilities are forced to fall as the strike rises.
2. Read the implied forward $m_T$ off that curve, at the strike where the probability is 50%.
3. Invert: $\sigma(T, K) = \dfrac{K_{\text{logit}} - m_T}{\Phi^{-1}(1 - \hat P)\,\sqrt{T}}$. Near the money, where numerator and denominator both approach zero, use the curve's local slope.
4. Leave a cell empty when $\hat P$ is outside 1% to 99%. The curve is steep enough that only strikes from 32% to 40% have values, none at every horizon. Nothing is interpolated into the empty cells.

**Why the implied forward:** earlier versions took drift from a separate average of nearby states. The numerator and denominator then hit zero at different strikes, which put a false spike in the surface (0.30 at K=35%, T=5y, next to values around 0.03) and a false "ATM ridge". Using one curve for both removes it.

**Implied sigma (logit scale, blank where the probability is outside 1% to 99%):**

| T \\ K | 32% | 33% | 34% | 35% | 36% | 37% | 38% | 40% |
|---|---|---|---|---|---|---|---|---|
| 1y | 0.066 | 0.062 | 0.074 | 0.053 | 0.064 | 0.066 | | |
| 3y | 0.038 | 0.034 | 0.026 | 0.036 | 0.033 | 0.033 | 0.036 | |
| 5y | | 0.029 | 0.024 | 0.024 | 0.030 | 0.024 | 0.025 | |
| 10y | | | | 0.024 | 0.023 | 0.024 | 0.027 | 0.019 |

Sigma falls with maturity. Across strikes it is roughly flat, with noise near the forward, where small errors in the numerator and denominator matter most. An earlier version of this README described a smile. Most of that shape came from the penalty described below.

**Crossing probabilities, for a state at today's national level:**

| K | T = 1y | T = 3y | T = 5y | T = 10y |
|---|---|---|---|---|
| 32% | 88.9% | 97.8% | | |
| 34% | 45.1% | 81.5% | 94.6% | |
| 35% | 15.6% | 47.6% | 79.0% | 98.2% |
| 36% | 6.4% | 20.4% | 49.3% | 94.3% |
| 38% | | 1.7% | 6.2% | 63.4% |
| 40% | | | | 17.8% |

The implied forward rises from 33.8% at 1 year to 38.7% at 10 years.

## Backtest

At each forecast date, every model was fit only on pairs whose outcome was already known, then used to forecast each state's crossings T years later (16,533 forecasts, forecast dates 2013 to 2023, strikes 25% to 40%). Lower Brier scores are better. "Contested" cells start within 3 points of the strike, a rule that doesn't depend on any model's forecast.

| Horizon | Forecast dates | Old fit, C=1 | Current fit | Random walk with drift | No change |
|---|---|---|---|---|---|
| 1y | 11 | 0.085 | 0.050 | 0.049 | 0.072 |
| 2y | 9 | 0.099 | 0.057 | 0.055 | 0.094 |
| 3y | 7 | 0.116 | 0.061 | 0.060 | 0.124 |
| 5y | 3 | 0.139 | 0.068 | 0.063 | 0.173 |
| 1y, contested | 11 | 0.217 | 0.138 | 0.137 | 0.204 |
| 3y, contested | 7 | 0.279 | 0.144 | 0.142 | 0.326 |
| 5y, contested | 3 | 0.287 | 0.115 | 0.102 | 0.392 |

**What was wrong:** scikit-learn's `LogisticRegression` applies an L2 penalty by default. On an unstandardized logit with a spread of about 0.2, that penalty is heavy: at T=1, K=32% it cut the fitted slope from about 29 to about 10. The flattened curve badly under-forecast likely crossings, lost to assuming no change at 1 and 2 years across all cells, and set much of the implied sigma. At T=1, K=35%, sigma was 0.127 and is now 0.053. The T=5, K=35% crossing probability was 52.5% and is now 79.0%.

| Forecast, old fit | Actual | Forecast, current fit | Actual | Forecast, random walk | Actual |
|---|---|---|---|---|---|
| 32% | 49% | 33% | 41% | 32% | 37% |
| 50% | 71% | 50% | 53% | 50% | 50% |
| 68% | 91% | 68% | 65% | 68% | 66% |
| 83% | 97% | 83% | 78% | 83% | 82% |

![Calibration chart](outputs/backtest_calibration.png)

**How to read this:**

- Intervals resample forecast dates and states, because forecasts share national shocks. The old fit's gap to the random walk excludes zero at every horizon. The current fit's gap includes zero at every horizon, so I can't say the random walk is better, though its point estimates are lower at all of them. Details are in `outputs/backtest_scores.csv`.
- The penalty was found with this backtest and fixed afterward, and these are the same years the fix is scored on. No parameter was tuned to them, but this isn't a clean holdout.
- The 5-year rows rest on 3 forecast dates. The 7- and 10-year columns of the surface can't be tested at all, because the data starts in 2011.
- State forecasts are the main test. The US series has far too few independent outcomes, though its scores in `outputs/backtest_national.csv` point the same way.

**How far to trust the surface:**

- These are historical-frequency probabilities, not risk-neutral ones. Nothing here trades.
- They describe a state at today's national level, not the national rate itself.
- State-years overlap and share national shocks, so the effective sample is much smaller than the row count.
- The fit extrapolates recent history, and history may have shifted. In contested cells, the random walk under-forecast crossings at every horizon when outcomes ended before 2023 (by 1 to 8 points on average) and over-forecast them when outcomes ended in 2023 or 2024 (by 3 to 14 points). The state average has been flat since 2021 (33.5%, 33.8%, 33.6%, 33.8%). That is only two outcome years, and it doesn't show a cause, so treat it as a lead and not a finding.
- BRFSS is self-reported. CDC's measured survey put adult obesity at [40.3% for August 2021 to August 2023](https://www.cdc.gov/nchs/products/databriefs/db508.htm), so these strikes are not measured-obesity levels.

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

### A state-level test

Testing this properly needs exposure that varies by state, so I tried limited-service restaurant density from Census County Business Patterns (NAICS 722513, per capita, 2012 to 2023, 51 states). Year fixed effects can then remove every national trend.

| Specification | Coefficient | t |
|---|---|---|
| State effects only, like the panel above | +34.4 | +13.1 |
| State and year effects | +4.3 | +1.8 |
| With income and inactivity | +2.6 | +1.3 |
| Density 2 years earlier | +1.8 | +0.7 |
| Density 4 years earlier | -1.9 | -0.5 |
| **Placebo: density 2 years later** | **+6.4** | **+2.8** |
| Change from 2012 to 2023, across states | +0.6 | +0.2 |

Coefficients are percentage points of obesity per log unit of density, with errors clustered by state (robust errors in the last row).

**This doesn't corroborate the food story.** The t of 13 falls to 1.8 once national trends are removed, which also shows how a t of 10 in the exposure panel can come from trend alone. Future density predicts obesity at least as well as past density does. That held on one matched sample of 2014 to 2021 (t = +1.2 for 2 years earlier, +2.4 for the same year, +2.3 for 2 years later), and with all three in one regression none is significant. A real effect should run from past density forward, so this points to pre-trends or reverse causality.

Income held up better. States with faster income growth saw smaller rises in obesity (-13.7 per log unit of income, t = -3.9, 49 states). That is an association, and I chose to look at it after seeing the data.

Limits: establishment counts aren't what people eat, density barely changes year to year, so its lags and leads are hard to separate, and failing with this one proxy doesn't rule out a food effect. A natural experiment would be a stronger test.

## Data sources

| Source | Coverage | Used for |
|---|---|---|
| [CDC BRFSS](https://data.cdc.gov/resource/hn4x-zwk7) | 50 states + DC, 2011-2024 | Obesity prevalence, physical inactivity |
| [Census ACS](https://api.census.gov/) | 50 states + DC, 2011-2023 | Median household income |
| [Census County Business Patterns](https://www2.census.gov/programs-surveys/cbp/datasets/) | 50 states + DC, 2012-2023 | Restaurant and grocery establishments |
| [Census population estimates](https://www2.census.gov/programs-surveys/popest/datasets/) | 50 states + DC, 2012-2023 | Per-capita density |
| [USDA ERS Sugar & Sweeteners](https://www.ers.usda.gov/data-products/sugar-and-sweeteners-yearbook-tables/) | 1966-2024 | Caloric sweeteners |
| [USDA ERS Oil Crops](https://www.ers.usda.gov/data-products/oil-crops-yearbook/) | 1999-2024 | Soybean oil |
| [USDA ERS Food Expenditure](https://www.ers.usda.gov/data-products/food-expenditure-series/) | 1997-2024 | Food-away-from-home share |
| Juul et al. 2022 (*Am J Clin Nutr*), CDC Data Brief 536 | 2001-2022 | Ultra-processed food share |

## Outputs

| File | Contents |
|---|---|
| `outputs/health_vol_surface_model1.csv` | Full surface with probabilities, forwards and notes |
| `outputs/health-surface-data.json` | Data behind the wireframe on [my site](https://arnavnair229.github.io) |
| `outputs/backtest_scores.csv`, `outputs/backtest_calibration.csv`, `outputs/backtest_by_origin.csv`, `outputs/backtest_national.csv` | Out-of-sample scores, intervals and calibration |
| `outputs/model_comparison_today.csv` | Current fit and random walk at today's level |
| `outputs/state_food_panel_results.csv` | Restaurant density regressions |
| `outputs/robustness_results.csv` | Stress tests on the national result |
| `outputs/exposure_regression_results.csv`, `outputs/panel_regression_results.csv` | Regression output |
| `outputs/*.png` | Figures |
