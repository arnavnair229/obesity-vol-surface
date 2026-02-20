# Profit Incentives vs Public Health: A Risk Surface for Obesity Thresholds

A quantitative framework applying options-pricing tools to US obesity prevalence data, constructing a calibrated threshold-crossing risk surface and tracing food supply shifts that track the epidemic.

---

## Abstract

US adult obesity prevalence rose from 27.4% to 33.1% over 2011-2024 (+0.49 pp/year, $p < 10^{-9}$, $R^2 = 0.97$). Using option-style machinery, I model logit-transformed prevalence as a diffusive underlying and invert empirical threshold-crossing probabilities into an implied uncertainty surface over (Maturity $T$, Strike $K$). Conditioning on the national spot via logistic regression produces a clean ATM ridge near $K = 32\%$, analogous to equity vol smiles.

Separately, I construct a Composite Exposure Index from USDA Economic Research Service supply data and published NHANES estimates. At the national level, cumulative food supply exposure tracks obesity with $r = +0.964$ ($R^2 = 93\%$, $p < 10^{-8}$). A state-level panel regression (51 states $\times$ 14 years, $n = 608$) confirms the relationship survives state fixed effects, clustered standard errors, and controls for physical inactivity and household income: **cumulative exposure $t = 10.0$, $p < 0.0001$, within-$R^2 = 79\%$**. Physical inactivity — the standard "lifestyle" explanation — is *not significant* in the panel specification ($p = 0.28$). Both soybean oil ($t = 6.8$) and food-away-from-home share ($t = 5.0$) are independently significant when decomposed.

The most striking pattern is not what's going up — it's what's going *down*. Caloric sweetener consumption has declined 20% since 1999, yet obesity has accelerated. The driver is not "more sugar" but a structural transformation: more seed oils (+37% per capita), more meals eaten away from home (42% to 53% of food spending), and a persistently high share of ultra-processed calories (55-57%).

---

## The Story in Five Facts

### 1. Obesity is accelerating, and it's not slowing down

From 2011 to 2024, US adult obesity prevalence rose from 27.4% to 33.1% — a gain of nearly 6 percentage points in 13 years. The trend is monotonic ($R^2 = 0.97$), the p-value is $10^{-10}$, and there is no inflection point. State-level distributions show the median shifting upward with an expanding IQR. This isn't a few outlier states — it's systemic.

At the current drift, the 35% national threshold — a level already exceeded by ~13 states — has a **70% probability of being reached within 10 years** from any state near today's national average.

### 2. The food supply has been quietly restructured

Using USDA government supply data and published NHANES dietary surveys, I track four indicators of what Americans actually consume:

| Indicator | 2011 | 2024 | Change | Trend |
|---|---|---|---|---|
| **Soybean oil per capita** | 59 lbs | 81 lbs | **+37%** | +1.87 lbs/yr (p < 0.001) |
| **Food-away-from-home share** | 44.5% | 53.0% | **+8.5 pp** | +0.55 pp/yr (p = 0.0001) |
| **Ultra-processed food calories** | 56.0% | 55.0% | -1.0 pp | Peaked ~57% in 2017 |
| **Caloric sweeteners per capita** | 132 lbs | 121 lbs | **-8%** | -0.75 lbs/yr (p < 0.001) |

Americans are eating less sugar than they did 25 years ago. But they're consuming 37% more soybean oil, spending a majority of their food dollars eating out for the first time in history, and getting over half their calories from ultra-processed products. The composition of the food supply has fundamentally shifted.

### 3. Cumulative exposure tracks obesity almost perfectly

Obesity is a stock variable — it represents accumulated metabolic damage, not a one-year snapshot. The correct specification is therefore to compare obesity levels against *cumulative* exposure (the integral of annual exposure deviations from baseline), not first-differenced year-over-year changes.

This framing yields the strongest result in the analysis:

| Model | $r$ | $R^2$ | p-value | 95% Bootstrap CI |
|---|---|---|---|---|
| **Obesity vs Cumulative Exposure** | **+0.964** | **0.930** | $< 10^{-8}$ | **[+0.944, +0.993]** |
| Obesity vs Soy Oil/capita | +0.950 | 0.903 | $< 10^{-7}$ | [+0.920, +0.978] |
| Obesity vs PC1 (common factor) | +0.921 | 0.848 | $< 10^{-5}$ | [+0.886, +0.972] |
| Obesity vs Exposure Index | +0.806 | 0.650 | 0.001 | [+0.510, +0.941] |
| Obesity vs Cumulative FAFH | +0.956 | 0.914 | $< 10^{-7}$ | — |

The cumulative exposure model explains **93% of the variance** in national obesity prevalence. The bootstrap CI lower bound of +0.944 means the relationship holds across resampling — this is not a fragile result.

### 4. One factor captures the whole food supply shift

Principal Component Analysis on the four exposure variables reveals that **82% of total variance** is captured by a single factor (PC1). This factor loads positively on soy oil and FAFH (things going up) and negatively on sweeteners and UPF (things going down or plateauing over the 2011-2024 window). It represents a single axis of food supply transformation.

This common factor correlates $r = +0.921$ with obesity. The food supply is not changing in random directions — it's moving along a single structural axis, and that axis points in the same direction as the obesity epidemic.

### 5. The sugar narrative is wrong

This is the counter-intuitive finding that makes the analysis honest: **caloric sweetener consumption has been declining since 1999** (from 153 to 121 lbs/capita, -21%), yet obesity has accelerated over the same period. The Pearson correlation between sweeteners and obesity is $r = -0.869$ — strongly *negative*.

If "too much sugar" were the primary driver, we would expect the obesity trend to bend downward as sweeteners declined. It hasn't. Whatever is driving the epidemic, it's not total sugar consumption. The data point instead toward seed oils, food-away-from-home consumption, and the broader ultra-processing of the food supply.

### 6. The signal survives a proper panel test

The national time-series results ($n = 14$) are suggestive but could reflect spurious co-trending. To address this, I run a state-level panel regression: 51 states $\times$ 14 years = 608 observations, with state fixed effects (absorb all permanent state characteristics), cluster-robust standard errors, and controls for physical inactivity and household income.

| Model | Exposure Coef | Clustered $t$ | p-value | Within $R^2$ | $n$ |
|---|---|---|---|---|---|
| State FE only | +0.039 | 29.7 | < 0.0001 | 0.561 | 709 |
| + Inactivity | +0.038 | 28.6 | < 0.0001 | 0.587 | 709 |
| + Inactivity + Income | +0.009 | 4.2 | < 0.0001 | 0.749 | 608 |
| **Cumulative Exposure + Controls** | **+0.007** | **10.0** | **< 0.0001** | **0.790** | **608** |
| Soy Oil + FAFH + Controls | see below | — | — | 0.772 | 608 |

In the preferred specification (Model 4), cumulative food supply exposure is the **only significant predictor**:
- Cumulative Exposure: $t = 10.0$, $p < 0.0001$
- Physical Inactivity: $t = -1.1$, **$p = 0.28$** (not significant)
- Household Income: $t = -1.2$, **$p = 0.23$** (not significant)

When decomposed into individual components, both soybean oil per capita ($t = 6.8$, $p < 0.0001$) and FAFH expenditure share ($t = 5.0$, $p < 0.0001$) are independently significant after controlling for each other, inactivity, and income.

The fact that physical inactivity drops out of significance in the panel — while food supply exposure remains dominant — is the most policy-relevant finding. At the population level, what Americans eat appears to matter more than whether they exercise.

---

## Statistical Methodology

### Threshold-Crossing Risk Surface

| Concept | Options Analogy | Health Mapping |
|---|---|---|
| Underlying $S_t$ | Stock price | $\text{logit}(\text{obesity prevalence})$ |
| Strike $K$ | Option strike | Obesity threshold (25%-45%) |
| Maturity $T$ | Time to expiry | Forecasting horizon (1-10 years) |
| Implied vol | Black-Scholes IV | Calibrated crossing uncertainty |
| Spot | Current price | US national 2024: 33.1% ($S_0 = -0.704$) |

**Surface construction (v2: Conditional + Logistic Smoothing):**

1. **Transform**: $S_t = \log\!\bigl(\frac{p_t}{1 - p_t}\bigr)$ where $p_t$ = state-level obesity prevalence
2. **Conditional probability**: For each $(T, K)$, fit a logistic regression $\Pr(p_{t+T} \geq K \mid S_t)$ on the full state panel, then evaluate at $S_t = S_0$
3. **Conditional drift**: $\mu_T = \mathbb{E}[S_{t+T} - S_t \mid S_t \in \text{ATM bucket}]$ using states near the national level
4. **Inversion**: $\sigma(T, K) = \dfrac{K_{\text{logit}} - (S_0 + \mu_T)}{\Phi^{-1}(1 - \hat{P}) \cdot \sqrt{T}}$
5. **NaN handling**: Extreme $\hat{P}$ (<1% or >99%) yields NaN — no clamped fake volatilities

### Exposure Analysis

**Composite Exposure Index**: Z-scored average of soy oil/capita, FAFH share, and UPF % of calories. Each component is standardized against its own time series. Sweeteners are tracked but excluded from the composite (they are declining — including them would dilute the signal from the three increasing components).

**Cumulative Exposure Model**: $\text{Obesity}_t \sim \alpha + \beta \cdot \sum_{i=0}^{t}(\text{Exposure}_i - \text{Exposure}_0)$. Motivation: obesity prevalence is a stock variable reflecting accumulated metabolic damage. First-differencing destroys the signal; cumulative integration preserves it.

**PCA**: StandardScaler $\rightarrow$ PCA on the 4-indicator matrix. PC1 captures 82% of variance — a single structural factor.

**Bootstrap**: 10,000 nonparametric resamples of the (year, obesity, exposure) pairs. 95% CI computed as the 2.5th and 97.5th percentiles of the bootstrapped Pearson r distribution.

**Granger tests**: Nested F-test comparing AR(1) to AR(1) + lagged $\Delta(\text{exposure})$. Non-significant at $n = 14$, reflecting limited power rather than absence of effect.

### Panel Regression

**State fixed effects**: $\text{Obesity}_{st} = \alpha_s + \beta \cdot \text{Exposure}_t + \gamma \cdot \text{Controls}_{st} + \varepsilon_{st}$. State dummies absorb all time-invariant differences (geography, baseline health infrastructure, demographics). The within-estimator identifies the effect of *time-varying* exposure on *within-state* obesity changes.

**Controls**: Physical inactivity rate (CDC BRFSS, state-year level), median household income (Census ACS, state-year level, log-transformed).

**Cluster-robust SEs**: Liang-Zeger sandwich estimator clustering by state (51 clusters), with finite-sample correction $\frac{G}{G-1} \cdot \frac{n-1}{n-k}$. Accounts for serial correlation within states.

---

## Threshold-Crossing Probabilities

Probability of a state-year exceeding threshold $K$, conditional on the 2024 national spot (33.1%):

| K (threshold) | T = 1y | T = 3y | T = 5y | T = 10y |
|---|---|---|---|---|
| 30% | 87.2% | 91.5% | 93.1% | 94.5% |
| 32% | 65.3% | 76.7% | 84.4% | 89.2% |
| 35% | 27.7% | 40.0% | 52.5% | 70.2% |
| 38% | 6.2% | 8.8% | 12.8% | 24.9% |
| 40% | 2.0% | 2.6% | 3.5% | 7.6% |

The implied $\sigma$ surface exhibits an ATM ridge near $K = 32\%$, with $\sigma$ decaying in maturity — consistent with mean-reverting noise around a strong upward drift.

---

## What This Does / Doesn't Claim

**Does:**
- Quantify threshold-crossing probabilities for US obesity using calibrated option-style machinery
- Document a 93% $R^2$ co-movement between cumulative food supply exposure and obesity at the national level
- Confirm the signal in a state-level panel ($n = 608$) with fixed effects and clustered SEs (within-$R^2 = 79\%$)
- Show that the food supply has structurally shifted (more seed oils, more FAFH, less sugar)
- Demonstrate that food supply exposure dominates physical inactivity in the panel specification
- Produce a physical-measure risk surface with honest NaN handling

**Doesn't:**
- Prove causation — the exposure variable is national (no state-level variation in treatment), so identification relies on time variation only
- Claim Granger predictive power — the tests are non-significant at $n = 14$
- Rule out all confounders — income and inactivity are controlled, but urbanization, demographics, and medication use are not
- Imply market pricing or tradeable risk
- Represent risk-neutral probabilities

The panel results are substantially stronger than the national time-series alone: state fixed effects remove permanent cross-sectional confounders, and the signal survives adding income and inactivity controls. The main remaining identification concern is that exposure varies only in the time dimension — any national trend correlated with both exposure and obesity could confound. A natural experiment or instrumental variable design would be needed for causal identification.

---

## Data Sources

| Source | Coverage | Key Metric |
|---|---|---|
| [CDC BRFSS](https://data.cdc.gov/resource/hn4x-zwk7) | 50 states + DC, 2011-2024 | Adult obesity prevalence, physical inactivity (BMI >= 30) |
| [Census ACS](https://api.census.gov/) | 50 states + DC, 2011-2023 | Median household income (state-year) |
| [USDA ERS Sugar & Sweeteners Yearbook](https://www.ers.usda.gov/data-products/sugar-and-sweeteners-yearbook-tables/) | 1966-2024 | Per capita caloric sweetener deliveries (lbs/yr) |
| [USDA ERS Oil Crops Yearbook](https://www.ers.usda.gov/data-products/oil-crops-yearbook/) | 1999-2024 | Soybean oil domestic disappearance (million lbs) |
| [USDA ERS Food Expenditure Series](https://www.ers.usda.gov/data-products/food-expenditure-series/) | 1997-2024 | Food-away-from-home as % of total food spending |
| Juul et al. 2022 (*Am J Clin Nutr*), CDC Data Brief 536 | 2001-2022 | Ultra-processed food % of total calories (NHANES) |

---

## Outputs

| File | Description |
|---|---|
| `outputs/health_vol_surface_model1.csv` | Full surface: $P_{\text{cross}}$, $\mu_T$, $\sigma$, diagnostics |
| `outputs/health-surface-data.json` | Three.js visualization data |
| `outputs/obesity_trends.png` | US national + state-level obesity trends |
| `outputs/exposure_index.csv` | Annual exposure index (USDA ERS + NHANES) |
| `outputs/exposure_trends.png` | 4-panel exposure trend plots |
| `outputs/exposure_outcome_analysis.png` | 8-panel research figure |
| `outputs/exposure_regression_results.csv` | Correlation and regression results |
| `outputs/merged_timeseries.csv` | Merged annual obesity + exposure data |
| `outputs/panel_regression_results.csv` | State-level panel regression coefficients |
| `outputs/panel_regression_figure.png` | 6-panel panel regression figure |

## Usage

```bash
# 1. Download fresh CDC data
python scripts/01_download_cdc_obesity.py

# 2. Build vol surface
python scripts/02_build_surface.py

# 3. Generate obesity trend plots
python scripts/03_obesity_trends.py

# 4. Download USDA ERS data and build exposure index
python scripts/04_exposure_data.py

# 5. Run exposure-outcome analysis (requires steps 2-4)
python scripts/05_exposure_regression.py

# 6. Run state-level panel regression (requires steps 1, 4)
python scripts/06_panel_regression.py

# Or build surface from raw JSON (step 2 standalone)
python scripts/build_all.py
```

## Project Structure

```
scripts/
  01_download_cdc_obesity.py   # CDC BRFSS data download
  02_build_surface.py          # Vol surface (v2: conditional + logistic)
  03_obesity_trends.py         # Prevalence trend plots
  04_exposure_data.py          # USDA ERS + NHANES exposure index
  05_exposure_regression.py    # Cumulative exposure, PCA, bootstrap, research figure
  06_panel_regression.py       # State-level panel with FE, clustered SEs, controls
  build_all.py                 # Surface builder from raw JSON
```

## Visualization

The surface is rendered as an interactive Three.js wireframe on my [portfolio site](https://arnavnair229.github.io).

## Dependencies

```
pip install -r requirements.txt
```
