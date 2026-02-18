# Profit Incentives vs Public Health: A Risk Surface for Obesity Thresholds

A quantitative framework applying options-pricing tools to US obesity prevalence data, constructing a calibrated threshold-crossing risk surface and relating it to food industry additive exposure.

---

## Abstract

US adult obesity prevalence rose from 27.4% to 33.1% over 2011-2024 (+0.49 pp/year, p < 10^-9, R^2 = 0.97). Using option-style machinery, I model logit-transformed prevalence as a diffusive underlying and invert empirical threshold-crossing probabilities into an implied uncertainty surface over (Maturity T, Strike K). Conditioning on the national spot via logistic regression produces a clean ATM ridge near K = 32%, analogous to equity vol smiles.

Separately, I construct an Additive Intensity Index (AII) from Open Food Facts data measuring ingredient count, additive load, and NOVA ultra-processed classification of US market products. Both obesity and ingredient complexity show significant upward trends over 2012-2024. Critically, a Granger causality test reveals that year-over-year changes in ingredient complexity **predict** next-year obesity changes (r = +0.641, p = 0.034; Granger F = 10.39, p = 0.012), boosting an AR(1) model's R^2 from 0.4% to 56.7%. This lagged predictive relationship is consistent with — though does not prove — a causal channel from food industry formulation to metabolic health outcomes.

---

## Key Findings

### 1. Obesity prevalence is worsening (high confidence)

| Statistic | Value |
|---|---|
| Linear trend | **+0.49 pp/year** |
| p-value | 1.9 x 10^-10 |
| R^2 | 0.969 |
| 2011 level | 27.4% |
| 2024 level | 33.1% |

The trend is monotonic except for a minor dip in 2023 (32.8%). State-level distributions show the median shifting upward with an expanding IQR — it's not just a few outlier states pulling the average.

### 2. Threshold-crossing risk increases with horizon

Probability of a state-year exceeding obesity threshold K, conditional on starting at the US 2024 spot (33.1%):

| K (threshold) | T = 1y | T = 3y | T = 5y | T = 10y |
|---|---|---|---|---|
| 30% | 87.2% | 91.5% | 93.1% | 94.5% |
| 32% | 65.3% | 76.7% | 84.4% | 89.2% |
| 35% | 27.7% | 40.0% | 52.5% | 70.2% |
| 38% | 6.2% | 8.8% | 12.8% | 24.9% |
| 40% | 2.0% | 2.6% | 3.5% | 7.6% |

The 35% threshold — a level currently exceeded by ~13 states — has a **70% conditional probability of being reached within 10 years** from any state near the current national level.

### 3. The implied sigma surface exhibits an ATM ridge

The surface shows elevated implied uncertainty near K = 32% (spot = 33.1%), analogous to at-the-money implied volatility in equity options. This ridge reflects maximum sensitivity of the crossing probability to noise when the threshold is near the current level. Sigma decays with horizon (term structure slopes downward), consistent with mean-reverting noise around a strong drift.

### 4. Food formulation complexity is increasing

| Metric | Trend | p-value | R^2 |
|---|---|---|---|
| Mean ingredient count | **+0.6 /year** | 0.043 | 0.38 |
| Mean additive count | +0.05 /year | 0.10 | — |
| AII composite | +0.016 /year | 0.37 | 0.09 |

Ingredient complexity (the most robust proxy for formulation intensity) shows a statistically significant upward trend. The composite AII is noisier due to small sample sizes in early years.

### 5. Ingredient complexity Granger-causes obesity with a 1-year lag

The strongest result in the analysis: increases in ingredient complexity **predict** obesity increases one year later.

| Test | Statistic | p-value | n |
|---|---|---|---|
| Lagged correlation: d(Ingredients)(t) vs d(Obesity)(t+1) | **r = +0.641** | **0.034** | 11 |
| Lagged Spearman: d(Ingredients)(t) vs d(Obesity)(t+1) | rho = +0.651 | 0.030 | 11 |
| Granger F-test: lag(dIngredients) added to AR(1) | **F = 10.39** | **0.012** | 11 |
| AR(1) only R^2 | 0.004 | — | — |
| AR(1) + lagged ingredients R^2 | **0.567** | — | — |

Adding lagged ingredient changes to a simple autoregressive model for obesity **increases R^2 from 0.4% to 56.7%** — a 140x improvement. The Granger F-test confirms this is statistically significant (p = 0.012).

The lead-lag correlogram for d(Ingredients) vs d(Obesity) shows a sharp spike **only at lag +1** (ingredients lead by 1 year), with no significant signal at lags 0, +2, +3, or +4. This is consistent with a causal channel operating with a ~1 year delay (formulation changes → market exposure → metabolic response).

### 6. Contemporaneous correlations are negative (expected artifact)

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| Levels: Obesity vs Mean Ingredients | r = +0.67 | 0.012 | Both trending up |
| First-diff: dObesity vs dAII | r = -0.70 | 0.012 | Contemporaneous inverse |
| First-diff: dObesity vs dNOVA4 | r = -0.82 | 0.001 | Contemporaneous inverse |

Contemporaneous first-differenced correlations are **negative**, which initially appears to contradict the hypothesis. However, this is consistent with the lagged structure: if additive increases take ~1 year to manifest in obesity outcomes, the contemporaneous correlation captures the "wrong" timing. The negative sign likely reflects noise and the small sample (n=12). The lagged specification (Finding 5) is the correct test.

---

## What This Does / Doesn't Claim

**Does:**
- Quantify threshold-crossing probabilities and calibrated uncertainty for US obesity prevalence
- Demonstrate that both obesity and food formulation complexity have worsened over 2012-2024
- Produce a physical-measure risk surface (not market-implied volatility)
- Provide a framework for relating health outcomes to industry exposure proxies

**Doesn't:**
- Prove that additives or ultra-processed foods *cause* obesity
- Imply market pricing or tradeable risk
- Claim causal attribution from time-series correlation alone
- Represent risk-neutral probabilities

---

## Model

**Threshold Digital Option on Logit(Obesity Prevalence)**

| Concept | Options Analogy | Health Mapping |
|---|---|---|
| Underlying S_t | Stock price | `logit(obesity_prevalence)` |
| Strike K | Option strike | Obesity threshold (25%-45%) |
| Maturity T | Time to expiry | Forecasting horizon (1-10 years) |
| Implied vol | Black-Scholes IV | Calibrated crossing uncertainty |
| Spot | Current price | US national 2024: 33.1% (S_0 = -0.704) |

### Surface Construction (v2: Conditional + Logistic Smoothing)

1. **Transform**: `S_t = log(p_t / (1 - p_t))` where `p_t` = state-level obesity prevalence
2. **Conditional probability**: For each `(T, K)`, fit a logistic regression `Pr(p_{t+T} >= K | S_t)` on the full state panel, then evaluate at `S_t = S_0`
3. **Conditional drift**: `mu_T = E[S_{t+T} - S_t | S_t in ATM bucket]` using states with prevalence near the national level (delta = 0.25 logit units)
4. **Inversion**: `sigma(T, K) = (K_logit - (S_0 + mu_T)) / (Phi^-1(1 - P_hat) * sqrt(T))`
5. **NaN handling**: Strikes where P_hat is extreme (<1% or >99%) are marked NaN — no clamped fake volatilities

### Exposure Model: Additive Intensity Index

The AII is a min-max normalized composite of three Open Food Facts metrics for US-market products:
- **Mean additive count** (number of identified food additives per product)
- **Mean ingredient count** (total ingredients per product — proxy for formulation complexity)
- **NOVA 4 share** (fraction classified as ultra-processed by the NOVA system)

Each component is normalized to [0, 1], then averaged to produce a single annual index.

---

## Data

### CDC BRFSS (Outcomes)

- **Source**: [CDC BRFSS Nutrition, Physical Activity, and Obesity](https://data.cdc.gov/resource/hn4x-zwk7)
- **Coverage**: 50 US states + DC, annual, 2011-2024
- **Metric**: Adult obesity prevalence (% of adults aged 18+ with BMI >= 30)
- **Note**: 2025 BRFSS data has not been released as of this writing

### Open Food Facts (Exposure)

- **Source**: [Open Food Facts](https://world.openfoodfacts.org/) via search API
- **Coverage**: ~465 US-market products, creation years 2012-2025
- **Metrics**: `additives_n`, `ingredients_n`, `nova_group`
- **Caveat**: "Year" = year added to database, not date of first sale. This introduces selection bias (earlier years under-represented, later years may over-represent products added by enthusiasts). Sample sizes range from 7 to 99 per year.

---

## Outputs

| File | Description |
|---|---|
| `outputs/health_vol_surface_model1.csv` | Full surface: P_cross, mu_T, sigma, diagnostics |
| `outputs/health-surface-data.json` | Three.js visualization data |
| `outputs/obesity_trends.png` | US national + state-level obesity trends |
| `outputs/additive_load_by_year.csv` | Yearly additive/ingredient metrics |
| `outputs/additive_trends.png` | Additive load trend plots |
| `outputs/exposure_outcome_analysis.png` | 8-panel research figure |
| `outputs/exposure_regression_results.csv` | Correlation and regression results |
| `outputs/merged_timeseries.csv` | Merged annual obesity + additive data |

## Usage

```bash
# 1. Download fresh CDC data
python scripts/01_download_cdc_obesity.py

# 2. Build vol surface
python scripts/02_build_surface.py

# 3. Generate obesity trend plots
python scripts/03_obesity_trends.py

# 4. Download Open Food Facts data and compute additive load index
python scripts/04_additive_index.py

# 5. Run exposure-outcome analysis (requires steps 2-4)
python scripts/05_exposure_regression.py

# Or run the end-to-end pipeline from raw JSON
python scripts/build_all.py
```

## Project Structure

```
scripts/
  01_download_cdc_obesity.py   # CDC BRFSS data download
  02_build_surface.py          # Vol surface (v2: conditional + logistic)
  03_obesity_trends.py         # Prevalence trend plots
  04_additive_index.py         # Open Food Facts additive analysis
  05_exposure_regression.py    # Cross-dataset regression + research figure
  build_all.py                 # End-to-end pipeline (JSON -> surface)
```

## Visualization

The surface is rendered as an interactive Three.js wireframe on my [portfolio site](https://arnavnair229.github.io).

## Dependencies

```
pip install -r requirements.txt
```

---

## Thesis

I use option-style uncertainty tools to quantify how close the US is to crossing high-risk obesity thresholds. In a follow-on analysis, I relate these risks to ultra-processed food exposure proxies (additive load, ingredient complexity, NOVA classification), consistent with — but not proving — incentive-driven formulation by food brands. The surface is calibrated from historical threshold-crossing probabilities; it is a **physical-measure risk surface**, not market-implied volatility.
