# Obesity Volatility Surface

A quantitative framework that models US obesity prevalence as an option-like underlying and constructs an implied uncertainty surface from CDC data.

## Model

**Threshold Digital Option on Logit(Obesity Prevalence)**

| Concept | Options Analogy | Health Mapping |
|---|---|---|
| Underlying S_t | Stock price | `logit(obesity_prevalence)` |
| Strike K | Option strike | Obesity threshold (25%-45%) |
| Maturity T | Time to expiry | Forecasting horizon (1-10 years) |
| Implied vol | Black-Scholes IV | Calibrated crossing uncertainty |

### Surface Construction

1. **Transform**: `S_t = log(p_t / (1 - p_t))` where `p_t` = state-level obesity prevalence
2. **Crossing probability**: `P_hat(T, K) = Pr(p_{t+T} >= K)` estimated empirically from the state panel
3. **Drift**: `mu_T = E[S_{t+T} - S_t]` from historical forward differences
4. **Inversion**: Assuming `S_{t+T} ~ N(S_0 + mu_T, sigma^2 * T)`, solve for sigma:

```
sigma(T, K) = |K_logit - (S_0 + mu_T)| / (Phi^-1(1 - P_hat) * sqrt(T))
```

This produces a 2D surface: **(Maturity T, Strike K) -> Implied Sigma**.

## Data

- **Source**: [CDC BRFSS Nutrition, Physical Activity, and Obesity](https://data.cdc.gov/resource/hn4x-zwk7)
- **Coverage**: 50 US states + DC, annual, 2011-2024
- **Metric**: Adult obesity prevalence (% of adults aged 18+ with BMI >= 30)

### Current Spot (2024)

| Metric | Value |
|---|---|
| US national obesity | 33.1% |
| Logit(spot) | -0.7037 |

## Surface Output (6 x 9 grid)

Maturities: 1, 2, 3, 5, 7, 10 years
Strikes: 25%, 28%, 30%, 32%, 35%, 38%, 40%, 42%, 45%

The spike near K=32% (close to the current spot of 33.1%) mirrors the ATM volatility behavior in equity options -- maximum uncertainty about whether the threshold will be crossed.

## Usage

### Download fresh data
```bash
python scripts/01_download_cdc_obesity.py
```

### Build surface (requires `data/raw/cdc_brfss_raw.json`)
```bash
python scripts/build_all.py
```

### Outputs
- `outputs/health_vol_surface_model1.csv` -- full surface with P_hat, mu_T, sigma
- `outputs/health-surface-data.json` -- JSON for Three.js visualization

## Visualization

The surface is rendered as an interactive Three.js wireframe on my [portfolio site](https://arnavnair229.github.io). The background animation shows the real (T, K) -> sigma surface with bilinear interpolation onto a dense render mesh.

## Dependencies

```
pip install -r requirements.txt
```

## Why This Exists

This project applies quantitative finance techniques to public health data to quantify and visualize health risk trajectories. The framework demonstrates how options pricing concepts (strike, maturity, implied vol) can provide insight into non-financial domains -- specifically, the increasing uncertainty around US obesity thresholds.
