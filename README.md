# Multivariate HEAVY vs GARCH: Bivariate SPY-BAC Replication

A replication study comparing **HEAVY** (High-frEquency-based Volatility) models against
classical **GARCH** models for covariance forecasting on a bivariate SPY-BAC equity pair
spanning the 2001–2009 sample period, covering the 2007–2009 financial crisis.

## Background

Shephard & Sheppard (2010) introduced the HEAVY model, which augments daily return-based
variance equations with realized measures computed from intraday data.  The key insight is
that realized covariance matrices—constructed from high-frequency (tick-by-tick) trades and
quotes—carry far more information about latent volatility than squared daily returns alone.

This project replicates and extends their bivariate results (SPY + BAC) and benchmarks
HEAVY against:

- **DCC-GARCH** – Engle (2002) Dynamic Conditional Correlation
- **BEKK-GARCH** – Engle & Kroner (1995) full-matrix GARCH
- **Scalar BEKK** – restricted scalar version

## Repository Layout

```
project/
├── data/
│   ├── generate_synthetic.py   # DCC-GARCH synthetic data generator (default)
│   └── fetch_data.py           # Massive API fetch (falls back to synthetic)
├── src/
│   ├── models/                 # HEAVY and GARCH model definitions
│   ├── estimation/             # MLE / quasi-MLE optimisation routines
│   ├── forecasting/            # Rolling-window and recursive forecasters
│   └── utils/
│       ├── matrix.py           # vech, ivech, duplication/elimination matrices, mat_sqrt
│       └── linalg.py           # stable_log_det, is_positive_definite, regularize_matrix
├── exp/                        # Experiment scripts (estimation, comparison tables)
├── tests/                      # Unit and integration tests
└── results/                    # Saved figures and tables (gitignored binaries)
```

## Data

Historical TAQ intraday data for 2001–2009 is not available via the Massive API for this
period, so **synthetic data** is used by default.  The synthetic generator (`data/generate_synthetic.py`)
produces a bivariate DCC-GARCH(1,1) process calibrated to empirical SPY-BAC moments:

| Asset | Annualised Vol | Notes              |
|-------|----------------|--------------------|
| SPY   | ~15 %          | S&P 500 ETF        |
| BAC   | ~30 %          | Bank of America    |

Average SPY-BAC correlation ≈ 0.50 with time-varying dynamics.

## Quick Start

```bash
# Install dependencies
pip install numpy scipy pandas

# Verify synthetic data generation
python -c "from data.generate_synthetic import generate_bivariate_data; d = generate_bivariate_data(100); print(d['returns'].shape)"

# Run data pipeline (falls back to synthetic automatically)
python data/fetch_data.py
```

## Key References

- Shephard, N. & Sheppard, K. (2010). *Realising the future: forecasting with
  high-frequency-based volatility (HEAVY) measures.* Journal of Applied
  Econometrics, 25(2), 197–231.
- Engle, R. F. (2002). *Dynamic Conditional Correlation.* Journal of Business &
  Economic Statistics, 20(3), 339–350.
- Engle, R. F. & Kroner, K. F. (1995). *Multivariate simultaneous generalized ARCH.*
  Econometric Theory, 11(1), 122–150.
