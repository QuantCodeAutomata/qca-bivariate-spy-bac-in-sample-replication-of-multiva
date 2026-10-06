# AGENTS.md — Persistent Memory for OpenHands

## Project Purpose

Replicate Shephard & Sheppard (2010) multivariate HEAVY model results for a bivariate
SPY-BAC portfolio (2001-02-01 to 2009-12-31, ≈2242 trading days) and benchmark against
DCC-GARCH, BEKK-GARCH, and scalar-BEKK.

---

## Repository Structure

```
project/
├── data/
│   ├── __init__.py
│   ├── generate_synthetic.py   # Canonical synthetic data source
│   └── fetch_data.py           # Massive API wrapper (falls back to synthetic)
├── src/
│   ├── __init__.py
│   ├── models/                 # Model class definitions
│   │   └── __init__.py
│   ├── estimation/             # Optimisers, log-likelihood functions
│   │   └── __init__.py
│   ├── forecasting/            # Rolling / recursive forecast engines
│   │   └── __init__.py
│   └── utils/
│       ├── __init__.py
│       ├── matrix.py           # vech / ivech / duplication / elimination / mat_sqrt
│       └── linalg.py           # stable_log_det / is_pd / regularize
├── exp/                        # Runnable experiment scripts
├── tests/                      # pytest suite
├── results/                    # Figures & tables (binary outputs gitignored)
├── .gitignore
├── README.md
└── AGENTS.md                   # ← this file
```

---

## Data Infrastructure

### Why Synthetic Data

Historical tick-by-tick TAQ data for 2001–2009 is **not available** through the Massive
API (which covers only recent years).  All experiments therefore use the synthetic
generator in `data/generate_synthetic.py` by default.

`data/fetch_data.py` first checks for a `MASSIVE_TOKEN` environment variable and attempts
to pull daily OHLCV bars.  If the token is absent, or the returned series is shorter than
500 rows, it automatically falls back to `generate_synthetic`.

### Data Contract

Every data-providing function returns a `dict` with these keys:

| Key       | Shape       | Description                                   |
|-----------|-------------|-----------------------------------------------|
| `returns` | `(T, k)`    | Daily log-returns (decimal, not percent)       |
| `P`       | `(T, k, k)` | Outer-product matrices `r_t r_t'`              |
| `V`       | `(T, k, k)` | Realized covariance matrices (noisy proxy)     |
| `dates`   | `(T,)`      | `pd.DatetimeIndex` of business days            |

`k = 2` for bivariate (SPY, BAC); `k` varies for multi-asset variants.

---

## HEAVY Model Equations

The HEAVY model maintains **two** separate recursions:

### 1. Returns Equation (daily-frequency)

```
H_t = Ω_R + A_R V_{t-1} A_R' + B_R H_{t-1} B_R'
```

- `H_t`   : k×k conditional covariance matrix for day t returns
- `Ω_R`   : k×k intercept (PSD)
- `A_R`   : k×k coefficient on lagged realized covariance
- `B_R`   : k×k coefficient on lagged H
- `V_{t-1}`: realized covariance from previous day (high-frequency proxy)

### 2. Realized Measure Equation (realized-frequency)

```
M_t = Ω_M + A_M V_{t-1} A_M' + B_M M_{t-1} B_M'
```

- `M_t`   : k×k conditional expectation of today's realized covariance
- `Ω_M`   : k×k intercept (PSD)
- `A_M`   : k×k coefficient on lagged realized covariance
- `B_M`   : k×k coefficient on lagged M

### Scalar/Diagonal Restrictions

In practice (especially for k > 2) the matrices `A_*`, `B_*` are restricted to:
- **Scalar HEAVY**: `A = a I`, `B = b I` (2 scalars per equation)
- **Diagonal HEAVY**: `A = diag(a_1,...,a_k)`, `B = diag(b_1,...,b_k)`
- **Full BEKK-HEAVY**: unrestricted k×k (rarely used for k > 3)

### Log-Likelihood

For returns:
```
ℓ_R(θ) = -½ Σ_t [ k log(2π) + log|H_t| + r_t' H_t^{-1} r_t ]
```

For realized measure (Wishart):
```
ℓ_M(θ) = -½ Σ_t [ (ν+k+1) log|V_t| + ν log|M_t| + ν tr(M_t^{-1} V_t) ]
```

Combined (two-step or joint):
```
ℓ(θ) = ℓ_R(θ) + ℓ_M(θ)
```

where ν is the Wishart degrees-of-freedom parameter.

---

## Key Implementation Decisions

### Parameterisation

All three models (HEAVY, GARCH, GARCH-X) use the same **Cholesky intercept
parameterisation**:

    theta = (vech(C_L), b, a)               # HEAVY-H / HEAVY-M / GARCH
    theta = (vech(C_L), b, a, d)            # GARCH-X

where `C_L` is the **lower-triangular Cholesky factor** of the SPD intercept
`C = C_L @ C_L.T`, ensuring the intercept is always positive-semi-definite.
This eliminates the need for constrained optimisation on the intercept.

### `vech` convention

Lower-triangular, **column-by-column** (Magnus & Neudecker, i.e. column j first:
elements (j,j), (j+1,j), …, (k-1,j)).  The inverse (`lower_chol_from_vech`) fills a
`(k,k)` lower-triangular matrix using the same ordering.  Both functions live in
`src/utils/matrix.py`.

### Stationarity constraints

Each model enforces `b + a < 1` (scalar raw parameters) via SLSQP inequality
constraints.  The actual recursion coefficients are `b²` and `a²`, so the true
persistence is `φ = b² + a²`.

### Initial parameters

Starting values are calibrated so the implied unconditional mean equals the sample
covariance of the first `min(252, T//4)` observations:

    scale = 1 - b₀² - a₀²
    C₀  = scale × sample_cov
    C_L₀ = chol(C₀)

Defaults: HEAVY b₀=0.7, a₀=0.15;  GARCH b₀=0.9, a₀=0.06;  GARCH-X b₀=0.7, a₀=0.15, d₀=0.05.

### PD enforcement (Numba path)

Inside `@njit` kernels the PD projection is done by `ensure_pd_numba` (eigenvalue
clipping at `ε = 1e-8`), using `np.linalg.eigh + np.diag`.  On the Python side,
`ensure_pd` from `src/utils/linalg.py` is used.

### Numba JIT kernels

Two shared kernels in `src/models/heavy.py` are imported and reused by `garch.py`
and `garch_x.py`:

- `_cov_recursion(C, b2, a2, V_array, init)` — generic SPD recursion
- `_neg_loglik_kernel(X_seq, eval_array)` — QMLE objective

`garch_x.py` adds `_garchx_recursion(C, b2, a2, d2, V_array, P_array, init)`
for the two-driver case.

### Variation-free HEAVY estimation

`ScalarHEAVY.fit_H()` and `ScalarHEAVY.fit_M()` are estimated **independently**
(variation-free property of HEAVY).  `fit()` is just a convenience wrapper calling
both.

### Sandwich standard errors

Implemented numerically in each model's `sandwich_se` / `sandwich_se_H` method:
1. Compute per-observation score via **central finite differences** of the full
   recursion (perturbation ε = 1e-5, p perturbed trajectories).
2. Compute numerical Hessian via **second-order finite differences** (O(p²) recursion evaluations).
3. Apply the sandwich formula SE = √diag(A⁻¹ B A⁻¹).

### Multi-step forecasting

| Model   | Closed form                                                               |
|---------|---------------------------------------------------------------------------|
| GARCH   | `ω_G + φ_G^{s-1} (H₁ - ω_G)`,  φ_G = b²+a²,  ω_G = C_G/(1-φ_G)       |
| HEAVY-M | Propagate M via M persistence φ_M = b_M²+a_M²                            |
| HEAVY-H | Recursion: C_H + b_H² H_{s-1} + a_H² M_{s-1}                             |

### QLIK decomposition

`qlik_copula_loss(H, V)` = joint QLIK − Σᵢ marginal QLIK_i.

### DM test

In `src/forecasting/qlik.py` the `dm_test` function uses `hac_variance` (Newey-West).
In `src/estimation/inference.py` there is also a `dm_test` that delegates to the same
`hac_variance`.  Both are equivalent scalar DM tests.

### Module dependencies

```
src.utils.linalg       ← imports: numpy, numba, scipy.linalg
src.utils.matrix       ← imports: numpy
src.models.heavy       ← imports: linalg, matrix, scipy.optimize, numba
src.models.garch       ← imports: linalg, matrix, heavy (kernels reused)
src.models.garch_x     ← imports: linalg, matrix, heavy (kernels reused)
src.estimation.qmle    ← imports: heavy, garch, garch_x, matrix
src.estimation.inference ← imports: numpy, arch.covariance.kernel.Bartlett
src.forecasting.multistep ← imports: heavy._unpack, linalg
src.forecasting.qlik   ← imports: linalg, estimation.inference
```

### Important numerical notes

- `stable_log_det` and `safe_inv` in `linalg.py` are kept for backward
  compatibility (used by legacy code).  New code uses `safe_logdet` and `safe_inv`.
- SLSQP with `(None, None)` bounds on vech(C_L) elements works correctly in scipy
  (treated as unconstrained).
- `_neg_loglik_kernel` returns `1e18` if any H_t fails the PD check (slogdet sign ≤ 0).
- `logdet_numba` and `trace_prod_numba` in `linalg.py` are Numba-compatible versions
  for use inside `@njit` kernels.

---

## Calibration Parameters (Synthetic Generator)

Unconditional daily variance = ω / (1 − α − β).

| Parameter            | SPY    | BAC    |
|----------------------|--------|--------|
| GARCH ω (daily)      | 9.0e-7 | 7.0e-6 |
| GARCH α              | 0.07   | 0.10   |
| GARCH β              | 0.92   | 0.88   |
| α + β                | 0.99   | 0.98   |
| Unc. daily var       | 9.0e-5 | 3.5e-4 |
| Annualised vol (unc) | ~15 %  | ~30 %  |
| Mean correlation     | 0.50   |        |
| Corr AR coeff (φ)    | 0.97   |        |
| Fisher-z innov. std  | 0.02   |        |
| Wishart ν (noise)    | 20     |        |

---

## Testing Conventions

- All tests live in `tests/`; run with `pytest tests/`.
- No mocks for matrix utilities—test against known analytical results.
- Fixture data (small arrays) may be committed under `tests/fixtures/` (excluded from the
  `*.pkl` gitignore via the `!tests/fixtures/**` exception).

### Critical: P_t is rank-1 (PSD), not PD

`P[t] = r_t @ r_t.T` is a **rank-1 outer product** with exactly one non-zero eigenvalue.
It is **NOT positive-definite**. Tests must:
- Check PSD (all eigvals ≥ −ε) for P, not `is_positive_definite(P[t])`.
- Never pass P as the H argument to `qlik_loss_t` or `qlik_loss`; use V (PD) instead.
- `V` (realized covariance, built from many intraday increments) IS PD.

---

## Confirmed Model API (as of session 3)

### `ScalarHEAVY.fit(P, V)`
Returns self.  Attributes after fitting:
- `params_H_` — 1-D array `[vech(C_L_H), b_raw_H, a_raw_H]`, len = nv + 2
- `params_M_` — 1-D array `[vech(C_L_M), b_raw_M, a_raw_M]`, len = nv + 2
- `H_seq_`    — (T, k, k) fitted H matrices
- `M_seq_`    — (T, k, k) fitted M matrices
- `loglik_H_` — float log-likelihood for H equation
- `loglik_M_` — float log-likelihood for M equation

Actual coefficients: `b_H = params_H_[nv]**2`, `a_H = params_H_[nv+1]**2`  
where `nv = k*(k+1)//2`.

### `ScalarGARCH.fit(P)`
Returns self.  Attributes:
- `params_`  — 1-D array `[vech(C_L), b_raw, a_raw]`, len = nv + 2
- `H_seq_`   — (T, k, k) fitted H matrices
- `loglik_`  — float

Actual coefficients: `b = params_[nv]**2`, `a = params_[nv+1]**2`.

### `ScalarGARCHX.fit(P, V)`
Returns self.  Attributes:
- `params_`  — 1-D array `[vech(C_L), b_raw, a_raw, d_raw]`, len = nv + 3
- `H_seq_`   — (T, k, k) fitted H matrices
- `loglik_`  — float

Actual coefficients: `b = params_[nv]**2`, `a = params_[nv+1]**2`, `d = params_[nv+2]**2`.

### `heavy_multistep_forecast(params_H, params_M, H1, M1, s_max)` → `(s_max, k, k)`
- `params_H` / `params_M` — raw param arrays from `heavy.params_H_` / `heavy.params_M_`
- `H1`, `M1` — (k,k) one-step-ahead covariance matrices
- Returns array of shape `(s_max, k, k)`: index `[s-1]` gives the s-step forecast

### `garch_multistep_forecast(params, H1, s_max)` → `(s_max, k, k)`
- `params` — raw param array from `garch.params_`
- `H1` — (k,k) one-step-ahead covariance
- Returns array of shape `(s_max, k, k)`: index `[s-1]` gives the s-step forecast

### `qlik_loss_t(H_t, P_t)` → float
Single-observation QLIK.  H_t must be PD.

### `qlik_loss(H_seq, P_seq)` → (T,) array
### `qlik_decomposed(H_seq, P_seq)` → dict
Keys: `'joint'` (T,), `'margins'` (T, k), `'copula'` (T,).

### `dm_test(d)` → float
Located in `src.estimation.inference`.  Also callable from `src.forecasting.qlik`.

---

## Experiment Script Conventions

All exp scripts live in `exp/`.

| Script                  | Key params                         | Notes                                  |
|-------------------------|------------------------------------|----------------------------------------|
| `exp1_bivariate_insample` | `T=None`                         | T=200 for fast test                    |
| `exp2_bivariate_oos`      | `T, Q, skip`                     | Q=100, skip=20 for fast smoke-test     |
| `exp3_sensitivity`        | `T=None`                         | T=200 for fast test                    |
| `exp4_tenasset_insample`  | `T=None, k_use=None`             | **k_use=3** for fast smoke-test (k=10 has 57-param optimisation, very slow) |
| `exp5_tenasset_oos`       | `T, Q, skip, horizons, k_use`    | **k_use=3** for fast smoke-test        |

All functions return a `dict` of results and are importable without side effects.

### One-step helper pattern (used in exp2, exp5)

To compute `H_{T+1|T}` from a fitted model without calling private methods:

```python
def _reconstruct_C(params, k):
    nv = k * (k + 1) // 2
    v = params[:nv]
    L = np.zeros((k, k))
    idx = 0
    for i in range(k):
        for j in range(i + 1):
            L[i, j] = float(v[idx]); idx += 1
    return L @ L.T

# HEAVY
nv = k*(k+1)//2
C_H = _reconstruct_C(heavy.params_H_, k)
b_H = float(heavy.params_H_[nv])**2; a_H = float(heavy.params_H_[nv+1])**2
H1  = C_H + b_H * heavy.H_seq_[-1] + a_H * V_last
H1  = 0.5*(H1 + H1.T) + np.eye(k)*1e-10

# GARCH
C_G = _reconstruct_C(garch.params_, k)
b_G = float(garch.params_[nv])**2; a_G = float(garch.params_[nv+1])**2
H1  = C_G + b_G * garch.H_seq_[-1] + a_G * P_last
H1  = 0.5*(H1 + H1.T) + np.eye(k)*1e-10
```
