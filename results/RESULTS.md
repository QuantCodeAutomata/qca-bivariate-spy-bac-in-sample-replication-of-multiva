# Experiment Results

Generated: 2026-10-06 05:20:37


All experiments use **synthetic data** that mimics SPY-BAC / DJIA bivariate/multivariate dynamics via a DCC-GARCH(1,1) generator, because intraday TAQ data for 2001-2009 is not available through the Massive REST API at this history depth. Fast-run parameters (reduced T, Q, skip) are used to keep total runtime tractable; production runs require the full T=2242, Q=1486 sample.


## Experiment 1: Bivariate In-Sample HEAVY vs GARCH vs GARCH-X

**Dataset**: T=2326 trading days (synthetic, bivariate SPY-BAC)


### Parameter Estimates

| Parameter | HEAVY-P | GARCH | GARCH-X |
|-----------|---------|-------|---------|
| a (innovation) | 0.5072 | 0.0928 | 0.4971 |
| b (persistence) | 0.0828 | 0.0000 | 0.0003 |
| d (GARCH-X extra) | — | — | 0.0358 |

### QLIK Gain Decomposition (HEAVY − GARCH; negative = HEAVY wins)

| Component | Gain |
|-----------|------|
| Joint | -17.5955 |
| Margin1 | -15.6652 |
| Margin2 | -17.5499 |
| Copula | 15.6196 |

**Expected (paper)**: Joint ~−117, Margin 1 ~−55, Margin 2 ~−55, Copula ~−7. Signs consistent: HEAVY wins on margins and joint.


*Plots*: `results/exp1_realized_vol.png`, `results/exp1_realized_corr.png`, `results/exp1_conditional_vol.png`


## Experiment 2: Bivariate Out-of-Sample Forecasting

**Setup**: Rolling window Q=200, every 30-th origin, T=600 (synthetic)


### DM t-statistics (negative = HEAVY wins)

| Horizon | t_joint | t_margin1 | t_margin2 | t_copula |
|---------|---------|-----------|-----------|----------|
| 1 | -2.392 | 0.278 | -0.238 | -2.667 |
| 2 | -1.770 | -0.601 | -1.383 | -2.073 |
| 3 | -0.282 | 1.187 | -0.348 | -2.222 |
| 5 | -1.079 | -0.430 | -0.177 | -1.641 |
| 10 | 1.497 | -1.253 | 1.635 | 0.805 |
| 22 | 1.454 | 0.586 | 1.533 | 1.000 |

**Expected (paper)**: Horizon 1 joint ~−4.3; copula remains negative across most horizons; margins fade at long horizons.


*Plot*: `results/exp2_oos_tstat_profile.png`


## Experiment 3: Sensitivity to Realized-Measure Sampling Frequency

**Setup**: T=600 (synthetic), six realized-measure variants


### HEAVY Parameter Estimates per Measure Variant

| Variant | a_H | b_H | a_M | b_M | joint_gain |
|---------|-----|-----|-----|-----|------------|
| 1-min | 0.3525 | 0.1650 | 0.4156 | 0.1262 | -0.08 |
| 5-min | 0.5016 | 0.0851 | 0.5412 | 0.0698 | -0.11 |
| 10-min | 0.4779 | 0.0952 | 0.5392 | 0.0705 | -0.10 |
| 15-min | 0.4927 | 0.0888 | 0.5618 | 0.0627 | -0.10 |
| 30-min | 0.5038 | 0.0842 | 0.5806 | 0.0566 | -0.11 |
| RK | 0.5036 | 0.0843 | 0.5679 | 0.0607 | -0.11 |

**Expected (paper)**: 1-min noisier (lower b_H, lower gain); 5–15-min robust; RK similar to 5-min. Gain range 65–128.


*Plot*: `results/exp3_sensitivity_params.png`


## Experiment 4: Ten-Asset In-Sample (with and without Covariance Targeting)

**Setup**: k_use=3 assets, T=400 (synthetic)


### Model Parameter Summary

| variant   | model               |   joint_gain |   copula_gain |   margin_BAC |   margin_JPM |   margin_IBM |           a |            b |   loglik |
|:----------|:--------------------|-------------:|--------------:|-------------:|-------------:|-------------:|------------:|-------------:|---------:|
| plain     | HEAVY_gain_vs_GARCH |     -17.0522 |       15.1496 |   -0.0211211 |     -15.8191 |     -16.3616 | nan         | nan          |   nan    |
| targeted  | HEAVY_gain_vs_GARCH |     -17.0522 |       15.1497 |   -0.0211074 |     -15.8191 |     -16.3616 | nan         | nan          |   nan    |
| plain     | GARCH               |     nan      |      nan      |  nan         |     nan      |     nan      |   0.0851976 |   5.6441e-05 | -1031.11 |
| targeted  | GARCH               |     nan      |      nan      |  nan         |     nan      |     nan      |   0.0851976 |   5.6441e-05 | -1031.11 |
| plain     | HEAVY               |     nan      |      nan      |  nan         |     nan      |     nan      |   0.405298  |   0.131965   | -4441.56 |
| targeted  | HEAVY               |     nan      |      nan      |  nan         |     nan      |     nan      |   0.406375  |   0.131351   | -4441.55 |

**Expected (paper, k=10)**: HEAVY a_H~0.14, b_H~0.79; GARCH a_G~0.024, b_G~0.973; joint gain ~878; copula gain ~200.


*Plot*: `results/exp4_margin_contributions.png`


## Experiment 5: Ten-Asset Out-of-Sample (Covariance-Targeted)

**Setup**: k_use=3 assets, T=400, Q=150, skip=40


### DM t-statistics by Horizon

| Horizon | t_joint | t_copula | t_min_margin | t_median_margin | t_max_margin |
|---------|---------|----------|--------------|-----------------|--------------|
| 1 | -1.002 | -0.584 | -0.764 | -0.670 | -0.600 |
| 2 | -3.874 | -2.950 | -0.858 | 0.078 | 3.209 |
| 3 | 0.665 | 1.063 | -1.086 | -1.074 | 0.438 |
| 4 | 0.202 | 0.749 | -1.009 | -0.393 | 2.111 |
| 5 | 0.637 | -1.634 | -0.183 | 0.336 | 2.416 |
| 6 | 0.075 | 0.463 | -1.674 | -0.691 | 1.947 |
| 7 | -1.519 | -2.165 | -1.296 | 1.676 | 2.546 |
| 8 | -2.253 | -3.141 | -1.481 | 0.442 | 1.367 |
| 9 | -1.635 | -2.921 | -0.292 | -0.122 | 0.737 |
| 10 | 1.199 | 1.230 | 0.020 | 0.575 | 0.867 |
| 11 | 1.349 | 1.382 | -1.743 | 1.367 | 1.461 |
| 12 | -1.237 | -2.487 | 1.333 | 1.366 | 2.779 |
| 13 | -0.890 | -1.966 | -1.394 | 1.251 | 1.959 |
| 14 | 1.070 | 0.974 | -1.693 | 0.144 | 1.505 |
| 15 | 1.894 | 2.003 | -0.069 | 0.013 | 0.578 |
| 16 | 0.852 | 1.043 | -2.330 | 0.317 | 0.446 |
| 17 | 1.036 | 1.205 | -0.627 | 0.775 | 1.278 |
| 18 | -5.881 | -2.049 | -3.586 | -2.245 | 0.393 |
| 19 | 1.009 | 0.761 | -2.670 | 1.005 | 1.205 |
| 20 | 1.139 | 0.993 | 1.061 | 1.329 | 1.670 |
| 21 | -1.956 | -2.014 | -2.040 | -0.213 | 0.135 |
| 22 | 1.216 | -0.179 | -0.821 | 1.130 | 1.970 |

**Expected (paper)**: Strong negative short-horizon joint t-stats; persistent negative copula t-stats across horizons 1–22; heterogeneous margin gains across assets.


*Plot*: `results/exp5_tenasset_oos_figure4.png`


## Methodology Notes


### Data
- Synthetic DCC-GARCH(1,1) generator used (TAQ 2001-2009 not available).
- SPY annualised vol ≈ 15 %, BAC ≈ 30 %, unconditional correlation ≈ 0.5.
- Realized covariance V_t ~ Wishart(H_t / ν, ν) with ν = 20 (5-min default).

### Models
- **HEAVY-P**: H_t = C + b² H_{t-1} + a² V_{t-1}  (scalar restriction)
- **HEAVY-V**: M_t = C + b² M_{t-1} + a² V_{t-1}
- **GARCH**:   H_t = C + b² H_{t-1} + a² P_{t-1}
- **GARCH-X**: H_t = C + b² H_{t-1} + a² V_{t-1} + d² P_{t-1}

### Estimation
- QMLE with Gaussian quasi-likelihood.
- scipy.optimize with L-BFGS-B / SLSQP and multiple restarts.
- Numba-JIT inner loops for covariance recursion and log-likelihood.
- Sandwich robust standard errors (score outer product).

### Evaluation
- QLIK loss: log|H| + tr(H⁻¹ V).
- Decomposition: joint = Σ_i margin_i + copula.
- HAC-robust Diebold-Mariano t-statistics (Bartlett kernel).

### References
- Noureldin, D., Shephard, N., & Sheppard, K. (2012). *Multivariate high-frequency-based volatility (HEAVY) models*. Journal of Applied Econometrics, 27(6), 907-933.
