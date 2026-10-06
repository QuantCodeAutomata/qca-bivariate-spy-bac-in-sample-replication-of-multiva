"""
Master experiment runner.

Runs all five experiments sequentially using reduced dataset sizes for speed,
then produces results/RESULTS.md summarising every output.

Production parameters match the paper (T=2242, Q=1486).
Fast-run parameters (used here) keep total runtime under ~10 minutes.
"""

import os
import sys
import json
import traceback
from datetime import datetime

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Fast-run parameters
# ---------------------------------------------------------------------------
T_BIV    = 600   # bivariate dataset length
Q_BIV    = 200   # rolling window (bivariate)
SKIP_BIV = 30    # evaluate every SKIP-th origin
T_TEN    = 400   # ten-asset dataset length
Q_TEN    = 150   # rolling window (ten-asset)
SKIP_TEN = 40    # evaluate every SKIP-th origin
K_USE    = 3     # reduce to 3 assets for ten-asset experiments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_run(name, fn, **kwargs):
    """Run fn(**kwargs) and return its result dict; on exception return {}."""
    print(f"\n{'='*70}")
    print(f"  Running {name} ...")
    print(f"{'='*70}")
    try:
        result = fn(**kwargs)
        print(f"  {name} completed successfully.")
        return result or {}
    except Exception:
        print(f"  ERROR in {name}:")
        traceback.print_exc()
        return {}


def _fmt(v):
    """Format a scalar value for Markdown."""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Experiment 1: Bivariate in-sample
# ---------------------------------------------------------------------------

def run_exp1():
    from exp.exp1_bivariate_insample import run_experiment
    return run_experiment()   # uses get_bivariate_data() internally


# ---------------------------------------------------------------------------
# Experiment 2: Bivariate OOS
# ---------------------------------------------------------------------------

def run_exp2():
    from exp.exp2_bivariate_oos import run_experiment
    return run_experiment(T=T_BIV, Q=Q_BIV, skip=SKIP_BIV)


# ---------------------------------------------------------------------------
# Experiment 3: Sensitivity
# ---------------------------------------------------------------------------

def run_exp3():
    from exp.exp3_sensitivity import run_experiment
    return run_experiment(T=T_BIV)


# ---------------------------------------------------------------------------
# Experiment 4: Ten-asset in-sample
# ---------------------------------------------------------------------------

def run_exp4():
    from exp.exp4_tenasset_insample import run_experiment
    return run_experiment(T=T_TEN, k_use=K_USE)


# ---------------------------------------------------------------------------
# Experiment 5: Ten-asset OOS
# ---------------------------------------------------------------------------

def run_exp5():
    from exp.exp5_tenasset_oos import run_experiment
    return run_experiment(T=T_TEN, Q=Q_TEN, skip=SKIP_TEN, k_use=K_USE)


# ---------------------------------------------------------------------------
# Generate RESULTS.md
# ---------------------------------------------------------------------------

def write_results_md(r1, r2, r3, r4, r5):
    """Collect results from all experiments and write results/RESULTS.md."""

    lines = []

    lines.append("# Experiment Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("")
    lines.append("All experiments use **synthetic data** that mimics SPY-BAC / DJIA "
                 "bivariate/multivariate dynamics via a DCC-GARCH(1,1) generator, "
                 "because intraday TAQ data for 2001-2009 is not available through "
                 "the Massive REST API at this history depth. "
                 "Fast-run parameters (reduced T, Q, skip) are used to keep total "
                 "runtime tractable; production runs require the full T=2242, Q=1486 "
                 "sample.\n")
    lines.append("")

    # ---- Experiment 1 -------------------------------------------------------
    lines.append("## Experiment 1: Bivariate In-Sample HEAVY vs GARCH vs GARCH-X\n")
    lines.append(f"**Dataset**: T={r1.get('T', 'N/A')} trading days (synthetic, bivariate SPY-BAC)\n")
    lines.append("")
    lines.append("### Parameter Estimates\n")
    lines.append("| Parameter | HEAVY-P | GARCH | GARCH-X |")
    lines.append("|-----------|---------|-------|---------|")
    lines.append(f"| a (innovation) | {_fmt(r1.get('a_H','N/A'))} | "
                 f"{_fmt(r1.get('a_G','N/A'))} | {_fmt(r1.get('a_GX','N/A'))} |")
    lines.append(f"| b (persistence) | {_fmt(r1.get('b_H','N/A'))} | "
                 f"{_fmt(r1.get('b_G','N/A'))} | {_fmt(r1.get('b_GX','N/A'))} |")
    lines.append(f"| d (GARCH-X extra) | — | — | {_fmt(r1.get('d_GX','N/A'))} |")
    lines.append("")
    lines.append("### QLIK Gain Decomposition (HEAVY − GARCH; negative = HEAVY wins)\n")
    lines.append("| Component | Gain |")
    lines.append("|-----------|------|")
    for comp in ('joint', 'margin1', 'margin2', 'copula'):
        key = f"{comp}_gain_HvsG"
        lines.append(f"| {comp.capitalize()} | {_fmt(r1.get(key, 'N/A'))} |")
    lines.append("")
    lines.append("**Expected (paper)**: Joint ~−117, Margin 1 ~−55, Margin 2 ~−55, "
                 "Copula ~−7. Signs consistent: HEAVY wins on margins and joint.\n")
    lines.append("")
    lines.append("*Plots*: `results/exp1_realized_vol.png`, "
                 "`results/exp1_realized_corr.png`, `results/exp1_conditional_vol.png`\n")
    lines.append("")

    # ---- Experiment 2 -------------------------------------------------------
    lines.append("## Experiment 2: Bivariate Out-of-Sample Forecasting\n")
    lines.append(f"**Setup**: Rolling window Q={Q_BIV}, every {SKIP_BIV}-th origin, T={T_BIV} (synthetic)\n")
    lines.append("")
    df2 = r2.get('results_df', pd.DataFrame())
    if not df2.empty:
        lines.append("### DM t-statistics (negative = HEAVY wins)\n")
        lines.append("| Horizon | t_joint | t_margin1 | t_margin2 | t_copula |")
        lines.append("|---------|---------|-----------|-----------|----------|")
        for _, row in df2.iterrows():
            lines.append(f"| {int(row['horizon'])} | {row.get('t_joint',float('nan')):.3f} | "
                         f"{row.get('t_margin1',float('nan')):.3f} | "
                         f"{row.get('t_margin2',float('nan')):.3f} | "
                         f"{row.get('t_copula',float('nan')):.3f} |")
        lines.append("")
    else:
        lines.append("*Experiment 2 did not produce results (check log).*\n")
    lines.append("**Expected (paper)**: Horizon 1 joint ~−4.3; copula remains negative "
                 "across most horizons; margins fade at long horizons.\n")
    lines.append("")
    lines.append("*Plot*: `results/exp2_oos_tstat_profile.png`\n")
    lines.append("")

    # ---- Experiment 3 -------------------------------------------------------
    lines.append("## Experiment 3: Sensitivity to Realized-Measure Sampling Frequency\n")
    lines.append(f"**Setup**: T={T_BIV} (synthetic), six realized-measure variants\n")
    lines.append("")
    df3 = r3.get('results_df', pd.DataFrame())
    if not df3.empty:
        lines.append("### HEAVY Parameter Estimates per Measure Variant\n")
        lines.append("| Variant | a_H | b_H | a_M | b_M | joint_gain |")
        lines.append("|---------|-----|-----|-----|-----|------------|")
        for _, row in df3.iterrows():
            variant = row.get('freq_label', row.get('variant', '?'))
            jg_raw = row.get('joint_gain_vs_garch', row.get('joint_gain', float('nan')))
            try:
                jg = f"{float(jg_raw):.2f}"
            except (ValueError, TypeError):
                jg = str(jg_raw)
            lines.append(f"| {variant} | "
                         f"{row.get('a_H',float('nan')):.4f} | "
                         f"{row.get('b_H',float('nan')):.4f} | "
                         f"{row.get('a_M',float('nan')):.4f} | "
                         f"{row.get('b_M',float('nan')):.4f} | "
                         f"{jg} |")
        lines.append("")
    else:
        lines.append("*Experiment 3 did not produce results (check log).*\n")
    lines.append("**Expected (paper)**: 1-min noisier (lower b_H, lower gain); "
                 "5–15-min robust; RK similar to 5-min. Gain range 65–128.\n")
    lines.append("")
    lines.append("*Plot*: `results/exp3_sensitivity_params.png`\n")
    lines.append("")

    # ---- Experiment 4 -------------------------------------------------------
    lines.append("## Experiment 4: Ten-Asset In-Sample (with and without Covariance Targeting)\n")
    lines.append(f"**Setup**: k_use={K_USE} assets, T={T_TEN} (synthetic)\n")
    lines.append("")
    df4 = r4.get('results_df', pd.DataFrame())
    if not df4.empty:
        lines.append("### Model Parameter Summary\n")
        lines.append(df4.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("*Experiment 4 did not produce results (check log).*\n")
    lines.append("**Expected (paper, k=10)**: HEAVY a_H~0.14, b_H~0.79; "
                 "GARCH a_G~0.024, b_G~0.973; joint gain ~878; copula gain ~200.\n")
    lines.append("")
    lines.append("*Plot*: `results/exp4_margin_contributions.png`\n")
    lines.append("")

    # ---- Experiment 5 -------------------------------------------------------
    lines.append("## Experiment 5: Ten-Asset Out-of-Sample (Covariance-Targeted)\n")
    lines.append(f"**Setup**: k_use={K_USE} assets, T={T_TEN}, Q={Q_TEN}, skip={SKIP_TEN}\n")
    lines.append("")
    df5 = r5.get('results_df', pd.DataFrame())
    if not df5.empty:
        lines.append("### DM t-statistics by Horizon\n")
        lines.append("| Horizon | t_joint | t_copula | t_min_margin | t_median_margin | t_max_margin |")
        lines.append("|---------|---------|----------|--------------|-----------------|--------------|")
        for _, row in df5.iterrows():
            lines.append(
                f"| {int(row['horizon'])} | "
                f"{row.get('t_joint',float('nan')):.3f} | "
                f"{row.get('t_copula',float('nan')):.3f} | "
                f"{row.get('t_min_margin',float('nan')):.3f} | "
                f"{row.get('t_median_margin',float('nan')):.3f} | "
                f"{row.get('t_max_margin',float('nan')):.3f} |"
            )
        lines.append("")
    else:
        lines.append("*Experiment 5 did not produce results (check log).*\n")
    lines.append("**Expected (paper)**: Strong negative short-horizon joint t-stats; "
                 "persistent negative copula t-stats across horizons 1–22; "
                 "heterogeneous margin gains across assets.\n")
    lines.append("")
    lines.append("*Plot*: `results/exp5_tenasset_oos_figure4.png`\n")
    lines.append("")

    # ---- Methodology --------------------------------------------------------
    lines.append("## Methodology Notes\n")
    lines.append("")
    lines.append("### Data")
    lines.append("- Synthetic DCC-GARCH(1,1) generator used (TAQ 2001-2009 not available).")
    lines.append("- SPY annualised vol ≈ 15 %, BAC ≈ 30 %, unconditional correlation ≈ 0.5.")
    lines.append("- Realized covariance V_t ~ Wishart(H_t / ν, ν) with ν = 20 (5-min default).")
    lines.append("")
    lines.append("### Models")
    lines.append("- **HEAVY-P**: H_t = C + b² H_{t-1} + a² V_{t-1}  (scalar restriction)")
    lines.append("- **HEAVY-V**: M_t = C + b² M_{t-1} + a² V_{t-1}")
    lines.append("- **GARCH**:   H_t = C + b² H_{t-1} + a² P_{t-1}")
    lines.append("- **GARCH-X**: H_t = C + b² H_{t-1} + a² V_{t-1} + d² P_{t-1}")
    lines.append("")
    lines.append("### Estimation")
    lines.append("- QMLE with Gaussian quasi-likelihood.")
    lines.append("- scipy.optimize with L-BFGS-B / SLSQP and multiple restarts.")
    lines.append("- Numba-JIT inner loops for covariance recursion and log-likelihood.")
    lines.append("- Sandwich robust standard errors (score outer product).")
    lines.append("")
    lines.append("### Evaluation")
    lines.append("- QLIK loss: log|H| + tr(H⁻¹ V).")
    lines.append("- Decomposition: joint = Σ_i margin_i + copula.")
    lines.append("- HAC-robust Diebold-Mariano t-statistics (Bartlett kernel).")
    lines.append("")
    lines.append("### References")
    lines.append("- Noureldin, D., Shephard, N., & Sheppard, K. (2012). "
                 "*Multivariate high-frequency-based volatility (HEAVY) models*. "
                 "Journal of Applied Econometrics, 27(6), 907-933.")
    lines.append("")

    md_path = os.path.join(RESULTS_DIR, 'RESULTS.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  RESULTS.md written to {md_path}")
    return md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Running all experiments ...")
    print(f"Fast-run params: T_biv={T_BIV}, Q_biv={Q_BIV}, skip_biv={SKIP_BIV}, "
          f"T_ten={T_TEN}, Q_ten={Q_TEN}, skip_ten={SKIP_TEN}, k_use={K_USE}")

    r1 = _safe_run("Experiment 1 (Bivariate In-Sample)",    run_exp1)
    r2 = _safe_run("Experiment 2 (Bivariate OOS)",          run_exp2)
    r3 = _safe_run("Experiment 3 (Sensitivity)",            run_exp3)
    r4 = _safe_run("Experiment 4 (Ten-Asset In-Sample)",    run_exp4)
    r5 = _safe_run("Experiment 5 (Ten-Asset OOS)",          run_exp5)

    md_path = write_results_md(r1, r2, r3, r4, r5)
    print(f"\nAll done. RESULTS.md at {md_path}")
