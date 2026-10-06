"""
Experiment 1: Bivariate SPY-BAC in-sample comparison of HEAVY, GARCH, and GARCH-X.

Steps:
  a) Load bivariate data.
  b) Estimate ScalarHEAVY, ScalarGARCH, ScalarGARCHX.
  c) Print parameter table.
  d) Compute and decompose in-sample QLIK losses.
  e) Save results CSV.
  f) Generate realized-vol, realized-corr, and conditional-vol plots.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.fetch_data import get_bivariate_data
from data.generate_synthetic import generate_bivariate_data
from src.estimation.qmle import estimate_heavy, estimate_garch, estimate_garch_x
from src.forecasting.qlik import qlik_decomposed

RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annualised_vol(V_ii, scale=252):
    """Annualised percentage volatility from daily variance series."""
    return 100.0 * np.sqrt(scale * V_ii)


def _extract_ab(params, k):
    """Return (a², b²) scalar coefficients from packed [vech(C_L), b_raw, a_raw]."""
    nv = k * (k + 1) // 2
    b_raw = float(params[nv])
    a_raw = float(params[nv + 1])
    return a_raw ** 2, b_raw ** 2   # (ARCH coeff, GARCH/persistence coeff)


def _extract_abgx(params, k):
    """Return (a², b², d²) for GARCH-X from packed [vech(C_L), b_raw, a_raw, d_raw]."""
    nv = k * (k + 1) // 2
    b_raw = float(params[nv])
    a_raw = float(params[nv + 1])
    d_raw = float(params[nv + 2])
    return a_raw ** 2, b_raw ** 2, d_raw ** 2


def _print_param_table(heavy, garch, garch_x, k):
    a_H, b_H = _extract_ab(heavy.params_H_, k)
    a_M, b_M = _extract_ab(heavy.params_M_, k)
    a_G, b_G = _extract_ab(garch.params_, k)
    a_GX, b_GX, d_GX = _extract_abgx(garch_x.params_, k)

    header = f"{'Model':<12} {'a_H/a_G':>10} {'b_H/b_G':>10} {'a_M':>10} {'b_M':>10} {'a_GX':>8} {'b_GX':>8} {'d_GX':>8}"
    print("\n" + "=" * len(header))
    print("Parameter Estimates")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    print(f"{'HEAVY':<12} {a_H:>10.4f} {b_H:>10.4f} {a_M:>10.4f} {b_M:>10.4f} {'—':>8} {'—':>8} {'—':>8}")
    print(f"{'GARCH':<12} {a_G:>10.4f} {b_G:>10.4f} {'—':>10} {'—':>10} {'—':>8} {'—':>8} {'—':>8}")
    print(f"{'GARCH-X':<12} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {a_GX:>8.4f} {b_GX:>8.4f} {d_GX:>8.4f}")
    print("=" * len(header))


def _print_loglik_table(ll_heavy, ll_garch, ll_garch_x):
    gain_HG = ll_heavy - ll_garch
    gain_HGX = ll_heavy - ll_garch_x
    print("\n" + "=" * 50)
    print("In-sample Joint Log-likelihoods")
    print("=" * 50)
    print(f"  HEAVY   : {ll_heavy:>12.2f}")
    print(f"  GARCH   : {ll_garch:>12.2f}")
    print(f"  GARCH-X : {ll_garch_x:>12.2f}")
    print(f"  HEAVY gain vs GARCH   : {gain_HG:>+12.2f}  ({'HEAVY wins' if gain_HG > 0 else 'GARCH wins'})")
    print(f"  HEAVY gain vs GARCH-X : {gain_HGX:>+12.2f}  ({'HEAVY wins' if gain_HGX > 0 else 'GARCH-X wins'})")
    print("=" * 50)


def _decomposed_gains(dec_heavy, dec_garch, dates):
    """Compute mean QLIK gain (HEAVY - GARCH) decomposed into margins and copula."""
    joint_gain = np.mean(dec_heavy['joint']) - np.mean(dec_garch['joint'])
    m1_gain = np.mean(dec_heavy['margins'][:, 0]) - np.mean(dec_garch['margins'][:, 0])
    m2_gain = np.mean(dec_heavy['margins'][:, 1]) - np.mean(dec_garch['margins'][:, 1])
    cop_gain = np.mean(dec_heavy['copula']) - np.mean(dec_garch['copula'])

    print("\n" + "=" * 55)
    print("QLIK Gain Decomposition (HEAVY - GARCH; negative = HEAVY wins)")
    print("=" * 55)
    print(f"  Joint          : {joint_gain:>+10.4f}")
    print(f"  Margin 1 (SPY) : {m1_gain:>+10.4f}")
    print(f"  Margin 2 (BAC) : {m2_gain:>+10.4f}")
    print(f"  Copula         : {cop_gain:>+10.4f}")
    print(f"  Sum check      : {m1_gain + m2_gain + cop_gain:>+10.4f}  (should = joint)")
    print("=" * 55)
    return joint_gain, m1_gain, m2_gain, cop_gain


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_realized_vol(V, dates, path):
    """Plot 1: annualised realized volatility for SPY and BAC."""
    rv_spy = _annualised_vol(V[:, 0, 0])
    rv_bac = _annualised_vol(V[:, 1, 1])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, rv_spy, label='SPY', color='steelblue', linewidth=0.8)
    ax.plot(dates, rv_bac, label='BAC', color='firebrick', linewidth=0.8, alpha=0.8)
    ax.set_title('Annualised Realized Volatility (100·√(252·V_{ii,t}))')
    ax.set_ylabel('Vol (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_realized_corr(V, dates, path):
    """Plot 2: realized correlation rho_t = V_12,t / sqrt(V_11,t * V_22,t)."""
    rho = V[:, 0, 1] / np.sqrt(V[:, 0, 0] * V[:, 1, 1])
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(dates, rho, color='darkorange', linewidth=0.8)
    ax.axhline(np.mean(rho), color='black', linestyle='--', linewidth=0.8, label=f'Mean={np.mean(rho):.3f}')
    ax.set_ylim(-1, 1)
    ax.set_title('Realized Correlation ρ_t = V_{12,t} / √(V_{11,t}·V_{22,t})')
    ax.set_ylabel('Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_conditional_vol(H_heavy, H_garch, dates, path):
    """Plot 3: fitted conditional standard deviations (HEAVY vs GARCH)."""
    cv_heavy_spy = _annualised_vol(H_heavy[:, 0, 0])
    cv_heavy_bac = _annualised_vol(H_heavy[:, 1, 1])
    cv_garch_spy = _annualised_vol(H_garch[:, 0, 0])
    cv_garch_bac = _annualised_vol(H_garch[:, 1, 1])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(dates, cv_heavy_spy, label='HEAVY', color='steelblue', linewidth=0.9)
    axes[0].plot(dates, cv_garch_spy, label='GARCH', color='grey', linewidth=0.9, linestyle='--')
    axes[0].set_ylabel('Vol (%)')
    axes[0].set_title('SPY Conditional Volatility')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dates, cv_heavy_bac, label='HEAVY', color='firebrick', linewidth=0.9)
    axes[1].plot(dates, cv_garch_bac, label='GARCH', color='grey', linewidth=0.9, linestyle='--')
    axes[1].set_ylabel('Vol (%)')
    axes[1].set_title('BAC Conditional Volatility')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_experiment(T=None):
    """
    Run Experiment 1: bivariate in-sample HEAVY vs GARCH vs GARCH-X.

    Parameters
    ----------
    T : number of observations (None = use fetch_data default)

    Returns
    -------
    dict of key results
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Bivariate In-Sample HEAVY vs GARCH vs GARCH-X")
    print("=" * 60)

    # ---- a) Load data ----
    if T is not None:
        data = generate_bivariate_data(n_days=T, seed=42)
        print(f"Note: Using synthetic data with T={T}")
    else:
        data = get_bivariate_data()
    returns = data['returns']
    P = data['P']
    V = data['V']
    dates = data['dates']
    T_actual = len(returns)

    # ---- b) Log data info ----
    print(f"\nData shape : returns={returns.shape}, P={P.shape}, V={V.shape}")
    print(f"Date range : {dates[0].date()} to {dates[-1].date()} ({T_actual} obs)")

    # ---- c-e) Estimate models ----
    k = P.shape[1]

    print("\nEstimating ScalarGARCH ...")
    garch = estimate_garch(P)
    a_G, b_G = _extract_ab(garch.params_, k)
    print(f"  Done. a_G={a_G:.4f}, b_G={b_G:.4f}, loglik={garch.loglik_:.2f}")

    print("Estimating ScalarHEAVY ...")
    heavy = estimate_heavy(P, V)
    a_H, b_H = _extract_ab(heavy.params_H_, k)
    a_M, b_M = _extract_ab(heavy.params_M_, k)
    print(f"  Done. a_H={a_H:.4f}, b_H={b_H:.4f}, "
          f"a_M={a_M:.4f}, b_M={b_M:.4f}, loglik={heavy.loglik_H_:.2f}")

    print("Estimating ScalarGARCH-X ...")
    garch_x = estimate_garch_x(P, V)
    a_GX, b_GX, d_GX = _extract_abgx(garch_x.params_, k)
    print(f"  Done. a_GX={a_GX:.4f}, b_GX={b_GX:.4f}, "
          f"d_GX={d_GX:.4f}, loglik={garch_x.loglik_:.2f}")

    # ---- f) Parameter table ----
    _print_param_table(heavy, garch, garch_x, k)

    # ---- g) Log-likelihood table ----
    _print_loglik_table(heavy.loglik_H_, garch.loglik_, garch_x.loglik_)

    # ---- h) QLIK decomposition ----
    print("\nDecomposing QLIK losses ...")
    dec_heavy = qlik_decomposed(heavy.H_seq_, P)
    dec_garch = qlik_decomposed(garch.H_seq_, P)
    dec_garch_x = qlik_decomposed(garch_x.H_seq_, P)

    joint_gain, m1_gain, m2_gain, cop_gain = _decomposed_gains(dec_heavy, dec_garch, dates)

    # ---- i) Save CSV ----
    results_df = pd.DataFrame({
        'model': ['HEAVY', 'GARCH', 'GARCH-X'],
        'a': [a_H, a_G, a_GX],
        'b': [b_H, b_G, b_GX],
        'a_M': [a_M, np.nan, np.nan],
        'b_M': [b_M, np.nan, np.nan],
        'd_GX': [np.nan, np.nan, d_GX],
        'loglik': [heavy.loglik_H_, garch.loglik_, garch_x.loglik_],
        'mean_joint_qlik': [
            np.mean(dec_heavy['joint']),
            np.mean(dec_garch['joint']),
            np.mean(dec_garch_x['joint']),
        ],
        'mean_margin1_qlik': [
            np.mean(dec_heavy['margins'][:, 0]),
            np.mean(dec_garch['margins'][:, 0]),
            np.mean(dec_garch_x['margins'][:, 0]),
        ],
        'mean_margin2_qlik': [
            np.mean(dec_heavy['margins'][:, 1]),
            np.mean(dec_garch['margins'][:, 1]),
            np.mean(dec_garch_x['margins'][:, 1]),
        ],
        'mean_copula_qlik': [
            np.mean(dec_heavy['copula']),
            np.mean(dec_garch['copula']),
            np.mean(dec_garch_x['copula']),
        ],
    })

    gain_row = pd.DataFrame({
        'model': ['HEAVY_gain_vs_GARCH'],
        'a': [np.nan], 'b': [np.nan], 'a_M': [np.nan], 'b_M': [np.nan],
        'd_GX': [np.nan], 'loglik': [heavy.loglik_H_ - garch.loglik_],
        'mean_joint_qlik': [joint_gain],
        'mean_margin1_qlik': [m1_gain],
        'mean_margin2_qlik': [m2_gain],
        'mean_copula_qlik': [cop_gain],
    })
    results_df = pd.concat([results_df, gain_row], ignore_index=True)

    csv_path = os.path.join(RESULTS_DIR, 'exp1_insample_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # ---- j-k) Plots ----
    print("\nGenerating plots ...")
    _plot_realized_vol(V, dates, os.path.join(RESULTS_DIR, 'exp1_realized_vol.png'))
    _plot_realized_corr(V, dates, os.path.join(RESULTS_DIR, 'exp1_realized_corr.png'))
    _plot_conditional_vol(heavy.H_seq_, garch.H_seq_, dates,
                          os.path.join(RESULTS_DIR, 'exp1_conditional_vol.png'))

    # ---- l) Summary ----
    summary = {
        'T': T_actual,
        'a_H': a_H, 'b_H': b_H, 'a_M': a_M, 'b_M': b_M,
        'a_G': a_G, 'b_G': b_G,
        'a_GX': a_GX, 'b_GX': b_GX, 'd_GX': d_GX,
        'heavy_loglik': heavy.loglik_H_,
        'garch_loglik': garch.loglik_,
        'garch_x_loglik': garch_x.loglik_,
        'joint_gain_HvsG': joint_gain,
        'margin1_gain_HvsG': m1_gain,
        'margin2_gain_HvsG': m2_gain,
        'copula_gain_HvsG': cop_gain,
    }

    print("\nExperiment 1 complete.")
    return summary


if __name__ == '__main__':
    result = run_experiment()
    print("\nSummary:")
    for k, v in result.items():
        print(f"  {k}: {v}")
