"""
Experiment 3: Sensitivity of HEAVY estimation to realized measure construction.

We simulate different sampling frequencies [1, 5, 10, 15, 30] minutes by
adding frequency-dependent microstructure noise to the true realized covariance.
A 'realized kernel' variant (flat-top kernel) is also included.

Key insight:
  - Higher frequency (1-min) -> more microstructure noise -> noisier V_t
  - Lower frequency (30-min) -> smoother but higher-variance V_t
  - Both effects alter estimated persistence parameters.
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

from data.generate_synthetic import generate_bivariate_data
from src.estimation.qmle import estimate_heavy, estimate_garch_x
from src.forecasting.qlik import qlik_decomposed

RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Noise scales keyed by (label, frequency_minutes)
# At 1-min: high microstructure noise (large noise_scale)
# At 30-min: lower but coarser (moderate noise_scale, lower SNR from fewer observations)
FREQ_CONFIGS = [
    ('1-min',  1,  0.35),   # highest noise (microstructure dominant)
    ('5-min',  5,  0.18),   # typical choice in literature
    ('10-min', 10, 0.12),   # moderate
    ('15-min', 15, 0.08),   # lower noise
    ('30-min', 30, 0.05),   # smoothest realized measure
    ('RK',     5,  0.10),   # realized kernel: flatter frequency-domain correction
]


# ---------------------------------------------------------------------------
# Realized measure variants
# ---------------------------------------------------------------------------

def _add_microstructure_noise(V_true, noise_scale, rng):
    """
    Perturb true realized covariance V_t with multiplicative noise.

    V_t^{(f)} = V_t * exp(noise_scale * xi_t)  (element-wise, then symmetrized and PD-projected)

    Higher noise_scale -> noisier realized measure.
    """
    T, k, _ = V_true.shape
    V_noisy = np.empty_like(V_true)
    for t in range(T):
        # Symmetric noise matrix
        Z = rng.standard_normal((k, k))
        xi = 0.5 * (Z + Z.T) / np.sqrt(k)
        V_t = V_true[t] * (1.0 + noise_scale * xi)
        # Project to PD
        w, ev = np.linalg.eigh(V_t)
        w = np.maximum(w, 1e-10)
        V_noisy[t] = ev @ np.diag(w) @ ev.T
    return V_noisy


def _realized_kernel_variant(V_true, noise_scale, rng):
    """
    Simulate a flat-top realized kernel: reduce noise relative to raw 5-min
    by applying a dampening factor to off-diagonal noise.
    """
    T, k, _ = V_true.shape
    V_rk = np.empty_like(V_true)
    for t in range(T):
        Z = rng.standard_normal((k, k))
        # Flat-top kernel: suppress off-diagonal noise by factor 0.5
        xi = 0.5 * (Z + Z.T) / np.sqrt(k)
        xi_diag = np.diag(np.diag(xi))
        xi_off = xi - xi_diag
        xi_combined = xi_diag + 0.5 * xi_off
        V_t = V_true[t] * (1.0 + noise_scale * xi_combined)
        w, ev = np.linalg.eigh(V_t)
        w = np.maximum(w, 1e-10)
        V_rk[t] = ev @ np.diag(w) @ ev.T
    return V_rk


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_experiment(T=None):
    """
    Run Experiment 3: sensitivity to realized measure frequency.

    Parameters
    ----------
    T : dataset length (None = 500)

    Returns
    -------
    dict with results DataFrame and parameter arrays
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Sensitivity to Realized Measure Construction")
    print("=" * 60)

    T_use = T if T is not None else 500
    rng = np.random.default_rng(seed=123)

    # Load base data (synthetic, V = true realized covariance)
    base_data = generate_bivariate_data(n_days=T_use, seed=42)
    P = base_data['P']
    V_true = base_data['V']   # baseline 5-min-equivalent

    print(f"\nBase data: T={T_use}, k=2")
    print("Note: R_t and P_t are FIXED across frequency variants; only V_t varies.")

    # Baseline GARCH (no V, so invariant to frequency)
    from src.estimation.qmle import estimate_garch
    print("\nEstimating baseline GARCH (invariant to frequency) ...")
    garch_base = estimate_garch(P)
    dec_garch = qlik_decomposed(garch_base.H_seq_, P)
    baseline_joint = float(np.mean(dec_garch['joint']))
    print(f"  GARCH joint mean QLIK: {baseline_joint:.4f}")

    rows = []
    param_records = []

    for label, freq_min, noise_scale in FREQ_CONFIGS:
        print(f"\n  Variant: {label} (freq={freq_min} min, noise_scale={noise_scale:.2f}) ...")

        if label == 'RK':
            V_var = _realized_kernel_variant(V_true, noise_scale, rng)
        else:
            V_var = _add_microstructure_noise(V_true, noise_scale, rng)

        heavy = estimate_heavy(P, V_var)
        k = P.shape[1]
        nv = k * (k + 1) // 2
        a_H = float(heavy.params_H_[nv + 1]) ** 2
        b_H = float(heavy.params_H_[nv]) ** 2
        a_M = float(heavy.params_M_[nv + 1]) ** 2
        b_M = float(heavy.params_M_[nv]) ** 2
        print(f"    HEAVY: a_H={a_H:.4f}, b_H={b_H:.4f}, a_M={a_M:.4f}, b_M={b_M:.4f}, loglik={heavy.loglik_H_:.2f}")

        dec_heavy = qlik_decomposed(heavy.H_seq_, P)
        joint_gain = float(np.mean(dec_heavy['joint'])) - baseline_joint

        garch_x = estimate_garch_x(P, V_var)
        a_GX = float(garch_x.params_[nv + 1]) ** 2
        b_GX = float(garch_x.params_[nv]) ** 2
        d_GX = float(garch_x.params_[nv + 2]) ** 2
        print(f"    GARCH-X: a_GX={a_GX:.4f}, b_GX={b_GX:.4f}, d_GX={d_GX:.4f}")

        rows.append({
            'freq_label': label,
            'freq_min': freq_min,
            'noise_scale': noise_scale,
            'a_H': a_H,
            'b_H': b_H,
            'a_M': a_M,
            'b_M': b_M,
            'heavy_loglik': heavy.loglik_H_,
            'joint_gain_vs_garch': joint_gain,
            'a_GX': a_GX,
            'b_GX': b_GX,
            'd_GX': d_GX,
        })
        param_records.append({'label': label, 'a_H': a_H, 'b_H': b_H})

    results_df = pd.DataFrame(rows)

    # ---- Print summary table ----
    print("\n" + "=" * 70)
    print("Sensitivity Summary: freq | a_H | b_H | a_M | b_M | joint_gain")
    print("=" * 70)
    print(f"{'Freq':>8} {'a_H':>8} {'b_H':>8} {'a_M':>8} {'b_M':>8} {'joint_gain':>12}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['freq_label']:>8} {row['a_H']:>8.4f} {row['b_H']:>8.4f} "
              f"{row['a_M']:>8.4f} {row['b_M']:>8.4f} {row['joint_gain_vs_garch']:>12.4f}")
    print("=" * 70)

    # ---- Save CSV ----
    csv_path = os.path.join(RESULTS_DIR, 'exp3_sensitivity_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # ---- Plot: frequency vs a_H and b_H ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = results_df['freq_label'].tolist()
    x = np.arange(len(labels))

    axes[0].bar(x, results_df['a_H'], color='steelblue', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].set_title('HEAVY: α_H (impact coefficient)')
    axes[0].set_ylabel('a_H')
    axes[0].grid(True, axis='y', alpha=0.4)

    axes[1].bar(x, results_df['b_H'], color='firebrick', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha='right')
    axes[1].set_title('HEAVY: β_H (persistence coefficient)')
    axes[1].set_ylabel('b_H')
    axes[1].grid(True, axis='y', alpha=0.4)

    fig.suptitle('Sensitivity of HEAVY Parameters to Realized Measure Frequency', fontsize=11)
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'exp3_sensitivity_params.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to: {png_path}")

    summary = {
        'T': T_use,
        'results_df': results_df,
        'baseline_garch_joint_qlik': baseline_joint,
    }
    print("\nExperiment 3 complete.")
    return summary


if __name__ == '__main__':
    result = run_experiment()
    print("\nSummary:")
    print(result['results_df'][['freq_label', 'a_H', 'b_H', 'a_M', 'b_M', 'joint_gain_vs_garch']].to_string(index=False))
