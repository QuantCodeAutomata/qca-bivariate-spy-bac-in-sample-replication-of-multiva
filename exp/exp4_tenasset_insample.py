"""
Experiment 4: Ten-asset in-sample HEAVY vs GARCH with and without covariance targeting.

Assets: BAC, JPM, IBM, MSFT, XOM, AA, AXP, DD, GE, KO  (k=10, k*=55)

For scalar models the full 10×10 matrix recursion is:
  H_t = C + b * H_{t-1} + a * V_{t-1}   (element-wise scalar coefficients)

WITHOUT covariance targeting (plain):
  C = (1-a-b) * omega_H  where omega_H = mean(P_t)  [standard targeting on H]

WITH covariance targeting (scaled V):
  V_tilde_t = V_t * kappa_scalar   (kappa_scalar = trace(omega_H) / trace(omega_M))
  H_t = (1-b-a_star) * omega_H + b * H_{t-1} + a_star * V_tilde_{t-1}
  This ensures E[V_tilde_t] matches omega_H in the trace sense.
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

from data.fetch_data import get_ten_asset_data
from data.generate_synthetic import generate_multiasset_data
from src.estimation.qmle import estimate_heavy, estimate_garch
from src.forecasting.qlik import qlik_decomposed

RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

ASSET_NAMES = ['BAC', 'JPM', 'IBM', 'MSFT', 'XOM', 'AA', 'AXP', 'DD', 'GE', 'KO']


# ---------------------------------------------------------------------------
# Targeted-V variant helper
# ---------------------------------------------------------------------------

def _scale_V_for_targeting(V, P):
    """
    Scale V_t so that its trace-unconditional mean equals that of P_t.

    kappa = trace(omega_M) / trace(omega_H)
    V_tilde_t = V_t / kappa  =>  trace(mean(V_tilde)) ≈ trace(mean(P))
    """
    omega_H_tr = float(np.mean([np.trace(P[t]) for t in range(len(P))]))
    omega_M_tr = float(np.mean([np.trace(V[t]) for t in range(len(V))]))
    if omega_M_tr < 1e-14:
        return V.copy()
    kappa = omega_M_tr / omega_H_tr
    return V / kappa


# ---------------------------------------------------------------------------
# Summary table printing
# ---------------------------------------------------------------------------

def _print_summary_table(results):
    """Print Table-II-style summary."""
    hdr = f"{'Model':>20} {'Variant':>12} {'a':>8} {'b':>8} {'loglik':>12}"
    print("\n" + "=" * len(hdr))
    print("Ten-Asset In-Sample Results (Table II style)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for row in results:
        print(f"{row['model']:>20} {row['variant']:>12} "
              f"{row['a']:>8.4f} {row['b']:>8.4f} {row['loglik']:>12.2f}")
    print("=" * len(hdr))


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_experiment(T=None, k_use=None):
    """
    Run Experiment 4: ten-asset in-sample HEAVY vs GARCH with/without targeting.

    Parameters
    ----------
    T     : dataset length (None = fetch_data default).
    k_use : number of assets to use (None = all 10).  Pass a small value such
            as 3 to obtain a fast smoke-test; production runs use k_use=None.

    Returns
    -------
    dict of results
    """
    print("\n" + "=" * 65)
    print("Experiment 4: Ten-Asset In-Sample HEAVY vs GARCH")
    print("  (with and without covariance targeting)")
    print("=" * 65)

    assets = ASSET_NAMES if k_use is None else ASSET_NAMES[:k_use]

    if T is not None:
        data = generate_multiasset_data(assets=assets, n_days=T, seed=42)
        print(f"Note: Using synthetic data with T={T}, k={len(assets)}")
    else:
        data = get_ten_asset_data()
    P = data['P']
    V = data['V']
    if k_use is not None and k_use < P.shape[1]:
        P = P[:, :k_use, :k_use]
        V = V[:, :k_use, :k_use]
    dates = data['dates']
    T_actual = len(P)
    k = P.shape[1]

    print(f"\nData: T={T_actual}, k={k}, k*={k*(k+1)//2}")
    print(f"Assets: {', '.join(assets)}")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")

    # ---- Scale V for targeted variant ----
    V_tilde = _scale_V_for_targeting(V, P)
    kappa = np.trace(np.mean(V, axis=0)) / np.trace(np.mean(P, axis=0))
    print(f"\nTrace-scaling kappa = {kappa:.4f}  (V_tilde = V / {kappa:.4f})")

    results_list = []
    model_store = {}

    nv = k * (k + 1) // 2   # length of vech(C_L) in packed params

    # ---- Model 1: GARCH (without extra targeting - uses omega_H=mean(P)) ----
    print("\nEstimating GARCH (plain) ...")
    garch_plain = estimate_garch(P)
    a_G = float(garch_plain.params_[nv + 1]) ** 2
    b_G = float(garch_plain.params_[nv]) ** 2
    print(f"  a_G={a_G:.4f}, b_G={b_G:.4f}, loglik={garch_plain.loglik_:.2f}")
    results_list.append({'model': 'GARCH', 'variant': 'plain', 'a': a_G, 'b': b_G,
                          'loglik': garch_plain.loglik_})
    model_store['garch_plain'] = garch_plain

    # ---- Model 2: GARCH (targeted - same for scalar model, included for completeness) ----
    # For scalar GARCH targeting means H_t = (1-a-b)*mean(P) + b*H_{t-1} + a*P_{t-1}
    # which is exactly what the plain version does already.
    print("\nEstimating GARCH (targeted) ... [equivalent to plain for scalar model]")
    garch_tgt = garch_plain  # same model
    results_list.append({'model': 'GARCH', 'variant': 'targeted',
                          'a': a_G, 'b': b_G, 'loglik': garch_plain.loglik_})
    model_store['garch_tgt'] = garch_tgt

    # ---- Model 3: HEAVY (plain = uses raw V, omega_M = mean(V)) ----
    print("\nEstimating HEAVY (plain, raw V) ...")
    heavy_plain = estimate_heavy(P, V)
    a_H = float(heavy_plain.params_H_[nv + 1]) ** 2
    b_H = float(heavy_plain.params_H_[nv]) ** 2
    a_M = float(heavy_plain.params_M_[nv + 1]) ** 2
    b_M = float(heavy_plain.params_M_[nv]) ** 2
    print(f"  a_H={a_H:.4f}, b_H={b_H:.4f}, a_M={a_M:.4f}, b_M={b_M:.4f}, "
          f"loglik={heavy_plain.loglik_H_:.2f}")
    results_list.append({'model': 'HEAVY', 'variant': 'plain',
                          'a': a_H, 'b': b_H, 'loglik': heavy_plain.loglik_H_})
    model_store['heavy_plain'] = heavy_plain

    # ---- Model 4: HEAVY (targeted = uses V_tilde, omega_M ≈ omega_H in trace sense) ----
    print("\nEstimating HEAVY (targeted, V_tilde scaled) ...")
    heavy_tgt = estimate_heavy(P, V_tilde)
    a_Ht = float(heavy_tgt.params_H_[nv + 1]) ** 2
    b_Ht = float(heavy_tgt.params_H_[nv]) ** 2
    a_Mt = float(heavy_tgt.params_M_[nv + 1]) ** 2
    b_Mt = float(heavy_tgt.params_M_[nv]) ** 2
    print(f"  a_H={a_Ht:.4f}, b_H={b_Ht:.4f}, a_M={a_Mt:.4f}, b_M={b_Mt:.4f}, "
          f"loglik={heavy_tgt.loglik_H_:.2f}")
    results_list.append({'model': 'HEAVY', 'variant': 'targeted',
                          'a': a_Ht, 'b': b_Ht, 'loglik': heavy_tgt.loglik_H_})
    model_store['heavy_tgt'] = heavy_tgt

    _print_summary_table(results_list)

    # ---- QLIK decomposition: margins + copula ----
    print("\nDecomposing QLIK losses ...")
    dec_garch = qlik_decomposed(garch_plain.H_seq_, P)
    dec_heavy_plain = qlik_decomposed(heavy_plain.H_seq_, P)
    dec_heavy_tgt = qlik_decomposed(heavy_tgt.H_seq_, P)

    def _margin_gains(dec_heavy, dec_garch):
        joint_gain = np.mean(dec_heavy['joint']) - np.mean(dec_garch['joint'])
        margin_gains = np.mean(dec_heavy['margins'], axis=0) - np.mean(dec_garch['margins'], axis=0)
        copula_gain = np.mean(dec_heavy['copula']) - np.mean(dec_garch['copula'])
        return joint_gain, margin_gains, copula_gain

    joint_gain_plain, mg_plain, cop_gain_plain = _margin_gains(dec_heavy_plain, dec_garch)
    joint_gain_tgt, mg_tgt, cop_gain_tgt = _margin_gains(dec_heavy_tgt, dec_garch)

    print("\nQKLIK gain (HEAVY - GARCH; negative = HEAVY wins):")
    print(f"  Plain HEAVY   - joint: {joint_gain_plain:+.4f}, copula: {cop_gain_plain:+.4f}")
    print(f"  Targeted HEAVY - joint: {joint_gain_tgt:+.4f}, copula: {cop_gain_tgt:+.4f}")

    # Per-margin gains
    print(f"\n  Per-margin gains (plain HEAVY vs GARCH):")
    for i, name in enumerate(assets):
        print(f"    {name:>6}: {mg_plain[i]:+.4f}")
    print(f"    {'Copula':>6}: {cop_gain_plain:+.4f}")

    # ---- Save CSV ----
    rows_all = []
    for variant, jg, mg, cg in [('plain', joint_gain_plain, mg_plain, cop_gain_plain),
                                  ('targeted', joint_gain_tgt, mg_tgt, cop_gain_tgt)]:
        row = {'variant': variant, 'model': 'HEAVY_gain_vs_GARCH',
               'joint_gain': jg, 'copula_gain': cg}
        for i, name in enumerate(assets):
            row[f'margin_{name}'] = mg[i]
        rows_all.append(row)

    # Also record model params
    for r in results_list:
        rows_all.append(r)

    results_df = pd.DataFrame(rows_all)
    csv_path = os.path.join(RESULTS_DIR, 'exp4_tenasset_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # ---- Bar chart: margin contributions ----
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(k + 1)
    labels = list(assets) + ['Copula']
    gains_plain = list(mg_plain) + [cop_gain_plain]
    gains_tgt = list(mg_tgt) + [cop_gain_tgt]

    width = 0.38
    bars1 = ax.bar(x - width/2, gains_plain, width, label='HEAVY (plain)', color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + width/2, gains_tgt, width, label='HEAVY (targeted)', color='firebrick', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('QLIK gain (HEAVY − GARCH)')
    ax.set_title('Ten-Asset Margin QLIK Gain Decomposition\n(negative = HEAVY wins)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'exp4_margin_contributions.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to: {png_path}")

    summary = {
        'T': T_actual,
        'k': k,
        'heavy_plain_a': a_H, 'heavy_plain_b': b_H,
        'heavy_tgt_a': a_Ht, 'heavy_tgt_b': b_Ht,
        'garch_a': a_G, 'garch_b': b_G,
        'heavy_plain_loglik': heavy_plain.loglik_H_,
        'heavy_tgt_loglik': heavy_tgt.loglik_H_,
        'garch_loglik': garch_plain.loglik_,
        'joint_gain_plain': joint_gain_plain,
        'joint_gain_tgt': joint_gain_tgt,
        'margin_gains_plain': mg_plain.tolist(),
        'copula_gain_plain': cop_gain_plain,
        'results_df': results_df,
    }
    print("\nExperiment 4 complete.")
    return summary


if __name__ == '__main__':
    result = run_experiment()
    print("\nSummary:")
    for k, v in result.items():
        if k != 'results_df':
            print(f"  {k}: {v}")
