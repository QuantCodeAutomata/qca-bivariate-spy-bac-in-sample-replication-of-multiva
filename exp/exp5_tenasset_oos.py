"""
Experiment 5: Ten-asset rolling out-of-sample forecast comparison.

Covariance-targeted HEAVY vs GARCH for horizons s=1..22.
Reports joint, copula, and per-margin QLIK t-statistics.
Also summarises min/median/max t-stat across 10 margins at each horizon.

Reproduces Figure-4-style plot from Shephard & Sheppard (2010).
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
from src.forecasting.multistep import heavy_multistep_forecast, garch_multistep_forecast
from src.forecasting.qlik import qlik_loss_t
from src.estimation.inference import dm_test

RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

ASSET_NAMES = ['BAC', 'JPM', 'IBM', 'MSFT', 'XOM', 'AA', 'AXP', 'DD', 'GE', 'KO']
FULL_WINDOW = 1486
ALL_HORIZONS = list(range(1, 23))   # 1..22


# ---------------------------------------------------------------------------
# Targeted-V helper
# ---------------------------------------------------------------------------

def _scale_V(V, P):
    """Scale V_t by trace(omega_H) / trace(omega_M)."""
    omega_H_tr = float(np.mean([np.trace(P[t]) for t in range(len(P))]))
    omega_M_tr = float(np.mean([np.trace(V[t]) for t in range(len(V))]))
    return V / max(omega_M_tr / omega_H_tr, 1e-10)


def _reconstruct_C(params, k):
    """Reconstruct intercept C = C_L @ C_L.T from packed [vech(C_L), ...] params."""
    nv = k * (k + 1) // 2
    v = params[:nv]
    L = np.zeros((k, k))
    idx = 0
    for i in range(k):
        for j in range(i + 1):
            L[i, j] = float(v[idx])
            idx += 1
    return L @ L.T


def _heavy_one_step(heavy, V_last, k):
    """H_{T+1|T} and M_{T+1|T} from a fitted HEAVY model (uses observed V_T)."""
    nv = k * (k + 1) // 2
    C_H = _reconstruct_C(heavy.params_H_, k)
    b_H = float(heavy.params_H_[nv]) ** 2
    a_H = float(heavy.params_H_[nv + 1]) ** 2
    C_M = _reconstruct_C(heavy.params_M_, k)
    b_M = float(heavy.params_M_[nv]) ** 2
    a_M = float(heavy.params_M_[nv + 1]) ** 2
    H_T = heavy.H_seq_[-1]
    M_T = heavy.M_seq_[-1]
    eps = np.eye(k) * 1e-10
    H1 = C_H + b_H * H_T + a_H * V_last
    H1 = 0.5 * (H1 + H1.T) + eps
    M1 = C_M + b_M * M_T + a_M * V_last
    M1 = 0.5 * (M1 + M1.T) + eps
    return H1, M1


def _garch_one_step(garch, P_last, k):
    """H_{T+1|T} from a fitted GARCH model using the observed P_T."""
    nv = k * (k + 1) // 2
    C_G = _reconstruct_C(garch.params_, k)
    b_G = float(garch.params_[nv]) ** 2
    a_G = float(garch.params_[nv + 1]) ** 2
    H_T = garch.H_seq_[-1]
    H1 = C_G + b_G * H_T + a_G * P_last
    eps = np.eye(k) * 1e-10
    return 0.5 * (H1 + H1.T) + eps


def _clip_pd(M, eps=1e-10):
    """Project (k,k) matrix to nearest PD matrix via eigendecomposition."""
    w, ev = np.linalg.eigh(M)
    return ev @ np.diag(np.maximum(w, eps)) @ ev.T


# ---------------------------------------------------------------------------
# Single forecast-origin evaluation (multi-horizon)
# ---------------------------------------------------------------------------

def _evaluate_origin_multihorizon(t, Q, P, V, horizons):
    """
    Estimate targeted HEAVY and GARCH on [t-Q:t], compute forecasts for all
    horizons, return per-horizon loss differentials (HEAVY - GARCH).

    Returns dict horizon -> {'joint', 'margins': list, 'copula'} or None.
    """
    T = len(P)
    max_s = max(horizons)
    if t + max_s > T:
        return None

    k = P.shape[1]
    P_win = P[t - Q: t]
    V_win = V[t - Q: t]
    V_win_tilde = _scale_V(V_win, P_win)

    heavy = estimate_heavy(P_win, V_win_tilde)
    garch = estimate_garch(P_win)

    # One-step forecasts using observed last values
    V_last = V_win_tilde[-1]
    P_last = P_win[-1]
    H1_heavy, M1_heavy = _heavy_one_step(heavy, V_last, k)
    H1_garch = _garch_one_step(garch, P_last, k)

    # Compute all multistep forecasts at once (more efficient than per-horizon)
    all_heavy = heavy_multistep_forecast(
        heavy.params_H_, heavy.params_M_, H1_heavy, M1_heavy, s_max=max_s)
    all_garch = garch_multistep_forecast(garch.params_, H1_garch, s_max=max_s)

    results = {}
    for s in horizons:
        if t + s > T:
            continue

        Hs_heavy = _clip_pd(all_heavy[s - 1])
        Hs_garch = _clip_pd(all_garch[s - 1])

        P_target = P[t + s - 1]

        loss_heavy = qlik_loss_t(Hs_heavy, P_target)
        loss_garch = qlik_loss_t(Hs_garch, P_target)

        m_heavy = [np.log(Hs_heavy[i, i]) + P_target[i, i] / Hs_heavy[i, i] for i in range(k)]
        m_garch = [np.log(Hs_garch[i, i]) + P_target[i, i] / Hs_garch[i, i] for i in range(k)]
        cop_heavy = loss_heavy - sum(m_heavy)
        cop_garch = loss_garch - sum(m_garch)

        results[s] = {
            'joint': loss_heavy - loss_garch,
            'margins': [m_heavy[i] - m_garch[i] for i in range(k)],
            'copula': cop_heavy - cop_garch,
        }
    return results


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_experiment(T=None, Q=None, skip=5, horizons=None, k_use=None):
    """
    Run Experiment 5: ten-asset rolling OOS with covariance targeting.

    Parameters
    ----------
    T        : dataset length
    Q        : rolling window (None -> min(FULL_WINDOW, T//3))
    skip     : use every SKIP-th forecast origin
    horizons : list of horizons (None -> 1..22)
    k_use    : number of assets to use (None = all 10). Pass a small value
               such as 3 for fast smoke-tests; production uses k_use=None.

    Returns
    -------
    dict of results
    """
    print("\n" + "=" * 65)
    print("Experiment 5: Ten-Asset Rolling OOS (Covariance-Targeted Models)")
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

    Q_use = Q if Q is not None else min(FULL_WINDOW, T_actual // 3)
    horizons_use = horizons if horizons is not None else ALL_HORIZONS
    max_s = max(horizons_use)

    n_origins_full = T_actual - Q_use - max_s
    if n_origins_full <= 0:
        raise ValueError(f"Dataset too short: T={T_actual}, Q={Q_use}, max_horizon={max_s}")

    print(f"\nT={T_actual}, Q={Q_use}, k={k}, horizons={horizons_use}")
    print(f"Using every {skip}-th forecast origin (~{n_origins_full // skip} origins)")

    # Accumulate loss differentials per horizon
    diffs = {s: {'joint': [], 'margins': [[] for _ in range(k)], 'copula': []}
             for s in horizons_use}

    origins = list(range(Q_use, T_actual - max_s, skip))
    n_origins = len(origins)

    for idx, t in enumerate(origins):
        result = _evaluate_origin_multihorizon(t, Q_use, P, V, horizons_use)
        if result is None:
            continue
        for s, vals in result.items():
            diffs[s]['joint'].append(vals['joint'])
            diffs[s]['copula'].append(vals['copula'])
            for i in range(k):
                diffs[s]['margins'][i].append(vals['margins'][i])

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{n_origins}] origins processed")

    print(f"  Done. {n_origins} origins evaluated.")

    # ---- Compute t-stats ----
    all_rows = []
    for s in horizons_use:
        joint_d = np.array(diffs[s]['joint'])
        cop_d = np.array(diffs[s]['copula'])
        if len(joint_d) < 5:
            continue

        row = {'horizon': s, 'n_evals': len(joint_d)}
        row['mean_joint'] = float(np.mean(joint_d))
        row['t_joint'] = float(dm_test(joint_d))
        row['mean_copula'] = float(np.mean(cop_d))
        row['t_copula'] = float(dm_test(cop_d))

        margin_tstats = []
        for i in range(k):
            md = np.array(diffs[s]['margins'][i])
            row[f'mean_margin_{ASSET_NAMES[i]}'] = float(np.mean(md))
            tstat_i = float(dm_test(md))
            row[f't_margin_{ASSET_NAMES[i]}'] = tstat_i
            margin_tstats.append(tstat_i)

        row['t_min_margin'] = float(np.min(margin_tstats))
        row['t_median_margin'] = float(np.median(margin_tstats))
        row['t_max_margin'] = float(np.max(margin_tstats))
        all_rows.append(row)

    if not all_rows:
        print("No valid results.")
        return {}

    results_df = pd.DataFrame(all_rows)

    # ---- Print summary ----
    print("\n" + "=" * 80)
    print("Ten-Asset OOS: DM t-statistics (negative = HEAVY wins)")
    print("=" * 80)
    print(f"{'Horizon':>8} {'t_joint':>10} {'t_copula':>10} "
          f"{'t_min_m':>10} {'t_med_m':>10} {'t_max_m':>10}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['horizon']:>8.0f} {row['t_joint']:>10.3f} {row['t_copula']:>10.3f} "
              f"{row['t_min_margin']:>10.3f} {row['t_median_margin']:>10.3f} "
              f"{row['t_max_margin']:>10.3f}")
    print("=" * 80)

    # ---- Save CSV ----
    csv_path = os.path.join(RESULTS_DIR, 'exp5_tenasset_oos_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # ---- Figure-4-style plot ----
    fig, ax = plt.subplots(figsize=(11, 5))
    h = results_df['horizon']
    ax.plot(h, results_df['t_joint'], '-o', color='black', linewidth=1.5, label='Joint')
    ax.plot(h, results_df['t_copula'], '-s', color='steelblue', linewidth=1.3, label='Copula')
    ax.fill_between(h, results_df['t_min_margin'], results_df['t_max_margin'],
                    alpha=0.25, color='firebrick', label='Margin range (min–max)')
    ax.plot(h, results_df['t_median_margin'], '--^', color='firebrick', linewidth=1.3,
            label='Margin median')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(-1.645, color='darkred', linewidth=0.8, linestyle='--',
               label='−1.645 (5% one-sided)')
    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel('DM t-statistic')
    ax.set_title('Figure 4 Style: Ten-Asset OOS t-statistics\n'
                 'Covariance-Targeted HEAVY vs GARCH  (negative = HEAVY wins)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'exp5_tenasset_oos_figure4.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to: {png_path}")

    summary = {
        'T': T_actual,
        'Q': Q_use,
        'k': k,
        'skip': skip,
        'horizons': horizons_use,
        'results_df': results_df,
    }
    print("\nExperiment 5 complete.")
    return summary


if __name__ == '__main__':
    result = run_experiment()
    if result:
        print("\nSummary columns:")
        df = result['results_df']
        cols = ['horizon', 't_joint', 't_copula', 't_min_margin', 't_median_margin', 't_max_margin']
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
