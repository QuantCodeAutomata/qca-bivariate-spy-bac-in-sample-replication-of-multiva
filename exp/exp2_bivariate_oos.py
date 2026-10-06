"""
Experiment 2: Rolling out-of-sample forecast comparison HEAVY vs GARCH (bivariate).

For horizons s in {1, 2, 3, 5, 10, 22}:
  - Roll a window of length Q forward through the dataset.
  - At each origin t, estimate HEAVY and GARCH on window [t-Q+1, t].
  - Compute s-step forecasts.
  - Evaluate QLIK loss differentials D_{t,s} = QLIK(HEAVY) - QLIK(GARCH).
  - Compute decomposed DM t-statistics (joint, copula, margin 1, margin 2).

IMPORTANT: Rolling estimation is expensive.  We use every SKIP-th day as a
forecast origin to keep runtime tractable.  This is noted in the output.
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
from src.estimation.qmle import estimate_heavy, estimate_garch
from src.forecasting.multistep import heavy_multistep_forecast, garch_multistep_forecast
from src.forecasting.qlik import qlik_loss_t
from src.estimation.inference import dm_test

RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

FULL_WINDOW = 1486  # rolling window for production run
HORIZONS = [1, 2, 3, 5, 10, 22]


# ---------------------------------------------------------------------------
# Helpers for one-step-ahead forecasts (replaces non-existent standalone fns)
# ---------------------------------------------------------------------------

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
    """Compute H_{T+1|T} and M_{T+1|T} from a fitted HEAVY model.

    Uses the observed V_T (=V_last) for the one-step update, which is
    the correct conditioning information at forecast origin T.
    """
    nv = k * (k + 1) // 2
    C_H = _reconstruct_C(heavy.params_H_, k)
    b_H = float(heavy.params_H_[nv]) ** 2    # b_raw^2
    a_H = float(heavy.params_H_[nv + 1]) ** 2  # a_raw^2

    C_M = _reconstruct_C(heavy.params_M_, k)
    b_M = float(heavy.params_M_[nv]) ** 2
    a_M = float(heavy.params_M_[nv + 1]) ** 2

    H_T = heavy.H_seq_[-1]
    M_T = heavy.M_seq_[-1]

    H1 = C_H + b_H * H_T + a_H * V_last
    M1 = C_M + b_M * M_T + a_M * V_last
    eps = np.eye(k) * 1e-10
    H1 = 0.5 * (H1 + H1.T) + eps
    M1 = 0.5 * (M1 + M1.T) + eps
    return H1, M1


def _garch_one_step(garch, P_last, k):
    """Compute H_{T+1|T} from a fitted GARCH model using the observed P_T."""
    nv = k * (k + 1) // 2
    C_G = _reconstruct_C(garch.params_, k)
    b_G = float(garch.params_[nv]) ** 2
    a_G = float(garch.params_[nv + 1]) ** 2
    H_T = garch.H_seq_[-1]
    H1 = C_G + b_G * H_T + a_G * P_last
    eps = np.eye(k) * 1e-10
    return 0.5 * (H1 + H1.T) + eps


# ---------------------------------------------------------------------------
# Single forecast-origin evaluation
# ---------------------------------------------------------------------------

def _evaluate_origin_multihorizon(t, Q, P, V, horizons):
    """
    Estimate HEAVY and GARCH once on [t-Q:t], then evaluate all horizons.

    Estimating once per origin (not once per (s,t) pair) is ~6× faster.
    Returns dict horizon -> {'joint','margin1','margin2','copula'} or None if
    t + max(horizons) > T.
    """
    T_total = len(P)
    max_s = max(horizons)
    if t + max_s > T_total:
        return None

    k = P.shape[1]
    P_win = P[t - Q: t]
    V_win = V[t - Q: t]

    heavy = estimate_heavy(P_win, V_win)
    garch = estimate_garch(P_win)

    V_last = V_win[-1]
    P_last = P_win[-1]
    H1_heavy, M1_heavy = _heavy_one_step(heavy, V_last, k)
    H1_garch = _garch_one_step(garch, P_last, k)

    all_heavy = heavy_multistep_forecast(
        heavy.params_H_, heavy.params_M_, H1_heavy, M1_heavy, s_max=max_s)
    all_garch = garch_multistep_forecast(garch.params_, H1_garch, s_max=max_s)

    results = {}
    for s in horizons:
        if t + s > T_total:
            continue
        Hs_heavy = all_heavy[s - 1]
        Hs_garch = all_garch[s - 1]
        P_target = P[t + s - 1]

        loss_heavy = qlik_loss_t(Hs_heavy, P_target)
        loss_garch = qlik_loss_t(Hs_garch, P_target)

        m_heavy = [np.log(Hs_heavy[i, i]) + P_target[i, i] / Hs_heavy[i, i] for i in range(k)]
        m_garch = [np.log(Hs_garch[i, i]) + P_target[i, i] / Hs_garch[i, i] for i in range(k)]
        cop_heavy = loss_heavy - sum(m_heavy)
        cop_garch = loss_garch - sum(m_garch)

        results[s] = {
            'joint':   loss_heavy - loss_garch,
            'margin1': m_heavy[0] - m_garch[0],
            'margin2': m_heavy[1] - m_garch[1],
            'copula':  cop_heavy  - cop_garch,
        }
    return results


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_experiment(T=None, Q=None, skip=5):
    """
    Run Experiment 2: rolling OOS comparison.

    Parameters
    ----------
    T    : dataset length (None = fetch_data default)
    Q    : rolling window length (None -> min(FULL_WINDOW, T//3))
    skip : use every SKIP-th forecast origin to reduce compute time

    Returns
    -------
    dict of results
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Bivariate Rolling OOS HEAVY vs GARCH")
    print("=" * 60)

    if T is not None:
        data = generate_bivariate_data(n_days=T, seed=42)
        print(f"Note: Using synthetic data with T={T}")
    else:
        data = get_bivariate_data()
    P = data['P']
    V = data['V']
    dates = data['dates']
    T_actual = len(P)

    Q_use = Q if Q is not None else min(FULL_WINDOW, T_actual // 3)
    n_origins_full = T_actual - Q_use - max(HORIZONS)
    if n_origins_full <= 0:
        raise ValueError(f"Dataset too short: T={T_actual}, Q={Q_use}, max_horizon={max(HORIZONS)}")

    print(f"\nT={T_actual}, Q={Q_use}, max_horizon={max(HORIZONS)}")
    print(f"Full origins available: {n_origins_full}")
    print(f"Using every {skip}-th origin (skip={skip}): approx {n_origins_full // skip} origins per horizon")

    all_rows = []
    oos_diffs = {s: {'joint': [], 'margin1': [], 'margin2': [], 'copula': []} for s in HORIZONS}

    # Estimate ONCE per origin, evaluate ALL horizons — avoids 6x redundant fitting
    max_s = max(HORIZONS)
    origins = list(range(Q_use, T_actual - max_s, skip))
    n_origins = len(origins)
    print(f"\n  Evaluating {n_origins} origins for all horizons {HORIZONS} ...")

    for done, t in enumerate(origins, 1):
        result = _evaluate_origin_multihorizon(t, Q_use, P, V, HORIZONS)
        if result is not None:
            for s, dct in result.items():
                for key in ('joint', 'margin1', 'margin2', 'copula'):
                    oos_diffs[s][key].append(dct[key])
        if done % 10 == 0 or done == n_origins:
            print(f"    [{done}/{n_origins}] origins processed")

    for s in HORIZONS:
        n_valid = len(oos_diffs[s]['joint'])
        print(f"  Horizon s={s:2d}: {n_valid} valid evaluations")

        if n_valid < 5:
            print(f"    Insufficient valid evaluations for s={s}; skipping.")
            continue

        row = {'horizon': s, 'n_evals': n_valid}
        for key in ('joint', 'margin1', 'margin2', 'copula'):
            diffs = np.array(oos_diffs[s][key])
            row[f'mean_{key}'] = float(np.mean(diffs))
            row[f't_{key}'] = float(dm_test(diffs))
        all_rows.append(row)

    if not all_rows:
        print("No valid results produced.")
        return {}

    results_df = pd.DataFrame(all_rows)

    print("\n" + "=" * 75)
    print("OOS Results: DM t-statistics (negative = HEAVY wins)")
    print("=" * 75)
    print(f"{'Horizon':>8} {'t_joint':>10} {'t_margin1':>12} {'t_margin2':>12} {'t_copula':>10}")
    print("-" * 75)
    for _, row in results_df.iterrows():
        print(f"{row['horizon']:>8.0f} {row['t_joint']:>10.3f} "
              f"{row['t_margin1']:>12.3f} {row['t_margin2']:>12.3f} "
              f"{row['t_copula']:>10.3f}")
    print("=" * 75)

    csv_path = os.path.join(RESULTS_DIR, 'exp2_oos_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # Plot horizon profile of t-statistics
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, label, style in [
        ('t_joint',   'Joint',    '-o'),
        ('t_margin1', 'Margin 1', '--s'),
        ('t_margin2', 'Margin 2', '--^'),
        ('t_copula',  'Copula',   ':d'),
    ]:
        if col in results_df.columns:
            ax.plot(results_df['horizon'], results_df[col], style, label=label, linewidth=1.4)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.axhline(-1.645, color='red', linewidth=0.8, linestyle='--', label='−1.645 (5% one-sided)')
    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel('DM t-statistic')
    ax.set_title('OOS t-statistics: HEAVY vs GARCH (bivariate)\n'
                 'Negative = HEAVY forecast is more accurate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'exp2_oos_tstat_profile.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to: {png_path}")

    summary = {
        'T': T_actual,
        'Q': Q_use,
        'skip': skip,
        'horizons': HORIZONS,
        'results_df': results_df,
    }
    print("\nExperiment 2 complete.")
    return summary


if __name__ == '__main__':
    result = run_experiment()
    if result:
        print("\nSummary:")
        print(f"  T={result['T']}, Q={result['Q']}, skip={result['skip']}")
        print(result['results_df'].to_string(index=False))
