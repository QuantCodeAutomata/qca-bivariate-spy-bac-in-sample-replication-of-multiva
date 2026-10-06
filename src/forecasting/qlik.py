"""
QLIK (quasi-log-likelihood) loss functions and the Diebold-Mariano test.

The joint QLIK loss at observation t is:

    L_t = log|H_t| + tr(H_t⁻¹ V_t)

where H_t is the forecast covariance and V_t is the realised covariance /
outer product of returns.

This module provides:

* :func:`qlik_loss`           — joint QLIK for a sequence of observations.
* :func:`qlik_margin_loss`    — marginal (univariate diagonal) QLIK.
* :func:`qlik_copula_loss`    — copula component = joint − Σ margins.
* :func:`loss_differential`   — QLIK difference between two forecast sequences.
* :func:`dm_test`             — HAC-robust Diebold-Mariano t-statistic.

Legacy helpers :func:`qlik_loss_t`, :func:`margin_loss`, :func:`qlik_decomposed`,
and :func:`mean_qlik_decomposed` are preserved for backward compatibility.
"""

from __future__ import annotations

import numpy as np

from src.utils.linalg import safe_inv, safe_logdet, trace_prod
from src.estimation.inference import hac_variance


# ---------------------------------------------------------------------------
# Core QLIK loss
# ---------------------------------------------------------------------------

def qlik_loss(
    H_forecast: np.ndarray,
    V_realized: np.ndarray,
) -> np.ndarray:
    """Per-observation QLIK loss: log|H_t| + tr(H_t⁻¹ V_t).

    Accepts either a single observation or a full sequence.

    Parameters
    ----------
    H_forecast:
        (k, k) or (T, k, k) forecast covariance matrices.
    V_realized:
        (k, k) or (T, k, k) realised covariances / outer products.

    Returns
    -------
    float or ndarray of shape (T,)
        QLIK loss for each observation.
    """
    H = np.asarray(H_forecast, dtype=float)
    V = np.asarray(V_realized, dtype=float)

    if H.ndim == 2:
        # Single observation
        ld = safe_logdet(H)
        return float(ld + trace_prod(safe_inv(H), V))

    T = H.shape[0]
    losses = np.empty(T)
    for t in range(T):
        H_t = H[t]
        ld = safe_logdet(H_t)
        losses[t] = ld + trace_prod(safe_inv(H_t), V[t])
    return losses


# ---------------------------------------------------------------------------
# Marginal QLIK
# ---------------------------------------------------------------------------

def qlik_margin_loss(
    h_ii_forecast: np.ndarray,
    V_ii_realized: np.ndarray,
) -> np.ndarray:
    """Element-wise marginal QLIK loss for a single diagonal element.

    For margin i:  L_t^{(i)} = log h_{ii,t} + V_{ii,t} / h_{ii,t}.

    Parameters
    ----------
    h_ii_forecast:
        Scalar or 1-D array of diagonal forecast values h_{ii,t}.
    V_ii_realized:
        Scalar or 1-D array of diagonal realised values V_{ii,t}.

    Returns
    -------
    float or ndarray of shape (T,)
    """
    h = np.asarray(h_ii_forecast, dtype=float)
    v = np.asarray(V_ii_realized, dtype=float)
    h = np.maximum(h, 1e-14)
    loss = np.log(h) + v / h
    return float(loss) if loss.ndim == 0 else loss


# ---------------------------------------------------------------------------
# Copula QLIK
# ---------------------------------------------------------------------------

def qlik_copula_loss(
    H_forecast: np.ndarray,
    V_realized: np.ndarray,
    margin_losses: np.ndarray | None = None,
) -> np.ndarray:
    """Copula QLIK: joint loss minus the sum of marginal losses.

    L_t^{copula} = L_t^{joint} − Σᵢ L_t^{(i)}

    Parameters
    ----------
    H_forecast:
        (T, k, k) forecast covariance matrices.
    V_realized:
        (T, k, k) realised covariance matrices.
    margin_losses:
        (T, k) precomputed marginal losses.  If *None* they are computed
        internally using :func:`qlik_margin_loss`.

    Returns
    -------
    ndarray of shape (T,)
    """
    H = np.asarray(H_forecast, dtype=float)
    V = np.asarray(V_realized, dtype=float)
    T, k, _ = H.shape

    joint = qlik_loss(H, V)

    if margin_losses is None:
        margin_losses = np.column_stack([
            qlik_margin_loss(H[:, i, i], V[:, i, i]) for i in range(k)
        ])

    return joint - np.sum(margin_losses, axis=1)


# ---------------------------------------------------------------------------
# Loss differential
# ---------------------------------------------------------------------------

def loss_differential(
    H_heavy: np.ndarray,
    H_garch: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """Per-observation QLIK differential: QLIK(HEAVY) − QLIK(GARCH).

    A negative value at observation t means HEAVY produced a lower loss at t.

    Parameters
    ----------
    H_heavy:
        (T, k, k) HEAVY conditional covariance forecasts.
    H_garch:
        (T, k, k) GARCH conditional covariance forecasts.
    V:
        (T, k, k) realised covariances used as evaluation targets.

    Returns
    -------
    ndarray of shape (T,)
        d_t = QLIK(H_heavy_t, V_t) − QLIK(H_garch_t, V_t).
    """
    return qlik_loss(H_heavy, V) - qlik_loss(H_garch, V)


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def dm_test(
    loss_diff_series: np.ndarray,
    bandwidth: int | None = None,
) -> float:
    """Diebold-Mariano HAC t-statistic for equal predictive accuracy.

    Tests H₀: E[d_t] = 0  where  d_t = L₁_t − L₂_t  (loss differential).

    The long-run variance is estimated using the Newey-West (Bartlett) HAC
    estimator via :func:`src.estimation.inference.hac_variance`.

    A negative t-statistic means model 1 has lower mean QLIK loss (wins).

    Parameters
    ----------
    loss_diff_series:
        1-D array of per-observation loss differentials d_t.
    bandwidth:
        Newey-West bandwidth.  If *None*, set to ⌊4·(T/100)^{2/9}⌋.

    Returns
    -------
    float
        DM t-statistic.
    """
    d = np.asarray(loss_diff_series, dtype=float)
    T = len(d)
    mean_d = np.mean(d)
    lrv = hac_variance(d, bandwidth)
    se = np.sqrt(max(lrv, 1e-14) / T)
    return float(mean_d / se)


# ---------------------------------------------------------------------------
# Legacy helpers (backward compatibility)
# ---------------------------------------------------------------------------

def qlik_loss_t(H_t: np.ndarray, P_t: np.ndarray) -> float:
    """QLIK loss for a single observation (legacy interface)."""
    return float(qlik_loss(H_t, P_t))


def margin_loss(H_seq: np.ndarray, P_seq: np.ndarray, i: int) -> np.ndarray:
    """Marginal QLIK for diagonal element *i* over a sequence (legacy)."""
    return qlik_margin_loss(H_seq[:, i, i], P_seq[:, i, i])


def qlik_decomposed(H_seq: np.ndarray, P_seq: np.ndarray) -> dict:
    """Decompose sequence QLIK into joint, margin, copula components."""
    T, k, _ = H_seq.shape
    joint = qlik_loss(H_seq, P_seq)
    margins = np.column_stack([margin_loss(H_seq, P_seq, i) for i in range(k)])
    copula = joint - margins.sum(axis=1)
    return {'joint': joint, 'margins': margins, 'copula': copula}


def mean_qlik_decomposed(H_seq: np.ndarray, P_seq: np.ndarray) -> dict:
    """Mean QLIK decomposition (scalar summary statistics)."""
    d = qlik_decomposed(H_seq, P_seq)
    return {
        'joint': float(np.mean(d['joint'])),
        'margins': np.mean(d['margins'], axis=0),
        'copula': float(np.mean(d['copula'])),
    }
