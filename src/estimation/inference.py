"""
Statistical inference utilities for QMLE-estimated volatility models.

Public API
----------
sandwich_cov(scores, hessian=None)
    Sandwich covariance matrix A⁻¹ B A⁻¹ from a (T, p) score matrix.

robust_se(params, score_fn, param_names=None)
    Convenience wrapper: call *score_fn* to get scores, compute sandwich SE.

hac_cov(series, bandwidth=None, kernel='bartlett')
    Long-run HAC covariance matrix using arch's Bartlett / Newey-West kernel.

hac_variance(x, bandwidth=None)
    Scalar HAC long-run variance (Newey-West), kept for compatibility.

dm_test(loss_diff, bandwidth=None)
    Diebold-Mariano (1995) t-statistic for equal predictive accuracy.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# sandwich_cov
# ---------------------------------------------------------------------------

def sandwich_cov(
    scores: np.ndarray,
    hessian: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the sandwich (robust) covariance matrix from QMLE scores.

    Given the (T × p) matrix of per-observation gradient contributions and an
    optional (p × p) Hessian of the total negative QMLE objective, returns:

        sandwich_cov = A⁻¹ B A⁻¹

    where  B = scores.T @ scores  (total outer-product of gradients)
    and    A = hessian  (if supplied) else the identity (pure B matrix).

    Parameters
    ----------
    scores:
        (T, p) array; row *t* is the gradient of l_t w.r.t. theta.
    hessian:
        (p, p) numerical Hessian of Σ_t Q_t (negative QMLE) at theta_hat.
        If *None*, the function returns B = scores.T @ scores.

    Returns
    -------
    np.ndarray of shape (p, p)
        Sandwich covariance (or B if *hessian* is None).
    """
    scores = np.asarray(scores, dtype=float)
    B = scores.T @ scores
    if hessian is None:
        return B
    A = np.asarray(hessian, dtype=float)
    try:
        A_inv = np.linalg.inv(A)
        return A_inv @ B @ A_inv
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
        return A_inv @ B @ A_inv


# ---------------------------------------------------------------------------
# robust_se
# ---------------------------------------------------------------------------

def robust_se(
    params: np.ndarray,
    score_fn,
    param_names: list[str] | None = None,
) -> np.ndarray | dict:
    """Compute robust standard errors using a user-supplied score function.

    Parameters
    ----------
    params:
        1-D parameter vector at which scores are evaluated.
    score_fn:
        Callable ``score_fn(params) -> np.ndarray of shape (T, p)``
        returning the matrix of per-observation score vectors.
    param_names:
        Optional list of parameter names.  If provided the function returns
        a dict ``{name: se}`` instead of a plain array.

    Returns
    -------
    ndarray of shape (p,), or dict {name: float} if *param_names* given.
        Robust standard errors.
    """
    scores = np.asarray(score_fn(params), dtype=float)
    B = scores.T @ scores
    se = np.sqrt(np.maximum(np.diag(B), 0.0))
    if param_names is not None:
        return dict(zip(param_names, se))
    return se


# ---------------------------------------------------------------------------
# hac_cov
# ---------------------------------------------------------------------------

def hac_cov(
    series: np.ndarray,
    bandwidth: float | None = None,
    kernel: str = 'bartlett',
) -> np.ndarray:
    """Long-run HAC covariance matrix via the arch Bartlett (Newey-West) kernel.

    For a 1-D input the result is a (1, 1) matrix; for a (T, p) matrix the
    result is (p, p).

    Parameters
    ----------
    series:
        1-D array of length T or (T, p) array.
    bandwidth:
        Kernel bandwidth h.  If *None*, the optimal bandwidth is selected
        automatically by :class:`arch.covariance.kernel.Bartlett`.
    kernel:
        Only ``'bartlett'`` is currently supported (arch implementation).

    Returns
    -------
    np.ndarray
        Long-run covariance matrix (shape (p, p)).
    """
    from arch.covariance.kernel import Bartlett

    series = np.asarray(series, dtype=float)
    if series.ndim == 1:
        series = series[:, None]

    if kernel.lower() != 'bartlett':
        raise NotImplementedError(
            f"Kernel '{kernel}' is not implemented; use 'bartlett'.")

    est = Bartlett(series, bandwidth=bandwidth, center=True)
    return np.asarray(est.cov.long_run)


# ---------------------------------------------------------------------------
# Scalar HAC variance (legacy / Diebold-Mariano)
# ---------------------------------------------------------------------------

def hac_variance(x: np.ndarray, bandwidth: int | None = None) -> float:
    """Newey-West HAC long-run variance for a scalar series.

    Parameters
    ----------
    x:
        1-D array of length T.
    bandwidth:
        Bartlett kernel bandwidth.  If *None*, set to ⌊4·(T/100)^{2/9}⌋.

    Returns
    -------
    float
        HAC long-run variance estimate (not divided by T).
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if bandwidth is None:
        bandwidth = max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))

    e = x - np.mean(x)
    gamma0 = np.dot(e, e) / T
    result = gamma0
    for j in range(1, bandwidth + 1):
        w = 1.0 - j / (bandwidth + 1.0)
        result += 2.0 * w * np.dot(e[j:], e[:-j]) / T
    return float(result)


def dm_test(loss_diff: np.ndarray, bandwidth: int | None = None) -> float:
    """Diebold-Mariano (1995) test statistic for equal predictive accuracy.

    Tests H₀: E[d_t] = 0  where d_t = L₁_t − L₂_t (loss differential).
    A negative t-statistic means model 1 has lower mean loss (model 1 wins).

    Standard error is estimated via :func:`hac_variance`.

    Parameters
    ----------
    loss_diff:
        1-D array of loss differentials D_t = L1_t − L2_t.
    bandwidth:
        Newey-West bandwidth (None = automatic).

    Returns
    -------
    float
        t-statistic.
    """
    loss_diff = np.asarray(loss_diff, dtype=float)
    T = len(loss_diff)
    mean_d = np.mean(loss_diff)
    lrv = hac_variance(loss_diff, bandwidth)
    se = np.sqrt(max(lrv, 1e-14) / T)
    return float(mean_d / se)
