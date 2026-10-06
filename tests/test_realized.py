"""
Tests for data generation, QLIK loss functions, decomposition identity, and DM statistic.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from data.generate_synthetic import generate_bivariate_data, generate_multiasset_data
from src.utils.linalg import is_positive_definite
from src.forecasting.qlik import (qlik_loss, qlik_loss_t, margin_loss, qlik_decomposed)
from src.estimation.inference import dm_test, hac_variance

_TEN_ASSETS = ['BAC', 'JPM', 'IBM', 'MSFT', 'XOM', 'AA', 'AXP', 'DD', 'GE', 'KO']


# ===========================================================================
# Synthetic data shape and validity
# ===========================================================================

def test_synthetic_data_shape():
    """generate_bivariate_data returns correct shapes for returns, P, V."""
    T = 50
    data = generate_bivariate_data(n_days=T, seed=0)

    assert data['returns'].shape == (T, 2), \
        f"returns shape {data['returns'].shape} != ({T}, 2)"
    assert data['P'].shape == (T, 2, 2), \
        f"P shape {data['P'].shape} != ({T}, 2, 2)"
    assert data['V'].shape == (T, 2, 2), \
        f"V shape {data['V'].shape} != ({T}, 2, 2)"
    assert len(data['dates']) == T


def test_synthetic_data_positive_definite():
    """
    V_t from synthetic data are positive definite; P_t are positive semidefinite.

    P_t = r_t r_t' is a rank-1 outer product, so it is PSD (not PD).
    V_t is a realized covariance matrix accumulated from many intraday returns,
    so it is genuinely PD (full rank).
    """
    data = generate_bivariate_data(n_days=80, seed=1)
    P, V = data['P'], data['V']

    for t in range(len(P)):
        # P_t is rank-1: all eigenvalues should be ≥ 0
        eigvals_P = np.linalg.eigvalsh(P[t])
        assert np.all(eigvals_P >= -1e-12), \
            f"P[{t}] has negative eigenvalue: {eigvals_P.min():.2e}"
        # V_t must be genuinely PD
        assert is_positive_definite(V[t]), f"V[{t}] is not PD"


def test_synthetic_ten_asset_shape():
    """generate_multiasset_data returns correct shapes for k=10."""
    T = 60
    data = generate_multiasset_data(assets=_TEN_ASSETS, n_days=T, seed=5)

    assert data['returns'].shape == (T, 10)
    assert data['P'].shape == (T, 10, 10)
    assert data['V'].shape == (T, 10, 10)


def test_synthetic_ten_asset_pd():
    """V_t from ten-asset synthetic data are PD; P_t are PSD (rank-1 outer products)."""
    data = generate_multiasset_data(assets=_TEN_ASSETS, n_days=40, seed=9)
    P, V = data['P'], data['V']

    for t in range(len(P)):
        eigvals_P = np.linalg.eigvalsh(P[t])
        assert np.all(eigvals_P >= -1e-12), \
            f"P[{t}] has negative eigenvalue: {eigvals_P.min():.2e} (ten-asset)"
        assert is_positive_definite(V[t]), f"V[{t}] is not PD (ten-asset)"


# ===========================================================================
# Realized correlation bounds
# ===========================================================================

def test_realized_correlation_bounds():
    """All realized correlations rho_12,t lie in (-1, 1)."""
    data = generate_bivariate_data(n_days=200, seed=2)
    V = data['V']
    rho = V[:, 0, 1] / np.sqrt(V[:, 0, 0] * V[:, 1, 1])

    assert np.all(rho >= -1.0), f"Some rho < -1: min={rho.min():.4f}"
    assert np.all(rho <= 1.0), f"Some rho > +1: max={rho.max():.4f}"


def test_realized_correlation_not_degenerate():
    """Realized correlations are not all exactly 0 or ±1."""
    data = generate_bivariate_data(n_days=200, seed=3)
    V = data['V']
    rho = V[:, 0, 1] / np.sqrt(V[:, 0, 0] * V[:, 1, 1])

    assert np.std(rho) > 1e-4, "Realized correlations have near-zero variance (degenerate)"
    assert np.abs(np.mean(rho)) < 0.99, "Mean realized correlation is degenerate (±1)"


# ===========================================================================
# QLIK loss positivity
# ===========================================================================

def test_qlik_loss_positive():
    """
    QLIK loss L_t = log|H_t| + tr(H_t^{-1} P_t) is finite for PD H_t.

    Note: P_t = r_t r_t' is a rank-1 outer product (PSD, not PD), so P cannot
    be used as H.  Instead we use V (realized covariance, genuinely PD) as H_t.
    """
    data = generate_bivariate_data(n_days=100, seed=4)
    P = data['P']
    V = data['V']   # PD realized covariance matrices

    # V is PD, P is the target (outer products)
    losses = qlik_loss(V, P)
    assert np.all(np.isfinite(losses)), "QLIK(V, P) has non-finite values"

    # For constant identity H_t = I, loss = log|I| + tr(I^{-1} P) = tr(P)
    # P_t = r_t r_t' has tr(P_t) = |r_t|^2 > 0, so all losses should be positive
    k = 2
    H_const = np.tile(np.eye(k)[np.newaxis], (len(P), 1, 1))
    losses2 = qlik_loss(H_const, P)
    assert np.all(losses2 > 0), "QLIK(I, P) should be positive since tr(P_t) = |r_t|^2 > 0"


def test_qlik_loss_t_known():
    """QLIK loss for H=I, P=I equals 0 + k = k."""
    k = 3
    H = np.eye(k)
    P = np.eye(k)
    loss = qlik_loss_t(H, P)
    expected = float(k)   # log|I| + tr(I) = 0 + k
    assert abs(loss - expected) < 1e-12, f"qlik_loss_t(I, I) = {loss:.6f} != {expected}"


# ===========================================================================
# Loss decomposition identity
# ===========================================================================

def test_loss_decomposition_bivariate():
    """
    Joint QLIK = sum of margin QLIKs + copula QLIK (to numerical precision).

    Tolerance: 1e-10.
    """
    data = generate_bivariate_data(n_days=150, seed=10)
    P = data['P']
    # Use a trivially known H (constant diagonal)
    k = 2
    H_seq = np.tile(np.eye(k)[np.newaxis], (len(P), 1, 1))

    dec = qlik_decomposed(H_seq, P)
    joint = dec['joint']
    margins = dec['margins']
    copula = dec['copula']

    residual = joint - margins.sum(axis=1) - copula
    assert np.all(np.abs(residual) < 1e-10), \
        f"Decomposition error: max|residual| = {np.max(np.abs(residual)):.2e}"


def test_loss_decomposition_random_H():
    """Decomposition holds for random PD H_t matrices."""
    rng = np.random.default_rng(77)
    T, k = 80, 2

    # Build random PD H_t matrices
    H_seq = np.empty((T, k, k))
    for t in range(T):
        B = rng.standard_normal((k, k))
        H_seq[t] = B @ B.T + np.eye(k)

    data = generate_bivariate_data(n_days=T, seed=11)
    P = data['P']

    dec = qlik_decomposed(H_seq, P)
    residual = dec['joint'] - dec['margins'].sum(axis=1) - dec['copula']
    assert np.all(np.abs(residual) < 1e-10), \
        f"Decomposition fails for random H: max|residual| = {np.max(np.abs(residual)):.2e}"


# ===========================================================================
# DM statistic
# ===========================================================================

def test_dm_statistic_finite():
    """DM statistic is finite for well-behaved loss differentials."""
    rng = np.random.default_rng(99)
    loss_diff = rng.standard_normal(200)
    t_stat = dm_test(loss_diff)
    assert np.isfinite(t_stat), f"DM statistic is not finite: {t_stat}"


def test_dm_statistic_zero_mean():
    """DM t-stat is close to 0 for a zero-mean series (large sample)."""
    rng = np.random.default_rng(123)
    # Large sample of i.i.d. zero-mean series: |t| should be small
    loss_diff = rng.standard_normal(2000)
    t_stat = dm_test(loss_diff)
    assert abs(t_stat) < 3.0, \
        f"|DM t-stat| = {abs(t_stat):.2f} too large for zero-mean i.i.d. data"


def test_dm_statistic_detects_large_mean():
    """DM t-stat should be clearly negative when HEAVY consistently beats GARCH."""
    rng = np.random.default_rng(321)
    # HEAVY consistently has lower loss (negative mean differential)
    loss_diff = rng.standard_normal(500) - 2.0   # mean = -2
    t_stat = dm_test(loss_diff)
    assert t_stat < -1.0, \
        f"Expected clearly negative t-stat for negative-mean diff, got {t_stat:.4f}"


def test_hac_variance_positive():
    """HAC variance estimate is positive for non-degenerate input."""
    rng = np.random.default_rng(66)
    x = rng.standard_normal(300)
    hac_var = hac_variance(x)
    assert hac_var > 0, f"HAC variance should be positive, got {hac_var:.6f}"


def test_hac_variance_decreases_with_smoothing():
    """
    HAC variance with larger bandwidth accounts for more autocorrelation.
    For white noise the estimate should be stable (within a factor of 2).
    """
    rng = np.random.default_rng(44)
    x = rng.standard_normal(500)
    hac1 = hac_variance(x, bandwidth=1)
    hac10 = hac_variance(x, bandwidth=10)
    # Both should be close to population variance (~1) for i.i.d. data
    assert 0.5 < hac1 < 2.0, f"hac_var(bw=1)={hac1:.4f} outside (0.5, 2.0)"
    assert 0.5 < hac10 < 2.0, f"hac_var(bw=10)={hac10:.4f} outside (0.5, 2.0)"
