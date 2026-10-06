"""
Tests for src/models/ and src/forecasting/multistep.py.

All tests use small synthetic data (T=100, k=2) for speed.

API reference (actual model signatures):
  ScalarHEAVY.fit(P, V)          -> sets params_H_, params_M_, H_seq_, M_seq_,
                                     loglik_H_, loglik_M_
  ScalarHEAVY.fit_H(P, V)        -> (params_H, nll_H, H_seq)
  ScalarHEAVY.loglik_H(params, P, V, H0) -> float (negative QMLE, minimised)
  ScalarHEAVY.compute_H_sequence(params_H, V, H0) -> (T,k,k)
  ScalarHEAVY.compute_M_sequence(params_M, V, M0) -> (T,k,k)
  params format: [vech(C_L), b_raw, a_raw]  (len = k*(k+1)//2 + 2)
  recursion coeff: b_raw**2 and a_raw**2

  ScalarGARCH.fit(P)              -> sets params_, H_seq_, loglik_
  ScalarGARCH.compute_H_sequence(params, P, H0) -> (T,k,k)
  ScalarGARCH.loglik_H(params, P, H0)  -> float (negative QMLE)

  ScalarGARCHX.fit(P, V)          -> sets params_, H_seq_, loglik_
  ScalarGARCHX.compute_H_sequence(params, V, P, H0) -> (T,k,k)
  params format: [vech(C_L), b_raw, a_raw, d_raw]  (len = k*(k+1)//2 + 3)

  heavy_multistep_forecast(params_H, params_M, H1, M1, s_max) -> (s_max,k,k)
  garch_multistep_forecast(params_G, H1, s_max, omega=None)   -> (s_max,k,k)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from data.generate_synthetic import generate_bivariate_data
from src.models.heavy import ScalarHEAVY
from src.models.garch import ScalarGARCH
from src.models.garch_x import ScalarGARCHX
from src.utils.linalg import is_positive_definite
from src.forecasting.multistep import heavy_multistep_forecast, garch_multistep_forecast

# ---- Shared synthetic data (T=100, k=2) ----
_DATA = generate_bivariate_data(n_days=100, seed=7)
_P = _DATA['P']
_V = _DATA['V']
_k = 2
_NV = _k * (_k + 1) // 2   # = 3 for k=2


# ===========================================================================
# ScalarHEAVY tests
# ===========================================================================

def test_heavy_recursion_shape():
    """HEAVY H sequence has shape (T, k, k) after fitting."""
    model = ScalarHEAVY(_k)
    model.fit(_P, _V)
    T = len(_V)
    assert model.H_seq_.shape == (T, _k, _k), \
        f"Expected ({T},{_k},{_k}), got {model.H_seq_.shape}"
    assert model.M_seq_.shape == (T, _k, _k), \
        f"Expected ({T},{_k},{_k}), got {model.M_seq_.shape}"


def test_heavy_pd():
    """All H_t from fitted HEAVY are positive definite."""
    model = ScalarHEAVY(_k)
    model.fit(_P, _V)
    for t in range(len(model.H_seq_)):
        assert is_positive_definite(model.H_seq_[t]), \
            f"H[{t}] is not PD:\n{model.H_seq_[t]}"


def test_heavy_loglik_finite():
    """HEAVY negative QMLE is finite after fitting."""
    model = ScalarHEAVY(_k)
    model.fit(_P, _V)
    assert np.isfinite(model.loglik_H_), \
        f"HEAVY loglik_H_ is not finite: {model.loglik_H_}"


def test_heavy_loglik_returns_large_for_bad_params():
    """HEAVY loglik_H returns 1e18 when b_raw + a_raw >= 1 (non-stationary)."""
    model = ScalarHEAVY(_k)
    H0 = np.mean(_P, axis=0)
    # Build params: [vech(C_L) (3 values), b_raw, a_raw] with b+a > 1
    bad_params = np.ones(_NV + 2) * 0.01
    bad_params[_NV] = 0.70      # b_raw
    bad_params[_NV + 1] = 0.31  # a_raw; 0.70 + 0.31 = 1.01 > 1
    nll = model.loglik_H(bad_params, _P, _V, H0)
    assert nll >= 1e10, f"Expected large value for non-stationary params, got {nll}"


def test_heavy_fit_runs():
    """ScalarHEAVY.fit() completes and sets params_H_, params_M_, H_seq_, M_seq_."""
    model = ScalarHEAVY(_k)
    model.fit(_P, _V)
    assert model.params_H_ is not None
    assert model.params_M_ is not None
    assert len(model.params_H_) == _NV + 2, \
        f"params_H_ length {len(model.params_H_)} != {_NV + 2}"
    assert len(model.params_M_) == _NV + 2, \
        f"params_M_ length {len(model.params_M_)} != {_NV + 2}"
    assert model.H_seq_ is not None
    assert model.M_seq_ is not None


# ===========================================================================
# ScalarGARCH tests
# ===========================================================================

def test_garch_recursion_shape():
    """GARCH H sequence has shape (T, k, k) after fitting."""
    model = ScalarGARCH(_k)
    model.fit(_P)
    T = len(_P)
    assert model.H_seq_.shape == (T, _k, _k), \
        f"Expected ({T},{_k},{_k}), got {model.H_seq_.shape}"


def test_garch_pd():
    """All H_t from fitted GARCH are positive definite."""
    model = ScalarGARCH(_k)
    model.fit(_P)
    for t in range(len(model.H_seq_)):
        assert is_positive_definite(model.H_seq_[t]), \
            f"H[{t}] is not PD:\n{model.H_seq_[t]}"


def test_garch_stationarity():
    """Fitted GARCH: b_raw² + a_raw² < 1 (covariance stationarity)."""
    model = ScalarGARCH(_k)
    model.fit(_P)
    b_raw = model.params_[_NV]
    a_raw = model.params_[_NV + 1]
    persistence = b_raw ** 2 + a_raw ** 2
    assert persistence < 1.0, \
        f"b² + a² = {persistence:.4f} >= 1 (non-stationary)"


def test_garch_fit_loglik_finite():
    """GARCH fitted negative QMLE is finite."""
    model = ScalarGARCH(_k)
    model.fit(_P)
    assert np.isfinite(model.loglik_), \
        f"GARCH loglik_ is not finite: {model.loglik_}"


# ===========================================================================
# ScalarGARCHX tests
# ===========================================================================

def test_garch_x_recursion_shape():
    """GARCH-X H sequence has shape (T, k, k) after fitting."""
    model = ScalarGARCHX(_k)
    model.fit(_P, _V)
    T = len(_P)
    assert model.H_seq_.shape == (T, _k, _k), \
        f"Expected ({T},{_k},{_k}), got {model.H_seq_.shape}"


def test_garch_x_fit_runs():
    """ScalarGARCHX.fit() completes and sets params_ and H_seq_."""
    model = ScalarGARCHX(_k)
    model.fit(_P, _V)
    assert model.params_ is not None
    assert len(model.params_) == _NV + 3, \
        f"GARCH-X params length {len(model.params_)} != {_NV + 3}"
    assert model.H_seq_ is not None
    assert np.isfinite(model.loglik_)


def test_garch_x_uses_both_P_and_V():
    """Fitted GARCH-X produces a different H sequence than fitted GARCH."""
    model_gx = ScalarGARCHX(_k)
    model_gx.fit(_P, _V)

    model_g = ScalarGARCH(_k)
    model_g.fit(_P)

    # GARCH-X explicitly uses V in the recursion; sequences should differ
    assert not np.allclose(model_gx.H_seq_, model_g.H_seq_), \
        "GARCH-X and GARCH H sequences should differ (GARCH-X uses V)"


def test_garch_x_compute_sequence_shape():
    """compute_H_sequence for GARCH-X returns (T, k, k)."""
    model = ScalarGARCHX(_k)
    model.fit(_P, _V)
    H0 = np.mean(_P, axis=0)
    # compute_H_sequence signature: (params, V_array, P_array, H0)
    H_seq = model.compute_H_sequence(model.params_, _V, _P, H0)
    assert H_seq.shape == (len(_P), _k, _k)


# ===========================================================================
# Qualitative comparison
# ===========================================================================

def test_heavy_loglik_not_far_below_garch():
    """
    On synthetic data, HEAVY negative QMLE should not be catastrophically
    above GARCH (both are negative QMLE; lower = better fit).

    HEAVY nests more information so it should not be drastically worse;
    on small samples the ordering is not guaranteed, but the gap should
    be within 50 units of the objective.
    """
    data = generate_bivariate_data(n_days=200, seed=42)
    P, V = data['P'], data['V']

    heavy = ScalarHEAVY(_k)
    heavy.fit(P, V)

    garch = ScalarGARCH(_k)
    garch.fit(P)

    # heavy.loglik_H_ and garch.loglik_ are both the negative QMLE
    # (lower = better).  Allow HEAVY to be at most 50 units worse.
    diff = heavy.loglik_H_ - garch.loglik_
    assert diff < 50.0, \
        f"HEAVY neg-loglik ({heavy.loglik_H_:.1f}) far above GARCH ({garch.loglik_:.1f})"


# ===========================================================================
# Multi-step forecast tests
# ===========================================================================

def test_heavy_multistep_forecast_shape():
    """heavy_multistep_forecast returns (s_max, k, k) array."""
    model = ScalarHEAVY(_k)
    model.fit(_P, _V)
    H1 = model.H_seq_[-1]
    M1 = model.M_seq_[-1]
    s_max = 10
    forecasts = heavy_multistep_forecast(model.params_H_, model.params_M_, H1, M1,
                                         s_max=s_max)
    assert forecasts.shape == (s_max, _k, _k), \
        f"Expected ({s_max},{_k},{_k}), got {forecasts.shape}"


def test_heavy_multistep_s1_returns_H1():
    """heavy_multistep_forecast with s_max=1 returns H1 as the first (only) entry."""
    model = ScalarHEAVY(_k)
    model.fit(_P, _V)
    H1 = model.H_seq_[-1].copy()
    M1 = model.M_seq_[-1].copy()

    forecasts = heavy_multistep_forecast(model.params_H_, model.params_M_, H1, M1, s_max=1)
    assert np.allclose(forecasts[0], H1, atol=1e-14), \
        "s_max=1 forecast[0] should equal the supplied H1"


def test_multistep_forecast_convergence():
    """
    For s -> inf, HEAVY forecast converges: 50-step and 100-step should agree.
    """
    model = ScalarHEAVY(_k)
    model.fit(_P, _V)
    H1 = model.H_seq_[-1]
    M1 = model.M_seq_[-1]

    forecasts = heavy_multistep_forecast(model.params_H_, model.params_M_, H1, M1,
                                         s_max=100)
    H50 = forecasts[49]
    H100 = forecasts[99]

    assert np.allclose(H50, H100, atol=1e-4), \
        f"HEAVY forecast has not converged by step 50 vs 100:\n{H50}\nvs\n{H100}"


def test_garch_multistep_forecast_shape():
    """garch_multistep_forecast returns (s_max, k, k) array."""
    model = ScalarGARCH(_k)
    model.fit(_P)
    H1 = model.H_seq_[-1]
    s_max = 10
    forecasts = garch_multistep_forecast(model.params_, H1, s_max=s_max)
    assert forecasts.shape == (s_max, _k, _k), \
        f"Expected ({s_max},{_k},{_k}), got {forecasts.shape}"


def test_garch_multistep_s1_returns_H1():
    """garch_multistep_forecast with s_max=1 returns H1 as the first entry."""
    model = ScalarGARCH(_k)
    model.fit(_P)
    H1 = model.H_seq_[-1].copy()

    forecasts = garch_multistep_forecast(model.params_, H1, s_max=1)
    assert np.allclose(forecasts[0], H1, atol=1e-14), \
        "s_max=1 forecast[0] should equal the supplied H1"


def test_garch_multistep_convergence():
    """GARCH s-step forecast converges: 50-step and 100-step should agree."""
    model = ScalarGARCH(_k)
    model.fit(_P)
    H1 = model.H_seq_[-1]

    forecasts = garch_multistep_forecast(model.params_, H1, s_max=100)
    H50 = forecasts[49]
    H100 = forecasts[99]

    assert np.allclose(H50, H100, atol=1e-4), \
        f"GARCH forecast has not converged by step 50 vs 100:\n{H50}\nvs\n{H100}"
