"""Synthetic bivariate and multi-asset data generator.

Produces daily log-returns and realized covariance proxies that mimic
SPY-BAC (and general k-asset) dynamics for the 2001-2009 period used in
Shephard & Sheppard (2010).

Data contract
-------------
Every function returns a ``dict`` with keys:

* ``'returns'`` - (T, k) float array of daily log-returns (decimal units).
* ``'P'``       - (T, k, k) outer-product matrices r_t r_t'.
* ``'V'``       - (T, k, k) realized covariance matrices (Wishart-noisy H_t).
* ``'dates'``   - pd.DatetimeIndex of T business days starting 2001-02-01.

Generation mechanism
--------------------
1. GARCH(1,1) marginal variances for each asset, calibrated so that
   annualised unconditional vol is ~15 % (SPY) and ~30 % (BAC).
2. Fisher-z AR(1) correlation process (bivariate) or scalar DCC(1,1)
   (multi-asset) for time-varying correlations.
3. Returns r_t ~ N(0, H_t) where H_t = D_t R_t D_t.
4. Realized covariance V_t ~ Wishart(H_t / nu, nu) so that
   E[V_t | H_t] = H_t  (unbiased, nu = 20 d.f.).

Calibration (bivariate SPY-BAC)
--------------------------------
  Asset | GARCH omega | GARCH alpha | GARCH beta | Ann. vol
  SPY   | 9.0e-7      | 0.07        | 0.92       | ~15 %
  BAC   | 7.0e-6      | 0.10        | 0.88       | ~30 %

Mean SPY-BAC correlation = 0.50, AR coeff phi = 0.97, Wishart nu = 20.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default calibration constants
# ---------------------------------------------------------------------------

# GARCH(1,1): unconditional daily variance = omega / (1 - alpha - beta).
# SPY: 9e-7 / 0.01 = 9e-5  -> daily vol ~ 0.95 % -> annual ~ 15 %
# BAC: 7e-6 / 0.02 = 3.5e-4 -> daily vol ~ 1.87 % -> annual ~ 30 %
_SPY_OMEGA: float = 9.0e-7
_SPY_ALPHA: float = 0.07
_SPY_BETA: float = 0.92

_BAC_OMEGA: float = 7.0e-6
_BAC_ALPHA: float = 0.10
_BAC_BETA: float = 0.88

_RHO_BAR: float = 0.50    # long-run SPY-BAC correlation
_PHI_RHO: float = 0.97    # AR coefficient for Fisher-z correlation process
_SIGMA_RHO: float = 0.02  # innovation std in Fisher-z space
_WISHART_NU: int = 20     # Wishart degrees-of-freedom for V_t noise

_START_DATE: str = "2001-02-01"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wishart_sample(H: np.ndarray, nu: int, rng: np.random.Generator) -> np.ndarray:
    """Draw one Wishart(H/nu, nu) sample with expectation H.

    Generates nu i.i.d. rows from N(0, H) via Cholesky, forming the
    (1/nu)-scaled outer-product sum so that E[result] = H.

    Parameters
    ----------
    H:
        (k, k) positive-definite scale matrix.
    nu:
        Degrees of freedom (nu > k required for almost-sure PD output).
    rng:
        Seeded numpy random Generator.

    Returns
    -------
    np.ndarray
        (k, k) symmetric PSD matrix with E[result] = H.
    """
    k = H.shape[0]
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(H)
        L = V @ np.diag(np.sqrt(np.clip(w, 1e-10, None)))
    Z = rng.standard_normal((nu, k))  # nu x k standard normals
    W = Z @ L.T                       # nu x k, each row ~ N(0, H)
    return (W.T @ W) / nu             # k x k, E = H


def _ensure_pd(H: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetrize H and clip eigenvalues to at least eps."""
    H = (H + H.T) * 0.5
    w, V = np.linalg.eigh(H)
    return V @ np.diag(np.clip(w, eps, None)) @ V.T


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_bivariate_data(n_days: int, seed: int = 42) -> dict:
    """Generate synthetic bivariate SPY-BAC daily data.

    Uses GARCH(1,1) marginal variance processes and a Fisher-z AR(1)
    correlation process.  Realized covariance V_t is Wishart-noisy.

    Parameters
    ----------
    n_days:
        Number of trading days T to simulate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``'returns'`` (T, 2), ``'P'`` (T, 2, 2), ``'V'`` (T, 2, 2),
        ``'dates'`` (pd.DatetimeIndex of length T).
    """
    rng = np.random.default_rng(seed)
    k = 2

    omega = np.array([_SPY_OMEGA, _BAC_OMEGA])
    alpha = np.array([_SPY_ALPHA, _BAC_ALPHA])
    beta = np.array([_SPY_BETA, _BAC_BETA])

    h = omega / (1.0 - alpha - beta)   # unconditional daily variances

    z_rho = np.arctanh(_RHO_BAR)       # Fisher-z correlation state
    z_bar = z_rho

    returns = np.empty((n_days, k), dtype=float)
    P = np.empty((n_days, k, k), dtype=float)
    V = np.empty((n_days, k, k), dtype=float)

    for t in range(n_days):
        rho = np.tanh(z_rho)
        corr = np.array([[1.0, rho], [rho, 1.0]])
        std = np.sqrt(h)
        H_t = _ensure_pd(np.outer(std, std) * corr)

        L = np.linalg.cholesky(H_t)
        r_t = L @ rng.standard_normal(k)
        returns[t] = r_t
        P[t] = np.outer(r_t, r_t)

        V[t] = _wishart_sample(H_t, _WISHART_NU, rng)

        h = np.clip(omega + alpha * r_t ** 2 + beta * h, 1e-12, None)
        z_rho = ((1.0 - _PHI_RHO) * z_bar
                 + _PHI_RHO * z_rho
                 + _SIGMA_RHO * rng.standard_normal())

    dates = pd.bdate_range(start=_START_DATE, periods=n_days)
    return {"returns": returns, "P": P, "V": V, "dates": dates}


def generate_multiasset_data(assets: list, n_days: int, seed: int = 42) -> dict:
    """Generate synthetic k-asset daily data via scalar DCC-GARCH(1,1).

    GARCH parameters are linearly interpolated between SPY and BAC calibration
    points.  Correlations follow a scalar DCC(1,1) model with a compound-
    symmetric long-run correlation structure (off-diagonal 0.35).

    Parameters
    ----------
    assets:
        List of k asset names (for labelling; not validated against any API).
    n_days:
        Number of trading days T to simulate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``'returns'`` (T, k), ``'P'`` (T, k, k), ``'V'`` (T, k, k),
        ``'dates'`` (pd.DatetimeIndex of length T).
    """
    rng = np.random.default_rng(seed)
    k = len(assets)

    omega = np.linspace(_SPY_OMEGA, _BAC_OMEGA, k)
    alpha = np.linspace(_SPY_ALPHA, _BAC_ALPHA, k)
    beta = np.linspace(_SPY_BETA, _BAC_BETA, k)

    h = omega / (1.0 - alpha - beta)

    rho_off = 0.35
    Qbar = rho_off * np.ones((k, k)) + (1.0 - rho_off) * np.eye(k)

    dcc_a: float = 0.04
    dcc_b: float = 0.95
    Q = Qbar.copy()

    returns = np.empty((n_days, k), dtype=float)
    P = np.empty((n_days, k, k), dtype=float)
    V = np.empty((n_days, k, k), dtype=float)

    for t in range(n_days):
        d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Q), 1e-12))
        R_t = _ensure_pd(np.outer(d_inv, d_inv) * Q)
        np.fill_diagonal(R_t, 1.0)

        std = np.sqrt(h)
        H_t = _ensure_pd(np.outer(std, std) * R_t)

        L = np.linalg.cholesky(H_t)
        r_t = L @ rng.standard_normal(k)
        returns[t] = r_t
        P[t] = np.outer(r_t, r_t)

        V[t] = _wishart_sample(H_t, _WISHART_NU, rng)

        h = np.clip(omega + alpha * r_t ** 2 + beta * h, 1e-12, None)

        z_t = r_t / np.sqrt(h)
        Q = ((1.0 - dcc_a - dcc_b) * Qbar
             + dcc_a * np.outer(z_t, z_t)
             + dcc_b * Q)
        Q = (Q + Q.T) * 0.5

    dates = pd.bdate_range(start=_START_DATE, periods=n_days)
    return {"returns": returns, "P": P, "V": V, "dates": dates}


def get_canonical_bivariate(seed: int = 42) -> dict:
    """Return the canonical ~2242-day bivariate SPY-BAC synthetic dataset.

    Covers business days from 2001-02-01 to 2009-12-31.

    Parameters
    ----------
    seed:
        Random seed.  Default 42.

    Returns
    -------
    dict
        Same structure as :func:`generate_bivariate_data`.
    """
    n_days = len(pd.bdate_range(start="2001-02-01", end="2009-12-31"))
    return generate_bivariate_data(n_days=n_days, seed=seed)


if __name__ == "__main__":
    data = get_canonical_bivariate()
    r = data["returns"]
    V_arr = data["V"]
    ann_vol = r.std(axis=0) * (252 ** 0.5) * 100
    print(f"Generated {len(data['dates'])} trading days "
          f"({data['dates'][0].date()} to {data['dates'][-1].date()})")
    print(f"  SPY annualised vol : {ann_vol[0]:.1f} %")
    print(f"  BAC annualised vol : {ann_vol[1]:.1f} %")
    print(f"  SPY-BAC correlation: {np.corrcoef(r.T)[0, 1]:.3f}")
    print(f"  Mean V_t[0,0]      : {V_arr[:, 0, 0].mean():.2e}")
    print(f"  Mean V_t[1,1]      : {V_arr[:, 1, 1].mean():.2e}")
