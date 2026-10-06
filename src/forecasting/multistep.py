"""
Vectorised multi-step-ahead forecast functions for HEAVY and GARCH models.

HEAVY multi-step forecasts (Noureldin, Shephard & Sheppard 2012):

    The recursion is jointly driven by the H and M state equations.

    M-equation propagation (s ≥ 2):
        E_T[M_{T+s}] = C_M + φ_M · E_T[M_{T+s-1}]
        where φ_M = b_M² + a_M²  and  E_T[M_{T+1}] = M_1 (given).

    H-equation propagation (s ≥ 2):
        E_T[H_{T+s}] = C_H + b_H² · E_T[H_{T+s-1}] + a_H² · E_T[M_{T+s-1}]
        with  E_T[H_{T+1}] = H_1 (given)  and  E_T[V_{T+s-1}] = E_T[M_{T+s-1}].

    Closed form equivalent for M (geometric series):
        E_T[M_{T+s}] = ω_M + φ_M^{s-1} · (M_1 − ω_M)
        where  ω_M = C_M / (1 − φ_M).

GARCH multi-step forecasts:

    E_T[H_{T+s}] = ω_G + φ_G^{s-1} · (H_1 − ω_G)
    where  φ_G = b_G² + a_G²  and  ω_G = C_G / (1 − φ_G).

    Equivalently:  Σᵢ₌₁^{s-1} (b+a)^{i-1} · C_G  +  (b+a)^{s-1} · H_1
    (where (b+a) denotes φ_G = b_G² + a_G²).
"""

from __future__ import annotations

import numpy as np

from src.models.heavy import _unpack
from src.utils.linalg import ensure_pd, symmetrize


# ---------------------------------------------------------------------------
# heavy_multistep_forecast
# ---------------------------------------------------------------------------

def heavy_multistep_forecast(
    params_H: np.ndarray,
    params_M: np.ndarray,
    H1: np.ndarray,
    M1: np.ndarray,
    s_max: int = 22,
) -> np.ndarray:
    """Vectorised multi-step HEAVY forecasts: H_{T+1|T}, …, H_{T+s_max|T}.

    Implements the recursive joint propagation of H and M described in
    Noureldin, Shephard & Sheppard (2012, eq. 2.5–2.7).

    The M-equation state is propagated using the closed-form persistence:

        E_T[M_{T+s}] = C_M + φ_M · E_T[M_{T+s-1}]   (s ≥ 2)

    where φ_M = b_M² + a_M² is the M-equation persistence.

    The H-equation state is then propagated using:

        E_T[H_{T+s}] = C_H + b_H² · E_T[H_{T+s-1}] + a_H² · E_T[M_{T+s-1}]

    Paper formula (scalar form):
        h_{T+s|T} = Σᵢ₌₁^{s-1} (b_H²)^{i-1} · c_H
                  + (b_H²)^{s-1} · h_{T+1|T}
                  + a_H² · Σᵢ₌₁^{s-1} (b_H²)^{i-1} · m_{T+s-i|T}

    Parameters
    ----------
    params_H:
        1-D array [vech(C_{L,H}), b_H, a_H].
    params_M:
        1-D array [vech(C_{L,M}), b_M, a_M].
    H1:
        (k, k) one-step-ahead H forecast H_{T+1|T}.
    M1:
        (k, k) one-step-ahead M forecast M_{T+1|T}.
    s_max:
        Number of forecast horizons to compute (≥ 1).

    Returns
    -------
    np.ndarray of shape (s_max, k, k)
        Forecast array [H_{T+1|T}, H_{T+2|T}, …, H_{T+s_max|T}].
    """
    k = H1.shape[0]
    C_H, b_H, a_H = _unpack(params_H, k)
    C_M, b_M, a_M = _unpack(params_M, k)
    beta_H = b_H ** 2      # coefficient on H_{t-1} in H recursion
    alpha_H = a_H ** 2     # coefficient on V_{t-1} in H recursion
    phi_M = b_M ** 2 + a_M ** 2   # persistence of M equation

    forecasts = np.empty((s_max, k, k))
    forecasts[0] = H1.copy()

    H_cur = H1.copy()
    M_cur = M1.copy()

    for s in range(1, s_max):
        # Propagate M: E[M_{T+s+1}] = C_M + phi_M · E[M_{T+s}]
        M_cur = C_M + phi_M * M_cur
        M_cur = ensure_pd(symmetrize(M_cur))

        # Propagate H: E[H_{T+s+1}] = C_H + beta_H · E[H_{T+s}] + alpha_H · E[M_{T+s}]
        H_cur = C_H + beta_H * H_cur + alpha_H * M_cur
        H_cur = ensure_pd(symmetrize(H_cur))

        forecasts[s] = H_cur

    return forecasts


# ---------------------------------------------------------------------------
# garch_multistep_forecast
# ---------------------------------------------------------------------------

def garch_multistep_forecast(
    params_G: np.ndarray,
    H1: np.ndarray,
    s_max: int = 22,
    omega: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised multi-step GARCH forecasts: H_{T+1|T}, …, H_{T+s_max|T}.

    Exploits the closed-form geometric mean-reversion formula:

        E_T[H_{T+s}] = ω_G + φ_G^{s-1} · (H_{T+1|T} − ω_G)

    where  φ_G = b_G² + a_G²  and  ω_G = C_G / (1 − φ_G)  is the
    unconditional covariance.

    Equivalently in expanded form:

        E_T[H_{T+s}] = Σᵢ₌₁^{s-1} φ_G^{i-1} · C_G  +  φ_G^{s-1} · H_{T+1|T}

    Parameters
    ----------
    params_G:
        1-D array [vech(C_{L,G}), b_G, a_G].
    H1:
        (k, k) one-step-ahead H forecast H_{T+1|T}.
    s_max:
        Number of forecast horizons to compute (≥ 1).
    omega:
        (k, k) precomputed unconditional covariance ω_G = C_G / (1 − φ_G).
        If *None* it is computed from *params_G*.

    Returns
    -------
    np.ndarray of shape (s_max, k, k)
        Forecast array [H_{T+1|T}, H_{T+2|T}, …, H_{T+s_max|T}].
    """
    k = H1.shape[0]
    C_G, b_G, a_G = _unpack(params_G, k)
    phi_G = b_G ** 2 + a_G ** 2   # persistence

    if omega is None:
        denom = max(1.0 - phi_G, 1e-10)
        omega = C_G / denom

    dH = H1 - omega   # deviation from unconditional mean

    forecasts = np.empty((s_max, k, k))
    for s in range(s_max):
        # s=0 → H_{T+1|T};  s=1 → H_{T+2|T};  etc.
        H_s = omega + (phi_G ** s) * dH
        forecasts[s] = ensure_pd(symmetrize(H_s))

    return forecasts
