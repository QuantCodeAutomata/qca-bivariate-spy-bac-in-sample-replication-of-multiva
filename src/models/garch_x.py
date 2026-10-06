"""
Scalar Multivariate GARCH-X Model.

Recursion:
    H_t = C_GX + b_GX² H_{t-1} + a_GX² V_{t-1} + d_GX² P_{t-1}

Both the lagged realised covariance V_{t-1} and the lagged outer product
P_{t-1} = R_{t-1} R_{t-1}ᵀ enter.  The intercept C_GX = C_{L,GX} C_{L,GX}ᵀ
is parameterised via its lower-triangular Cholesky factor.

QMLE:  l_{GX,t} = -½ (log|H_t| + tr(H_t⁻¹ P_t))

Stationarity constraint:  b_GX + a_GX < 1  (d_GX is an additional regressor
with no separate stationarity requirement beyond positivity).

Multi-step forecast:
    When params_M (HEAVY-V parameters) are provided, the M-equation is used to
    propagate E_T[V_{T+s-1}].  Otherwise, V is held fixed at M_T.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import minimize

from src.models.heavy import (
    _neg_loglik_kernel,
    _init_cov,
    _make_init_params,
    _unpack,
    ensure_pd_numba,
    logdet_numba,
    trace_prod_numba,
)
from src.utils.linalg import (
    ensure_pd,
    safe_inv,
    safe_logdet,
    symmetrize,
    trace_prod,
)
from src.utils.matrix import lower_chol_from_vech, vech, vech_size


# ---------------------------------------------------------------------------
# Numba kernel for the GARCH-X recursion
# ---------------------------------------------------------------------------

@njit(cache=True)
def _garchx_recursion(C: np.ndarray, b2: float, a2: float, d2: float,
                      V_array: np.ndarray, P_array: np.ndarray,
                      init: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Numba-compiled GARCH-X recursion.

    H_t = C + b2·H_{t-1} + a2·V_{t-1} + d2·P_{t-1}

    Pre-sample (t=0): both V_0 and P_0 are approximated by *init*.

    Parameters
    ----------
    C:
        (k, k) SPD intercept.
    b2, a2, d2:
        Scalar coefficients (= b_GX², a_GX², d_GX²).
    V_array:
        (T, k, k) realised covariances.
    P_array:
        (T, k, k) outer products of returns.
    init:
        (k, k) initial condition H_0 (also used as V_0 and P_0).
    epsilon:
        Minimum eigenvalue floor.

    Returns
    -------
    np.ndarray of shape (T, k, k)
    """
    T = V_array.shape[0]
    k = V_array.shape[1]
    H = np.empty((T, k, k))
    H_prev = init.copy()
    for t in range(T):
        V_prev = init if t == 0 else V_array[t - 1]
        P_prev = init if t == 0 else P_array[t - 1]
        H_t = C + b2 * H_prev + a2 * V_prev + d2 * P_prev
        H_t = ensure_pd_numba(H_t, epsilon)
        H[t] = H_t
        H_prev = H_t
    return H


def _unpack_gx(params: np.ndarray, k: int):
    """Unpack GARCH-X parameter vector (vech(C_L), b, a, d)."""
    nv = vech_size(k)
    C_L = lower_chol_from_vech(params[:nv], k)
    C = C_L @ C_L.T
    b = float(params[nv])
    a = float(params[nv + 1])
    d = float(params[nv + 2])
    return C, b, a, d


# ---------------------------------------------------------------------------
# ScalarGARCHX class
# ---------------------------------------------------------------------------

class ScalarGARCHX:
    """Scalar Multivariate GARCH-X model.

    Recursion: H_t = C_GX + b_GX² H_{t-1} + a_GX² V_{t-1} + d_GX² P_{t-1}

    Parameters are stored as theta_GX = (vech(C_{L,GX}), b_GX, a_GX, d_GX).

    Parameters
    ----------
    k:
        Number of assets.
    n_days:
        Sample length (informational).

    Attributes (after fitting)
    --------------------------
    params_  : ndarray [vech(C_{L,GX}), b_GX, a_GX, d_GX]
    H_seq_   : (T, k, k) in-sample H sequence
    loglik_  : float  (negative QMLE at optimum)
    """

    def __init__(self, k: int, n_days: int = 0) -> None:
        self.k = k
        self.n_days = n_days
        self.params_: np.ndarray | None = None
        self.H_seq_: np.ndarray | None = None
        self.loglik_: float | None = None

    @property
    def _n_vech(self) -> int:
        return vech_size(self.k)

    def _bounds(self) -> list:
        nv = self._n_vech
        return [(None, None)] * nv + [(0.0, 0.9999), (0.001, 0.9999), (0.0, 10.0)]

    # ------------------------------------------------------------------ #
    # Recursion
    # ------------------------------------------------------------------ #

    def compute_H_sequence(self, params: np.ndarray,
                           V_array: np.ndarray,
                           P_array: np.ndarray,
                           H0: np.ndarray) -> np.ndarray:
        """Run H_t = C_GX + b² H_{t-1} + a² V_{t-1} + d² P_{t-1} recursion.

        Parameters
        ----------
        params:
            [vech(C_{L,GX}), b_GX, a_GX, d_GX].
        V_array:
            (T, k, k) realised covariances.
        P_array:
            (T, k, k) outer products of returns.
        H0:
            (k, k) initial condition H_0.

        Returns
        -------
        ndarray of shape (T, k, k)
        """
        C, b, a, d = _unpack_gx(params, self.k)
        return _garchx_recursion(
            np.ascontiguousarray(C, dtype=float),
            b ** 2, a ** 2, d ** 2,
            np.ascontiguousarray(V_array, dtype=float),
            np.ascontiguousarray(P_array, dtype=float),
            np.ascontiguousarray(H0, dtype=float),
        )

    # ------------------------------------------------------------------ #
    # Negative QMLE
    # ------------------------------------------------------------------ #

    def loglik_H(self, params: np.ndarray,
                 P_array: np.ndarray,
                 V_array: np.ndarray,
                 H0: np.ndarray) -> float:
        """Negative QMLE: ½ Σ_t (log|H_t| + tr(H_t⁻¹ P_t)).

        Parameters
        ----------
        params:
            [vech(C_{L,GX}), b_GX, a_GX, d_GX].
        P_array:
            (T, k, k) outer products (evaluation target).
        V_array:
            (T, k, k) realised covariances (recursion input).
        H0:
            (k, k) initial condition.
        """
        nv = self._n_vech
        b, a = float(params[nv]), float(params[nv + 1])
        d = float(params[nv + 2])
        if b + a >= 1.0 or b < 0.0 or a < 0.0 or d < 0.0:
            return 1e18
        H_seq = self.compute_H_sequence(params, V_array, P_array, H0)
        return float(_neg_loglik_kernel(
            np.ascontiguousarray(H_seq, dtype=float),
            np.ascontiguousarray(P_array, dtype=float),
        ))

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def fit(self, P_array: np.ndarray,
            V_array: np.ndarray,
            H0: np.ndarray | None = None,
            ) -> tuple[np.ndarray, float, np.ndarray]:
        """Fit ScalarGARCHX by QMLE.

        Parameters
        ----------
        P_array:
            (T, k, k) outer products of daily returns.
        V_array:
            (T, k, k) realised covariances.
        H0:
            (k, k) initial condition.  If None, computed from P_array[:252].

        Returns
        -------
        params : ndarray [vech(C_{L,GX}), b_GX, a_GX, d_GX]
        loglik : float  (negative QMLE)
        H_sequence : ndarray (T, k, k)
        """
        P_array = np.ascontiguousarray(P_array, dtype=float)
        V_array = np.ascontiguousarray(V_array, dtype=float)
        if H0 is None:
            H0 = _init_cov(P_array)
        H0 = np.ascontiguousarray(ensure_pd(H0), dtype=float)

        b0, a0 = 0.7, 0.15
        nv = self._n_vech
        x0_base = _make_init_params(H0, self.k, b0, a0)
        x0 = np.concatenate([x0_base[:-2], [b0, a0, 0.05]])  # append d=0.05
        bounds = self._bounds()
        constraints = [{'type': 'ineq',
                         'fun': lambda p, nv=nv: 0.9999 - p[nv] - p[nv + 1]}]

        def obj(params):
            return self.loglik_H(params, P_array, V_array, H0)

        best_res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'ftol': 1e-9, 'maxiter': 500})

        for b_try, a_try, d_try in [(0.85, 0.10, 0.03), (0.60, 0.20, 0.08)]:
            x_base_try = _make_init_params(H0, self.k, b_try, a_try)
            x_try = np.concatenate([x_base_try[:-2], [b_try, a_try, d_try]])
            res2 = minimize(obj, x_try, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'ftol': 1e-9, 'maxiter': 500})
            if np.isfinite(res2.fun) and res2.fun < best_res.fun:
                best_res = res2

        params = best_res.x
        H_seq = self.compute_H_sequence(params, V_array, P_array, H0)
        nll = float(_neg_loglik_kernel(
            np.ascontiguousarray(H_seq), np.ascontiguousarray(P_array)))

        self.params_ = params
        self.H_seq_ = H_seq
        self.loglik_ = nll
        return params, nll, H_seq

    # ------------------------------------------------------------------ #
    # Forecasting
    # ------------------------------------------------------------------ #

    def forecast_H(self, params: np.ndarray,
                   params_M: np.ndarray | None,
                   H_T: np.ndarray,
                   M_T: np.ndarray,
                   h_steps: int = 1) -> np.ndarray:
        """Multi-step GARCH-X H forecasts.

        E_T[H_{T+s}] = C_GX + (b² + d²) E_T[H_{T+s-1}] + a² E_T[V_{T+s-1}]

        where E_T[V_{T+s-1}] = E_T[M_{T+s-1}] from the HEAVY-V equation if
        *params_M* is provided; otherwise M_T is held constant.

        Parameters
        ----------
        params:
            [vech(C_{L,GX}), b_GX, a_GX, d_GX].
        params_M:
            [vech(C_{L,M}), b_M, a_M] for the HEAVY-V M-equation.
            If None, M is held fixed at *M_T* for all horizons.
        H_T:
            (k, k) last in-sample conditional covariance.
        M_T:
            (k, k) last in-sample M value / realised covariance proxy.
        h_steps:
            Number of forecast horizons.

        Returns
        -------
        ndarray of shape (h_steps, k, k)
        """
        from src.models.heavy import _unpack as _unpack_heavy
        C_GX, b_GX, a_GX, d_GX = _unpack_gx(params, self.k)
        beta_H = b_GX ** 2 + d_GX ** 2  # combined persistence for H
        alpha_V = a_GX ** 2              # reactivity to V

        if params_M is not None:
            C_M, b_M, a_M = _unpack_heavy(params_M, self.k)
            phi_M = b_M ** 2 + a_M ** 2
        else:
            C_M = None
            phi_M = None

        forecasts = np.empty((h_steps, self.k, self.k))
        H_cur = ensure_pd(symmetrize(C_GX + beta_H * H_T + alpha_V * M_T))
        M_cur = M_T.copy()
        forecasts[0] = H_cur

        for s in range(1, h_steps):
            if params_M is not None:
                M_cur = ensure_pd(symmetrize(C_M + phi_M * M_cur))
            H_cur = ensure_pd(symmetrize(C_GX + beta_H * H_cur + alpha_V * M_cur))
            forecasts[s] = H_cur

        return forecasts

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def sandwich_se(self, params: np.ndarray,
                    P_array: np.ndarray,
                    V_array: np.ndarray,
                    H0: np.ndarray) -> np.ndarray:
        """Robust sandwich standard errors for GARCH-X parameters.

        Parameters
        ----------
        params:
            Estimated [vech(C_{L,GX}), b_GX, a_GX, d_GX].
        P_array, V_array:
            (T, k, k) outer products and realised covariances.
        H0:
            (k, k) initial condition.

        Returns
        -------
        ndarray of shape (p,)
        """
        P_array = np.ascontiguousarray(P_array, dtype=float)
        V_array = np.ascontiguousarray(V_array, dtype=float)
        H0 = np.ascontiguousarray(ensure_pd(H0), dtype=float)

        T = P_array.shape[0]
        p = len(params)
        eps = 1e-5

        scores = np.zeros((T, p))
        for j in range(p):
            d = np.zeros(p)
            d[j] = eps
            H_plus = self.compute_H_sequence(params + d, V_array, P_array, H0)
            H_minus = self.compute_H_sequence(params - d, V_array, P_array, H0)
            for t in range(T):
                H_p_t = ensure_pd(H_plus[t])
                H_m_t = ensure_pd(H_minus[t])
                P_t = P_array[t]
                l_p = -0.5 * (safe_logdet(H_p_t) + trace_prod(safe_inv(H_p_t), P_t))
                l_m = -0.5 * (safe_logdet(H_m_t) + trace_prod(safe_inv(H_m_t), P_t))
                scores[t, j] = (l_p - l_m) / (2.0 * eps)

        H_hess = np.zeros((p, p))
        for j in range(p):
            d = np.zeros(p)
            d[j] = eps
            g_p = np.array([
                (self.loglik_H(params + d + np.eye(p)[i] * eps, P_array, V_array, H0)
                 - self.loglik_H(params + d - np.eye(p)[i] * eps, P_array, V_array, H0))
                / (2.0 * eps) for i in range(p)
            ])
            g_m = np.array([
                (self.loglik_H(params - d + np.eye(p)[i] * eps, P_array, V_array, H0)
                 - self.loglik_H(params - d - np.eye(p)[i] * eps, P_array, V_array, H0))
                / (2.0 * eps) for i in range(p)
            ])
            H_hess[j] = (g_p - g_m) / (2.0 * eps)

        B = scores.T @ scores
        try:
            A_inv = np.linalg.inv(H_hess)
            cov = A_inv @ B @ A_inv
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            se = np.full(p, np.nan)
        return se
