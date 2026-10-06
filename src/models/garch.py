"""
Scalar Multivariate GARCH(1,1) Model.

Recursion:
    H_t = C_G + b_G² H_{t-1} + a_G² P_{t-1}

where P_t = R_t R_tᵀ is the outer product of daily returns and
C_G = C_{L,G} C_{L,G}ᵀ is a SPD intercept parameterised via its lower-triangular
Cholesky factor.

QMLE:  l_{G,t} = -½ (log|H_t| + tr(H_t⁻¹ P_t))

Stationarity constraint:  b_G + a_G < 1.

Multi-step forecast (s ≥ 1):
    H_{T+s|T} = C_G·Σᵢ₌₀^{s-2} φ_G^i + φ_G^{s-1}·H_{T+1|T}
              = ω_G + φ_G^{s-1}·(H_{T+1|T} − ω_G)
    where  φ_G = b_G² + a_G²  and  ω_G = C_G / (1 − φ_G).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.models.heavy import (   # reuse the generic Numba kernels
    _cov_recursion,
    _neg_loglik_kernel,
    _init_cov,
    _make_init_params,
    _unpack,
)
from src.utils.linalg import (
    ensure_pd,
    safe_inv,
    safe_logdet,
    symmetrize,
    trace_prod,
)
from src.utils.matrix import vech, vech_size


class ScalarGARCH:
    """Scalar Multivariate GARCH(1,1) model.

    Recursion: H_t = C_G + b_G² H_{t-1} + a_G² P_{t-1}

    Parameters are stored as theta_G = (vech(C_{L,G}), b_G, a_G) where
    C_{L,G} is the lower-triangular Cholesky factor of the SPD intercept.

    Parameters
    ----------
    k:
        Number of assets.
    n_days:
        Sample length (informational).

    Attributes (after fitting)
    --------------------------
    params_  : ndarray [vech(C_{L,G}), b_G, a_G]
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
        return [(None, None)] * self._n_vech + [(0.0, 0.9999), (0.001, 0.9999)]

    # ------------------------------------------------------------------ #
    # Recursion
    # ------------------------------------------------------------------ #

    def compute_H_sequence(self, params: np.ndarray,
                           P_array: np.ndarray,
                           H0: np.ndarray) -> np.ndarray:
        """Run H_t = C_G + b_G² H_{t-1} + a_G² P_{t-1} recursion.

        Parameters
        ----------
        params:
            [vech(C_{L,G}), b_G, a_G].
        P_array:
            (T, k, k) outer products P_t = R_t R_tᵀ used in the recursion.
        H0:
            (k, k) initial condition H_0.

        Returns
        -------
        ndarray of shape (T, k, k)
        """
        C, b, a = _unpack(params, self.k)
        return _cov_recursion(
            np.ascontiguousarray(C, dtype=float),
            b ** 2, a ** 2,
            np.ascontiguousarray(P_array, dtype=float),
            np.ascontiguousarray(H0, dtype=float),
        )

    # ------------------------------------------------------------------ #
    # Negative QMLE
    # ------------------------------------------------------------------ #

    def loglik_H(self, params: np.ndarray,
                 P_array: np.ndarray,
                 H0: np.ndarray) -> float:
        """Negative QMLE: ½ Σ_t (log|H_t| + tr(H_t⁻¹ P_t)).

        Parameters
        ----------
        params:
            [vech(C_{L,G}), b_G, a_G].
        P_array:
            (T, k, k) outer products (used both as recursion driver and target).
        H0:
            (k, k) initial condition.
        """
        nv = self._n_vech
        b, a = float(params[nv]), float(params[nv + 1])
        if b + a >= 1.0 or b < 0.0 or a < 0.0:
            return 1e18
        H_seq = self.compute_H_sequence(params, P_array, H0)
        return float(_neg_loglik_kernel(
            np.ascontiguousarray(H_seq, dtype=float),
            np.ascontiguousarray(P_array, dtype=float),
        ))

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #

    def fit(self, P_array: np.ndarray,
            H0: np.ndarray | None = None,
            ) -> tuple[np.ndarray, float, np.ndarray]:
        """Fit ScalarGARCH by QMLE.

        Parameters
        ----------
        P_array:
            (T, k, k) outer products of daily returns.
        H0:
            (k, k) initial condition.  If None, computed as the sample mean
            of P_array[:252].

        Returns
        -------
        params : ndarray [vech(C_{L,G}), b_G, a_G]
        loglik : float  (negative QMLE)
        H_sequence : ndarray (T, k, k)
        """
        P_array = np.ascontiguousarray(P_array, dtype=float)
        if H0 is None:
            H0 = _init_cov(P_array)
        H0 = np.ascontiguousarray(ensure_pd(H0), dtype=float)

        b0, a0 = 0.9, 0.06
        x0 = _make_init_params(H0, self.k, b0, a0)
        nv = self._n_vech
        bounds = self._bounds()
        constraints = [{'type': 'ineq',
                         'fun': lambda p, nv=nv: 0.9999 - p[nv] - p[nv + 1]}]

        def obj(params):
            return self.loglik_H(params, P_array, H0)

        best_res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'ftol': 1e-9, 'maxiter': 500})

        for b_try, a_try in [(0.85, 0.10), (0.80, 0.15), (0.70, 0.20)]:
            x_try = _make_init_params(H0, self.k, b_try, a_try)
            res2 = minimize(obj, x_try, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'ftol': 1e-9, 'maxiter': 500})
            if np.isfinite(res2.fun) and res2.fun < best_res.fun:
                best_res = res2

        params = best_res.x
        H_seq = self.compute_H_sequence(params, P_array, H0)
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
                   H_T: np.ndarray,
                   h_steps: int = 1) -> np.ndarray:
        """Multi-step H forecasts via geometric mean-reversion.

        E_T[H_{T+s}] = ω_G + φ_G^{s-1}·(H_{T+1|T} − ω_G)

        where  φ_G = b_G² + a_G²  and  ω_G = C_G / (1 − φ_G).

        Equivalently: Σᵢ₌₀^{s-2} φ_G^i · C_G  +  φ_G^{s-1} · H_{T+1|T}.

        Parameters
        ----------
        params:
            [vech(C_{L,G}), b_G, a_G].
        H_T:
            (k, k) last in-sample conditional covariance.
        h_steps:
            Number of forecast horizons.

        Returns
        -------
        ndarray of shape (h_steps, k, k)
        """
        C_G, b_G, a_G = _unpack(params, self.k)
        phi_G = b_G ** 2 + a_G ** 2
        denom = max(1.0 - phi_G, 1e-10)
        omega_G = C_G / denom

        H1 = ensure_pd(symmetrize(C_G + phi_G * H_T))
        forecasts = np.empty((h_steps, self.k, self.k))
        forecasts[0] = H1

        for s in range(1, h_steps):
            H_s = omega_G + (phi_G ** s) * (H1 - omega_G)
            forecasts[s] = ensure_pd(symmetrize(H_s))

        return forecasts

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def sandwich_se(self, params: np.ndarray,
                    P_array: np.ndarray,
                    H0: np.ndarray) -> np.ndarray:
        """Robust sandwich standard errors for GARCH parameters.

        See :meth:`ScalarHEAVY.sandwich_se_H` for the methodology.

        Parameters
        ----------
        params:
            Estimated [vech(C_{L,G}), b_G, a_G].
        P_array:
            (T, k, k) outer products.
        H0:
            (k, k) initial condition.

        Returns
        -------
        ndarray of shape (p,)
        """
        P_array = np.ascontiguousarray(P_array, dtype=float)
        H0 = np.ascontiguousarray(ensure_pd(H0), dtype=float)

        T = P_array.shape[0]
        p = len(params)
        eps = 1e-5

        scores = np.zeros((T, p))
        for j in range(p):
            d = np.zeros(p)
            d[j] = eps
            H_plus = self.compute_H_sequence(params + d, P_array, H0)
            H_minus = self.compute_H_sequence(params - d, P_array, H0)
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
                (self.loglik_H(params + d + np.eye(p)[i] * eps, P_array, H0)
                 - self.loglik_H(params + d - np.eye(p)[i] * eps, P_array, H0))
                / (2.0 * eps) for i in range(p)
            ])
            g_m = np.array([
                (self.loglik_H(params - d + np.eye(p)[i] * eps, P_array, H0)
                 - self.loglik_H(params - d - np.eye(p)[i] * eps, P_array, H0))
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
