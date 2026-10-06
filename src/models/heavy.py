"""
Scalar Multivariate HEAVY Model — Noureldin, Shephard & Sheppard (2012).

The HEAVY model has two variation-free equations:

H-equation (HEAVY-P):
    H_t = C_H C_H' + b_H² H_{t-1} + a_H² V_{t-1}

    QMLE:  l_{H,t} = -½ (log|H_t| + tr(H_t⁻¹ P_t))
    where  P_t = R_t R_t'  (outer product of daily returns).

M-equation (HEAVY-V):
    M_t = C_M C_M' + b_M² M_{t-1} + a_M² V_{t-1}

    QMLE:  l_{M,t} = -½ (log|M_t| + tr(M_t⁻¹ V_t))

Scalar restriction: a single scalar multiplies every vech element of C, H, M, V.

Parameters stored as theta = (vech(C_L), b, a) where C_L is the lower-triangular
Cholesky factor of the SPD intercept C = C_L C_L'.  Non-negativity of the
intercept is guaranteed by construction; stationarity is enforced via
the constraint b + a < 1.

Estimation is by QMLE with scipy SLSQP (L-BFGS-B used as a secondary solver).
Robust sandwich standard errors are available via :meth:`sandwich_se_H`.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.optimize import minimize

from src.utils.linalg import (
    ensure_pd,
    ensure_pd_numba,
    logdet_numba,
    safe_inv,
    safe_logdet,
    symmetrize,
    trace_prod,
    trace_prod_numba,
)
from src.utils.matrix import lower_chol_from_vech, vech, vech_size


# ---------------------------------------------------------------------------
# Numba-JIT inner kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _cov_recursion(C: np.ndarray, b2: float, a2: float,
                   V_array: np.ndarray, init: np.ndarray,
                   epsilon: float = 1e-8) -> np.ndarray:
    """Numba-compiled covariance recursion X_t = C + b2·X_{t-1} + a2·Z_{t-1}.

    Used for both the H-equation (b2 = b_H², a2 = a_H², Z = V) and the
    M-equation (b2 = b_M², a2 = a_M², Z = V).

    Parameters
    ----------
    C:
        (k, k) SPD intercept matrix.
    b2:
        Scalar coefficient for the lagged covariance X_{t-1}.
    a2:
        Scalar coefficient for the lagged realized measure Z_{t-1}.
    V_array:
        (T, k, k) array; V_array[t] = Z_t.  The pre-sample Z_0 is
        approximated by *init*.
    init:
        (k, k) initial condition; used as X_0 and as Z_0.
    epsilon:
        Minimum eigenvalue for PD enforcement at each step.

    Returns
    -------
    np.ndarray of shape (T, k, k)
        X_1, …, X_T computed from the recursion.
    """
    T = V_array.shape[0]
    k = V_array.shape[1]
    X = np.empty((T, k, k))
    X_prev = init.copy()
    for t in range(T):
        Z_prev = init if t == 0 else V_array[t - 1]
        X_t = C + b2 * X_prev + a2 * Z_prev
        X_t = ensure_pd_numba(X_t, epsilon)
        X[t] = X_t
        X_prev = X_t
    return X


@njit(cache=True)
def _neg_loglik_kernel(X_seq: np.ndarray,
                       eval_array: np.ndarray) -> float:
    """Numba-compiled negative QMLE: ½ Σ_t (log|X_t| + tr(X_t⁻¹ E_t)).

    Parameters
    ----------
    X_seq:
        (T, k, k) array of conditional covariance forecasts.
    eval_array:
        (T, k, k) array of realised quantities evaluated against X_seq.

    Returns
    -------
    float
        Negative quasi-log-likelihood (to be minimised).
    """
    T = X_seq.shape[0]
    total = 0.0
    for t in range(T):
        X_t = X_seq[t]
        E_t = eval_array[t]
        ld = logdet_numba(X_t)
        if ld >= 1e17:
            return 1e18
        X_inv = np.linalg.inv(X_t)
        total += ld + trace_prod_numba(X_inv, E_t)
    return 0.5 * total


# ---------------------------------------------------------------------------
# Helper: unpack parameter vector → (C, b_raw, a_raw)
# ---------------------------------------------------------------------------

def _unpack(params: np.ndarray, k: int):
    """Return (C, b, a) from the packed parameter vector.

    Parameters
    ----------
    params:
        1-D array [vech(C_L), b, a] of length k*(k+1)//2 + 2.
    k:
        Number of assets.

    Returns
    -------
    C : (k, k) SPD intercept = C_L @ C_L.T
    b : float  — raw persistence parameter (coefficient in recursion = b²)
    a : float  — raw reactivity parameter  (coefficient in recursion = a²)
    """
    nv = vech_size(k)
    C_L = lower_chol_from_vech(params[:nv], k)
    C = C_L @ C_L.T
    b = float(params[nv])
    a = float(params[nv + 1])
    return C, b, a


def _make_init_params(init_cov: np.ndarray, k: int,
                      b0: float, a0: float) -> np.ndarray:
    """Build the initial parameter vector from a target covariance matrix.

    The intercept is scaled so that the unconditional mean equals *init_cov*
    under stationarity: C = init_cov · (1 − b0² − a0²).

    Parameters
    ----------
    init_cov:
        (k, k) SPD matrix (e.g. sample covariance used for initialisation).
    b0, a0:
        Starting values for the raw persistence / reactivity parameters.
    """
    scale = max(1.0 - b0 ** 2 - a0 ** 2, 1e-4)
    C0 = ensure_pd(init_cov * scale)
    try:
        C_L0 = np.linalg.cholesky(C0)
    except np.linalg.LinAlgError:
        C_L0 = np.linalg.cholesky(C0 + 1e-6 * np.eye(k))
    return np.concatenate([vech(C_L0), [b0, a0]])


def _init_cov(arr: np.ndarray) -> np.ndarray:
    """Compute sample-mean covariance from the first min(252, T//4) observations."""
    n = max(2, min(252, arr.shape[0] // 4))
    return ensure_pd(np.mean(arr[:n], axis=0))


# ---------------------------------------------------------------------------
# ScalarHEAVY class
# ---------------------------------------------------------------------------

class ScalarHEAVY:
    """Scalar Multivariate HEAVY model (Noureldin, Shephard & Sheppard 2012).

    Two variation-free equations are estimated separately by QMLE:

    H-equation (HEAVY-P):
        H_t = C_H + b_H² H_{t-1} + a_H² V_{t-1},  C_H = C_{L,H} C_{L,H}ᵀ
        evaluated against the outer-product P_t = R_t R_tᵀ.

    M-equation (HEAVY-V):
        M_t = C_M + b_M² M_{t-1} + a_M² V_{t-1},  C_M = C_{L,M} C_{L,M}ᵀ
        evaluated against the realised covariance V_t.

    Parameters
    ----------
    k:
        Number of assets.
    n_days:
        Length of the estimation sample (informational; not used internally).

    Attributes (set after fitting)
    ----------
    params_H_ : ndarray [vech(C_{L,H}), b_H, a_H]
    params_M_ : ndarray [vech(C_{L,M}), b_M, a_M]
    H_seq_    : (T, k, k)  in-sample H_t sequence
    M_seq_    : (T, k, k)  in-sample M_t sequence
    loglik_H_ : float  negative QMLE for H equation
    loglik_M_ : float  negative QMLE for M equation
    """

    def __init__(self, k: int, n_days: int = 0) -> None:
        self.k = k
        self.n_days = n_days
        self.params_H_: np.ndarray | None = None
        self.params_M_: np.ndarray | None = None
        self.H_seq_: np.ndarray | None = None
        self.M_seq_: np.ndarray | None = None
        self.loglik_H_: float | None = None
        self.loglik_M_: float | None = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @property
    def _n_vech(self) -> int:
        return vech_size(self.k)

    def _n_params(self) -> int:
        return self._n_vech + 2

    def _bounds(self) -> list:
        """Return SLSQP bounds list: unconstrained vech entries, then b, a."""
        nv = self._n_vech
        return [(None, None)] * nv + [(0.0, 0.9999), (0.001, 0.9999)]

    @staticmethod
    def _stationary_constraint(params: np.ndarray, nv: int) -> float:
        """Inequality constraint b + a ≤ 0.9999 (return ≥ 0 means feasible)."""
        return 0.9999 - params[nv] - params[nv + 1]

    # ------------------------------------------------------------------ #
    # Recursion methods
    # ------------------------------------------------------------------ #

    def compute_H_sequence(self, params_H: np.ndarray,
                           V_array: np.ndarray,
                           H0: np.ndarray) -> np.ndarray:
        """Run H_t recursion given parameters and initial condition.

        H_t = C_H + b_H² H_{t-1} + a_H² V_{t-1}

        Parameters
        ----------
        params_H:
            1-D array [vech(C_{L,H}), b_H, a_H].
        V_array:
            (T, k, k) realised covariances V_1, …, V_T.
        H0:
            (k, k) initial condition H_0.

        Returns
        -------
        ndarray of shape (T, k, k)
            H_1, …, H_T.
        """
        C, b, a = _unpack(params_H, self.k)
        return _cov_recursion(
            np.ascontiguousarray(C, dtype=float),
            b ** 2, a ** 2,
            np.ascontiguousarray(V_array, dtype=float),
            np.ascontiguousarray(H0, dtype=float),
        )

    def compute_M_sequence(self, params_M: np.ndarray,
                           V_array: np.ndarray,
                           M0: np.ndarray) -> np.ndarray:
        """Run M_t recursion given parameters and initial condition.

        M_t = C_M + b_M² M_{t-1} + a_M² V_{t-1}

        Parameters
        ----------
        params_M:
            1-D array [vech(C_{L,M}), b_M, a_M].
        V_array:
            (T, k, k) realised covariances V_1, …, V_T.
        M0:
            (k, k) initial condition M_0.

        Returns
        -------
        ndarray of shape (T, k, k)
            M_1, …, M_T.
        """
        C, b, a = _unpack(params_M, self.k)
        return _cov_recursion(
            np.ascontiguousarray(C, dtype=float),
            b ** 2, a ** 2,
            np.ascontiguousarray(V_array, dtype=float),
            np.ascontiguousarray(M0, dtype=float),
        )

    # ------------------------------------------------------------------ #
    # Negative QMLE objectives (to minimise)
    # ------------------------------------------------------------------ #

    def loglik_H(self, params: np.ndarray,
                 P_array: np.ndarray,
                 V_array: np.ndarray,
                 H0: np.ndarray) -> float:
        """Negative QMLE for the H-equation: ½ Σ_t (log|H_t| + tr(H_t⁻¹ P_t)).

        Parameters
        ----------
        params:
            [vech(C_{L,H}), b_H, a_H].
        P_array:
            (T, k, k) outer products P_t = R_t R_tᵀ.
        V_array:
            (T, k, k) realised covariances used in the recursion.
        H0:
            (k, k) initial condition.

        Returns
        -------
        float
            Negative QMLE value (to be minimised).
        """
        nv = self._n_vech
        b, a = float(params[nv]), float(params[nv + 1])
        if b + a >= 1.0 or b < 0.0 or a < 0.0:
            return 1e18
        H_seq = self.compute_H_sequence(params, V_array, H0)
        return float(_neg_loglik_kernel(
            np.ascontiguousarray(H_seq, dtype=float),
            np.ascontiguousarray(P_array, dtype=float),
        ))

    def loglik_M(self, params: np.ndarray,
                 V_array: np.ndarray,
                 M0: np.ndarray) -> float:
        """Negative QMLE for the M-equation: ½ Σ_t (log|M_t| + tr(M_t⁻¹ V_t)).

        Parameters
        ----------
        params:
            [vech(C_{L,M}), b_M, a_M].
        V_array:
            (T, k, k) realised covariances (evaluation targets).
        M0:
            (k, k) initial condition.

        Returns
        -------
        float
            Negative QMLE value (to be minimised).
        """
        nv = self._n_vech
        b, a = float(params[nv]), float(params[nv + 1])
        if b + a >= 1.0 or b < 0.0 or a < 0.0:
            return 1e18
        M_seq = self.compute_M_sequence(params, V_array, M0)
        return float(_neg_loglik_kernel(
            np.ascontiguousarray(M_seq, dtype=float),
            np.ascontiguousarray(V_array, dtype=float),
        ))

    # ------------------------------------------------------------------ #
    # Fitting methods
    # ------------------------------------------------------------------ #

    def fit_H(self, P_array: np.ndarray,
              V_array: np.ndarray,
              H0: np.ndarray | None = None,
              ) -> tuple[np.ndarray, float, np.ndarray]:
        """Estimate H-equation parameters by QMLE.

        H_t = C_H + b_H² H_{t-1} + a_H²V_{t-1}  evaluated against P_t.

        Parameters
        ----------
        P_array:
            (T, k, k) outer products P_t = R_t R_tᵀ.
        V_array:
            (T, k, k) realised covariances (drive the recursion).
        H0:
            (k, k) initial condition.  If None, computed from the
            sample mean of P_array[:252].

        Returns
        -------
        params : ndarray [vech(C_{L,H}), b_H, a_H]
        loglik : float  (negative QMLE at optimum)
        H_sequence : ndarray (T, k, k)
        """
        P_array = np.ascontiguousarray(P_array, dtype=float)
        V_array = np.ascontiguousarray(V_array, dtype=float)
        if H0 is None:
            H0 = _init_cov(P_array)
        H0 = np.ascontiguousarray(ensure_pd(H0), dtype=float)

        b0, a0 = 0.7, 0.15
        x0 = _make_init_params(H0, self.k, b0, a0)
        nv = self._n_vech
        bounds = self._bounds()
        constraints = [{'type': 'ineq',
                         'fun': lambda p, nv=nv: self._stationary_constraint(p, nv)}]

        def obj(params):
            return self.loglik_H(params, P_array, V_array, H0)

        best_res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'ftol': 1e-9, 'maxiter': 500})

        # Multiple restarts with perturbed starting values
        for b_try, a_try in [(0.85, 0.10), (0.60, 0.20), (0.50, 0.30)]:
            x_try = _make_init_params(H0, self.k, b_try, a_try)
            res2 = minimize(obj, x_try, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'ftol': 1e-9, 'maxiter': 500})
            if np.isfinite(res2.fun) and res2.fun < best_res.fun:
                best_res = res2

        params_H = best_res.x
        H_seq = self.compute_H_sequence(params_H, V_array, H0)
        nll = float(_neg_loglik_kernel(
            np.ascontiguousarray(H_seq), np.ascontiguousarray(P_array)))

        self.params_H_ = params_H
        self.H_seq_ = H_seq
        self.loglik_H_ = nll
        return params_H, nll, H_seq

    def fit_M(self, V_array: np.ndarray,
              M0: np.ndarray | None = None,
              ) -> tuple[np.ndarray, float, np.ndarray]:
        """Estimate M-equation parameters by QMLE.

        M_t = C_M + b_M² M_{t-1} + a_M² V_{t-1}  evaluated against V_t.

        Parameters
        ----------
        V_array:
            (T, k, k) realised covariances.
        M0:
            (k, k) initial condition.  If None, computed from the
            sample mean of V_array[:252].

        Returns
        -------
        params : ndarray [vech(C_{L,M}), b_M, a_M]
        loglik : float  (negative QMLE at optimum)
        M_sequence : ndarray (T, k, k)
        """
        V_array = np.ascontiguousarray(V_array, dtype=float)
        if M0 is None:
            M0 = _init_cov(V_array)
        M0 = np.ascontiguousarray(ensure_pd(M0), dtype=float)

        b0, a0 = 0.7, 0.15
        x0 = _make_init_params(M0, self.k, b0, a0)
        nv = self._n_vech
        bounds = self._bounds()
        constraints = [{'type': 'ineq',
                         'fun': lambda p, nv=nv: self._stationary_constraint(p, nv)}]

        def obj(params):
            return self.loglik_M(params, V_array, M0)

        best_res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'ftol': 1e-9, 'maxiter': 500})

        for b_try, a_try in [(0.85, 0.10), (0.60, 0.20), (0.50, 0.30)]:
            x_try = _make_init_params(M0, self.k, b_try, a_try)
            res2 = minimize(obj, x_try, method='SLSQP', bounds=bounds,
                            constraints=constraints,
                            options={'ftol': 1e-9, 'maxiter': 500})
            if np.isfinite(res2.fun) and res2.fun < best_res.fun:
                best_res = res2

        params_M = best_res.x
        M_seq = self.compute_M_sequence(params_M, V_array, M0)
        nll = float(_neg_loglik_kernel(
            np.ascontiguousarray(M_seq), np.ascontiguousarray(V_array)))

        self.params_M_ = params_M
        self.M_seq_ = M_seq
        self.loglik_M_ = nll
        return params_M, nll, M_seq

    def fit(self, P_array: np.ndarray,
            V_array: np.ndarray,
            ) -> tuple[tuple, tuple]:
        """Fit both HEAVY-P (H) and HEAVY-V (M) equations.

        Variation-freeness: the two equations are estimated independently.

        Parameters
        ----------
        P_array:
            (T, k, k) outer products of daily returns.
        V_array:
            (T, k, k) realised covariances.

        Returns
        -------
        results_H : (params_H, loglik_H, H_sequence)
        results_M : (params_M, loglik_M, M_sequence)
        """
        results_H = self.fit_H(P_array, V_array)
        results_M = self.fit_M(V_array)
        return results_H, results_M

    # ------------------------------------------------------------------ #
    # Forecasting
    # ------------------------------------------------------------------ #

    def forecast_H(self, params_H: np.ndarray,
                   params_M: np.ndarray,
                   H_T: np.ndarray,
                   M_T: np.ndarray,
                   h_steps: int = 1) -> np.ndarray:
        """Multi-step ahead H forecast using the paper recursion.

        For s = 1:  H_{T+1|T} = C_H + b_H² H_T + a_H² M_T  (M_T used as
                    the best available estimate of V_T ≈ E_T[V_T]).
        For s ≥ 2:  propagate jointly with M using
                    E_T[H_{T+s}] = C_H + b_H² E_T[H_{T+s-1}] + a_H² E_T[V_{T+s-1}]
                    where E_T[V_{T+s-1}] = E_T[M_{T+s-1}].

        M propagation (s ≥ 2):
            E_T[M_{T+s}] = C_M + (b_M² + a_M²) E_T[M_{T+s-1}]

        Paper reference: eq. (2.5)–(2.7) in Noureldin et al. (2012).

        Parameters
        ----------
        params_H, params_M:
            Parameter vectors for the H and M equations.
        H_T:
            (k, k) last in-sample conditional covariance H_T.
        M_T:
            (k, k) last in-sample M forecast M_T (≈ E_{T-1}[V_T]).
        h_steps:
            Number of steps ahead (≥ 1).

        Returns
        -------
        ndarray of shape (h_steps, k, k)
            H_{T+1|T}, …, H_{T+h_steps|T}.
        """
        C_H, b_H, a_H = _unpack(params_H, self.k)
        C_M, b_M, a_M = _unpack(params_M, self.k)
        beta_H = b_H ** 2        # actual coefficient in H recursion
        alpha_H = a_H ** 2       # actual coefficient in H recursion
        phi_M = b_M ** 2 + a_M ** 2   # persistence of M equation

        forecasts = np.empty((h_steps, self.k, self.k))
        H_cur = C_H + beta_H * H_T + alpha_H * M_T   # H_{T+1|T}
        M_cur = M_T.copy()                             # M_{T+1|T} used for s=1

        forecasts[0] = H_cur

        for s in range(1, h_steps):
            # Propagate M: E[M_{T+s+1}] = C_M + phi_M * E[M_{T+s}]
            M_cur = C_M + phi_M * M_cur
            M_cur = ensure_pd(symmetrize(M_cur))

            # Propagate H: E[H_{T+s+1}] = C_H + beta_H * E[H_{T+s}] + alpha_H * E[M_{T+s}]
            H_cur = C_H + beta_H * H_cur + alpha_H * M_cur
            H_cur = ensure_pd(symmetrize(H_cur))

            forecasts[s] = H_cur

        return forecasts

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def sandwich_se_H(self, params_H: np.ndarray,
                      P_array: np.ndarray,
                      V_array: np.ndarray,
                      H0: np.ndarray) -> np.ndarray:
        """Robust sandwich standard errors for H-equation parameters.

        Computes SE = √diag(A⁻¹ B A⁻¹) where
            A = numerical Hessian of  Σ_t Q_t  (negative QMLE)
            B = Σ_t ŝ_t ŝ_tᵀ  (outer product of per-obs scores).

        Per-observation scores are obtained by numerical differentiation of
        each contribution l_{H,t} = −½(log|H_t| + tr(H_t⁻¹ P_t)) with
        respect to params_H, accounting for the full recursion dependency.

        Parameters
        ----------
        params_H:
            Estimated parameter vector [vech(C_{L,H}), b_H, a_H].
        P_array:
            (T, k, k) outer products.
        V_array:
            (T, k, k) realised covariances (recursion inputs).
        H0:
            (k, k) initial condition.

        Returns
        -------
        ndarray of shape (p,)
            Robust standard errors for each element of params_H.
        """
        P_array = np.ascontiguousarray(P_array, dtype=float)
        V_array = np.ascontiguousarray(V_array, dtype=float)
        H0 = np.ascontiguousarray(ensure_pd(H0), dtype=float)

        T = P_array.shape[0]
        p = len(params_H)
        eps = 1e-5

        # Per-observation scores via central differences of full recursion
        scores = np.zeros((T, p))
        for j in range(p):
            d = np.zeros(p)
            d[j] = eps
            H_plus = self.compute_H_sequence(params_H + d, V_array, H0)
            H_minus = self.compute_H_sequence(params_H - d, V_array, H0)
            for t in range(T):
                H_p_t = ensure_pd(H_plus[t])
                H_m_t = ensure_pd(H_minus[t])
                P_t = P_array[t]
                l_p = -0.5 * (safe_logdet(H_p_t) + trace_prod(safe_inv(H_p_t), P_t))
                l_m = -0.5 * (safe_logdet(H_m_t) + trace_prod(safe_inv(H_m_t), P_t))
                scores[t, j] = (l_p - l_m) / (2.0 * eps)

        # Numerical Hessian of total negative QMLE via finite differences of gradient
        H_hess = np.zeros((p, p))
        for j in range(p):
            d = np.zeros(p)
            d[j] = eps
            g_p = np.array([
                (self.loglik_H(params_H + d + np.eye(p)[i] * eps, P_array, V_array, H0)
                 - self.loglik_H(params_H + d - np.eye(p)[i] * eps, P_array, V_array, H0))
                / (2.0 * eps) for i in range(p)
            ])
            g_m = np.array([
                (self.loglik_H(params_H - d + np.eye(p)[i] * eps, P_array, V_array, H0)
                 - self.loglik_H(params_H - d - np.eye(p)[i] * eps, P_array, V_array, H0))
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
