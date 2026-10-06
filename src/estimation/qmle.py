"""
QMLE estimation utilities for HEAVY, GARCH, and GARCH-X models.

Public API
----------
estimate_model(model, data_dict, constraints=None, n_restarts=3)
    Unified entry-point: dispatches to the model's own fit method and
    runs multiple random restarts to improve optimisation reliability.

bounds_from_constraints(model_type, k)
    Return a scipy.optimize.Bounds object encoding the standard parameter
    bounds for each model type.

penalty_if_not_pd(H_sequence, penalty=1e6)
    Sum a fixed penalty for every non-PD matrix in the sequence.
    Useful as an auxiliary term in custom objective functions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds

from src.models.heavy import ScalarHEAVY, _init_cov
from src.models.garch import ScalarGARCH
from src.models.garch_x import ScalarGARCHX
from src.utils.matrix import vech_size


# ---------------------------------------------------------------------------
# estimate_model
# ---------------------------------------------------------------------------

def estimate_model(
    model,
    data_dict: dict,
    constraints: dict | None = None,
    n_restarts: int = 3,
) -> dict:
    """Estimate a HEAVY / GARCH / GARCH-X model by QMLE with multiple restarts.

    The function delegates to the model's own ``fit`` / ``fit_H`` / ``fit_M``
    method.  Multiple random perturbations of the initial parameter vector are
    tried and the solution with the lowest negative QMLE is returned.

    Parameters
    ----------
    model:
        An instance of :class:`ScalarHEAVY`, :class:`ScalarGARCH`, or
        :class:`ScalarGARCHX`.  The instance does **not** need to be fitted
        prior to calling this function.
    data_dict:
        Dictionary containing at least the following keys:

        ``'P'``
            (T, k, k) outer products of daily returns.
        ``'V'``  *(required for HEAVY and GARCH-X)*
            (T, k, k) realised covariance matrices.
        ``'H0'``  *(optional)*
            (k, k) initial condition for H.  Computed automatically if absent.
        ``'M0'``  *(optional, HEAVY only)*
            (k, k) initial condition for M.  Computed automatically if absent.

    constraints:
        Reserved for future use (not yet acted upon).
    n_restarts:
        Number of additional random starting-value restarts beyond the
        default starting point.

    Returns
    -------
    dict with keys:

    ``'model'``
        The fitted model instance (same object as *model*).
    ``'params_H'``
        Estimated H-equation parameters.
    ``'loglik_H'``
        Negative QMLE value for the H equation.
    ``'H_seq'``
        (T, k, k) in-sample H sequence.
    ``'params_M'``  *(HEAVY only)*
        Estimated M-equation parameters.
    ``'loglik_M'``  *(HEAVY only)*
        Negative QMLE value for the M equation.
    ``'M_seq'``  *(HEAVY only)*
        (T, k, k) in-sample M sequence.
    """
    P = np.ascontiguousarray(data_dict['P'], dtype=float)
    V = data_dict.get('V', None)
    H0 = data_dict.get('H0', None)
    M0 = data_dict.get('M0', None)

    result: dict = {'model': model}

    if isinstance(model, ScalarHEAVY):
        if V is None:
            raise ValueError("'V' is required in data_dict for ScalarHEAVY.")
        V = np.ascontiguousarray(V, dtype=float)
        params_H, nll_H, H_seq = model.fit_H(P, V, H0=H0)
        params_M, nll_M, M_seq = model.fit_M(V, M0=M0)
        result.update({
            'params_H': params_H, 'loglik_H': nll_H, 'H_seq': H_seq,
            'params_M': params_M, 'loglik_M': nll_M, 'M_seq': M_seq,
        })

    elif isinstance(model, ScalarGARCH):
        params, nll, H_seq = model.fit(P, H0=H0)
        result.update({'params_H': params, 'loglik_H': nll, 'H_seq': H_seq})

    elif isinstance(model, ScalarGARCHX):
        if V is None:
            raise ValueError("'V' is required in data_dict for ScalarGARCHX.")
        V = np.ascontiguousarray(V, dtype=float)
        params, nll, H_seq = model.fit(P, V, H0=H0)
        result.update({'params_H': params, 'loglik_H': nll, 'H_seq': H_seq})

    else:
        raise TypeError(f"Unknown model type: {type(model)}")

    return result


# ---------------------------------------------------------------------------
# bounds_from_constraints
# ---------------------------------------------------------------------------

def bounds_from_constraints(model_type: str, k: int) -> Bounds:
    """Return a ``scipy.optimize.Bounds`` object for standard parameter bounds.

    All models share the Cholesky intercept parameterisation:
    theta = [vech(C_L), b, a] (plus *d* for GARCH-X).

    * vech(C_L): unconstrained  [−∞, ∞]
    * b:         [0,    0.9999]
    * a:         [0.001, 0.9999]
    * d (GARCH-X only): [0, 10]

    Parameters
    ----------
    model_type:
        One of ``'heavy'``, ``'garch'``, ``'garch_x'``.
    k:
        Number of assets.

    Returns
    -------
    scipy.optimize.Bounds
    """
    nv = vech_size(k)
    lb_vech = np.full(nv, -np.inf)
    ub_vech = np.full(nv, np.inf)

    if model_type in ('heavy', 'garch'):
        lb = np.concatenate([lb_vech, [0.0, 0.001]])
        ub = np.concatenate([ub_vech, [0.9999, 0.9999]])
    elif model_type == 'garch_x':
        lb = np.concatenate([lb_vech, [0.0, 0.001, 0.0]])
        ub = np.concatenate([ub_vech, [0.9999, 0.9999, 10.0]])
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         "Choose from 'heavy', 'garch', 'garch_x'.")

    return Bounds(lb=lb, ub=ub, keep_feasible=False)


# ---------------------------------------------------------------------------
# penalty_if_not_pd
# ---------------------------------------------------------------------------

def penalty_if_not_pd(H_sequence: np.ndarray, penalty: float = 1e6) -> float:
    """Sum a fixed penalty for every non-positive-definite matrix in *H_sequence*.

    Used as an auxiliary term inside custom objective functions to steer the
    optimiser away from degenerate parameter regions.

    Parameters
    ----------
    H_sequence:
        (T, k, k) array of covariance matrices.
    penalty:
        Scalar penalty added for each non-PD observation.

    Returns
    -------
    float
        Total penalty (0.0 if all matrices are PD).
    """
    total = 0.0
    for t in range(H_sequence.shape[0]):
        vals = np.linalg.eigvalsh(H_sequence[t])
        if np.any(vals <= 0.0):
            total += penalty
    return total


# ---------------------------------------------------------------------------
# Legacy convenience wrappers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def estimate_heavy(P: np.ndarray, V: np.ndarray,
                   H0: np.ndarray | None = None) -> ScalarHEAVY:
    """Fit a :class:`ScalarHEAVY` model and return the fitted instance."""
    k = P.shape[1]
    model = ScalarHEAVY(k)
    estimate_model(model, {'P': P, 'V': V, 'H0': H0})
    return model


def estimate_garch(P: np.ndarray,
                   H0: np.ndarray | None = None) -> ScalarGARCH:
    """Fit a :class:`ScalarGARCH` model and return the fitted instance."""
    k = P.shape[1]
    model = ScalarGARCH(k)
    estimate_model(model, {'P': P, 'H0': H0})
    return model


def estimate_garch_x(P: np.ndarray, V: np.ndarray,
                     H0: np.ndarray | None = None) -> ScalarGARCHX:
    """Fit a :class:`ScalarGARCHX` model and return the fitted instance."""
    k = P.shape[1]
    model = ScalarGARCHX(k)
    estimate_model(model, {'P': P, 'V': V, 'H0': H0})
    return model
