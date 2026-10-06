"""Numerically stable linear algebra utilities for covariance matrix operations.

Exports both pure-NumPy helpers and Numba-JIT compiled variants that can be
called from inside ``@njit`` kernels.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.linalg import cho_factor, cho_solve


def stable_log_det(A: np.ndarray) -> float:
    """Compute log|det(A)| for a positive-definite matrix.

    Uses :func:`numpy.linalg.slogdet` and raises if the determinant sign is
    non-positive (indicating A is not positive definite).

    Parameters
    ----------
    A:
        Symmetric (k, k) positive-definite array.

    Returns
    -------
    float
        Natural logarithm of the determinant.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix is not positive definite (sign <= 0).
    """
    A = np.asarray(A, dtype=float)
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        raise np.linalg.LinAlgError(
            f"Matrix is not positive definite: slogdet sign = {sign}"
        )
    return float(logdet)


def stable_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve A @ X = B using Cholesky when possible, falling back to lstsq.

    Attempts a Cholesky factorisation of A for efficiency and numerical
    stability.  If A is not positive definite (Cholesky raises), falls back
    to :func:`numpy.linalg.lstsq` with ``rcond=None``.

    Parameters
    ----------
    A:
        (k, k) coefficient matrix, ideally symmetric positive definite.
    B:
        (k,) or (k, n) right-hand-side array.

    Returns
    -------
    np.ndarray
        Solution array X with same shape as B.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    try:
        c, low = cho_factor(A, lower=True, check_finite=False)
        return cho_solve((c, low), B, check_finite=False)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, B, rcond=None)[0]


def is_positive_definite(A: np.ndarray) -> bool:
    """Return True if A is symmetric positive definite via Cholesky test.

    Parameters
    ----------
    A:
        Square (k, k) array.

    Returns
    -------
    bool
        True if A passes the Cholesky decomposition check.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def regularize_matrix(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Clip eigenvalues at eps to enforce positive definiteness.

    Symmetrises A, computes its eigendecomposition, clips all eigenvalues
    to at least eps, and reconstructs the matrix.  Safer than adding eps*I
    when the matrix has large negative eigenvalues.

    Parameters
    ----------
    A:
        Symmetric (k, k) array to regularize.
    eps:
        Minimum eigenvalue floor.  Default 1e-8.

    Returns
    -------
    np.ndarray
        Regularised symmetric (k, k) PD array.
    """
    A = np.asarray(A, dtype=float)
    A = (A + A.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.clip(eigenvalues, eps, None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


# ---------------------------------------------------------------------------
# Kept for backward compatibility with existing model code
# ---------------------------------------------------------------------------

def mat_sqrt(A: np.ndarray) -> np.ndarray:
    """Symmetric PSD matrix square root S such that S @ S = A (approx).

    Negative eigenvalues are clipped to zero.

    Parameters
    ----------
    A:
        Symmetric (k, k) PSD array.

    Returns
    -------
    np.ndarray
        Symmetric (k, k) square-root matrix.
    """
    A = np.asarray(A, dtype=float)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 0.0)
    return V @ np.diag(np.sqrt(w)) @ V.T


def mat_sqrt_inv(A: np.ndarray) -> np.ndarray:
    """Symmetric PD matrix inverse square root A^{-1/2}.

    Eigenvalues are clipped at 1e-14 to avoid division by zero.

    Parameters
    ----------
    A:
        Symmetric (k, k) PD array.

    Returns
    -------
    np.ndarray
        Symmetric (k, k) inverse square-root matrix.
    """
    A = np.asarray(A, dtype=float)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 1e-14)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def safe_inv(A: np.ndarray) -> np.ndarray:
    """Compute A^{-1} via Cholesky for PD matrix A; fall back to pinv.

    Parameters
    ----------
    A:
        (k, k) array, ideally positive definite.

    Returns
    -------
    np.ndarray
        (k, k) inverse or pseudo-inverse array.
    """
    A = np.asarray(A, dtype=float)
    try:
        c, low = cho_factor(A, lower=True, check_finite=False)
        return cho_solve((c, low), np.eye(A.shape[0]), check_finite=False)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


# ---------------------------------------------------------------------------
# Additional helpers required by models (NumPy versions)
# ---------------------------------------------------------------------------

def symmetrize(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part of A: (A + Aᵀ) / 2."""
    return (A + A.T) * 0.5


def ensure_pd(A: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Project A onto the PD cone by clipping eigenvalues to *epsilon*.

    Parameters
    ----------
    A:
        (k, k) symmetric array.
    epsilon:
        Minimum eigenvalue after clipping.
    """
    A_sym = symmetrize(np.asarray(A, dtype=float))
    vals, vecs = np.linalg.eigh(A_sym)
    vals = np.maximum(vals, epsilon)
    return (vecs * vals) @ vecs.T


def safe_logdet(A: np.ndarray) -> float:
    """log|A| via slogdet; returns np.inf if A is not positive definite."""
    sign, ld = np.linalg.slogdet(np.asarray(A, dtype=float))
    return float(ld) if sign > 0 else np.inf


def trace_prod(A: np.ndarray, B: np.ndarray) -> float:
    """Compute tr(A @ B) as ∑ᵢⱼ Aᵢⱼ Bⱼᵢ (works for symmetric B via element-wise)."""
    return float(np.einsum("ij,ji->", A, B))


# ---------------------------------------------------------------------------
# Numba-JIT variants — callable from @njit recursion kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def ensure_pd_numba(A: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Numba-compiled PD projection via eigenvalue clipping.

    Mirrors :func:`ensure_pd`; usable from ``@njit`` functions.
    """
    k = A.shape[0]
    A_sym = (A + A.T) * 0.5
    vals, vecs = np.linalg.eigh(A_sym)
    for i in range(k):
        if vals[i] < epsilon:
            vals[i] = epsilon
    return vecs @ np.diag(vals) @ vecs.T


@njit(cache=True)
def logdet_numba(A: np.ndarray) -> float:
    """Numba-compiled log|A|; returns 1e18 if not positive definite."""
    sign, ld = np.linalg.slogdet(A)
    if sign <= 0.0:
        return 1e18
    return ld


@njit(cache=True)
def trace_prod_numba(A: np.ndarray, B: np.ndarray) -> float:
    """Numba-compiled tr(A @ B) for symmetric B."""
    k = A.shape[0]
    val = 0.0
    for i in range(k):
        for j in range(k):
            val += A[i, j] * B[j, i]
    return val
