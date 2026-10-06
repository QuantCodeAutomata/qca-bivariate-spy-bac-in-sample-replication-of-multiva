"""Matrix utilities for multivariate volatility models.

Provides vech/ivech, duplication and elimination matrices, symmetric square
root and inverse square root, and a symmetrize helper.  All operations follow
the Magnus & Neudecker (1988) lower-triangular, column-by-column vech
convention.
"""

from __future__ import annotations

import numpy as np


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Return (A + A.T) / 2.

    Parameters
    ----------
    A:
        Square (k, k) array.

    Returns
    -------
    np.ndarray
        Symmetric (k, k) array.
    """
    return (A + A.T) * 0.5


def vech(A: np.ndarray) -> np.ndarray:
    """Vectorise the lower triangle of a symmetric matrix.

    Extracts elements A[i, j] for i >= j, ordered column by column
    (j = 0, 1, ..., k-1; i = j, j+1, ..., k-1), following the Magnus &
    Neudecker (1988) convention.

    Parameters
    ----------
    A:
        Square (k, k) array.  Only the lower-triangular part is read.

    Returns
    -------
    np.ndarray
        1-D array of length k*(k+1)//2.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1., 0.], [3., 4.]])
    >>> vech(A)
    array([1., 3., 4.])
    """
    A = np.asarray(A, dtype=float)
    k = A.shape[0]
    rows, cols = zip(*[(i, j) for j in range(k) for i in range(j, k)])
    return A[np.array(rows), np.array(cols)]


def ivech(v: np.ndarray, k: int | None = None) -> np.ndarray:
    """Reconstruct a symmetric matrix from its vech vector.

    Inverse of :func:`vech`.  Sets both A[i, j] and A[j, i] to v[s].

    Parameters
    ----------
    v:
        1-D array of length k*(k+1)//2.
    k:
        Dimension of the output matrix.  If *None*, inferred from len(v).

    Returns
    -------
    np.ndarray
        Symmetric (k, k) array.

    Examples
    --------
    >>> import numpy as np
    >>> ivech(np.array([1., 3., 4.]), k=2)
    array([[1., 3.],
           [3., 4.]])
    """
    v = np.asarray(v, dtype=float)
    if k is None:
        k = int(round((-1.0 + (1.0 + 8.0 * len(v)) ** 0.5) / 2.0))
    A = np.zeros((k, k), dtype=float)
    pairs = [(i, j) for j in range(k) for i in range(j, k)]
    rows = np.array([p[0] for p in pairs])
    cols = np.array([p[1] for p in pairs])
    A[rows, cols] = v
    A[cols, rows] = v
    return A


def vec(A: np.ndarray) -> np.ndarray:
    """Column-major vectorisation vec(A).

    Parameters
    ----------
    A:
        (m, n) array.

    Returns
    -------
    np.ndarray
        1-D array of length m*n, columns stacked top-to-bottom.
    """
    return np.asarray(A).ravel(order="F")


def duplication_matrix(k: int) -> np.ndarray:
    """Return the k^2 x (k*(k+1)//2) duplication matrix D_k.

    D_k satisfies  D_k @ vech(A) = vec(A)  for any symmetric (k, k) matrix A,
    where vec stacks columns top-to-bottom.

    Parameters
    ----------
    k:
        Matrix dimension.

    Returns
    -------
    np.ndarray
        Float array of shape (k*k, k*(k+1)//2).

    Examples
    --------
    >>> import numpy as np
    >>> D = duplication_matrix(2)
    >>> A = np.array([[1., 2.], [2., 3.]])
    >>> np.allclose(D @ vech(A), vec(A))
    True
    """
    m = k * (k + 1) // 2
    D = np.zeros((k * k, m), dtype=float)

    # Build lookup: (row >= col) -> column-major vech index
    vech_lookup: dict[tuple[int, int], int] = {}
    s = 0
    for j in range(k):
        for i in range(j, k):
            vech_lookup[(i, j)] = s
            s += 1

    # vec position p = j*k + i corresponds to element A[i, j]
    for j in range(k):
        for i in range(k):
            p = j * k + i
            r, c = (i, j) if i >= j else (j, i)   # lower-tri key
            D[p, vech_lookup[(r, c)]] = 1.0

    return D


def elimination_matrix(k: int) -> np.ndarray:
    """Return the (k*(k+1)//2) x k^2 elimination matrix L_k.

    L_k satisfies  L_k @ vec(A) = vech(A)  for any symmetric (k, k) matrix A.

    Parameters
    ----------
    k:
        Matrix dimension.

    Returns
    -------
    np.ndarray
        Float array of shape (k*(k+1)//2, k*k).

    Examples
    --------
    >>> import numpy as np
    >>> L = elimination_matrix(2)
    >>> A = np.array([[1., 2.], [2., 3.]])
    >>> np.allclose(L @ vec(A), vech(A))
    True
    """
    m = k * (k + 1) // 2
    L = np.zeros((m, k * k), dtype=float)
    s = 0
    for j in range(k):
        for i in range(j, k):
            p = j * k + i   # vec index for element A[i, j]
            L[s, p] = 1.0
            s += 1
    return L


def commutation_matrix(m: int, n: int) -> np.ndarray:
    """Return the mn x mn commutation matrix K_{m,n}.

    K_{m,n} satisfies  K_{m,n} @ vec(A) = vec(A.T)  for any (m, n) matrix A.

    Parameters
    ----------
    m:
        Number of rows of A.
    n:
        Number of columns of A.

    Returns
    -------
    np.ndarray
        Float array of shape (m*n, m*n).
    """
    K = np.zeros((m * n, m * n), dtype=float)
    for i in range(m):
        for j in range(n):
            K[j * m + i, i * n + j] = 1.0
    return K


def mat_sqrt(A: np.ndarray) -> np.ndarray:
    """Symmetric positive-semidefinite matrix square root.

    Computes S such that S @ S = A (approximately) via eigendecomposition.
    Negative eigenvalues are clipped to zero, making the function safe for
    near-PSD matrices with small numerical negatives.

    Parameters
    ----------
    A:
        Symmetric (k, k) PSD array.

    Returns
    -------
    np.ndarray
        Symmetric (k, k) square-root matrix.
    """
    A = symmetrize(A)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def mat_inv_sqrt(A: np.ndarray) -> np.ndarray:
    """Symmetric positive-definite matrix inverse square root A^{-1/2}.

    Computes S such that S @ A @ S = I (approximately), via eigendecomposition.
    Eigenvalues are clipped at 1e-12 to avoid division by zero on near-singular
    inputs.

    Parameters
    ----------
    A:
        Symmetric (k, k) PD array.

    Returns
    -------
    np.ndarray
        Symmetric (k, k) inverse square-root matrix.
    """
    A = symmetrize(A)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T


def vech_size(k: int) -> int:
    """Return the length of vech(A) for a k x k symmetric matrix: k*(k+1)//2."""
    return k * (k + 1) // 2


def lower_chol_from_vech(v: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct a lower-triangular (k, k) matrix L from vech(L).

    Uses the same column-major ordering as :func:`vech`:
    elements are filled column by column, (j,j), (j+1,j), …, (k-1,j).

    Parameters
    ----------
    v:
        1-D array of length k*(k+1)//2 holding the lower-triangular entries.
    k:
        Dimension of the output matrix.

    Returns
    -------
    np.ndarray
        Lower-triangular (k, k) float array.

    Examples
    --------
    >>> import numpy as np
    >>> lower_chol_from_vech(np.array([1., 2., 3.]), k=2)
    array([[1., 0.],
           [2., 3.]])
    """
    v = np.asarray(v, dtype=float)
    L = np.zeros((k, k), dtype=float)
    s = 0
    for j in range(k):
        for i in range(j, k):
            L[i, j] = v[s]
            s += 1
    return L


def outer_product(x: np.ndarray) -> np.ndarray:
    """Compute the outer product x xᵀ.

    Parameters
    ----------
    x:
        1-D array of length k.

    Returns
    -------
    np.ndarray
        (k, k) symmetric array.
    """
    return np.outer(x, x)
