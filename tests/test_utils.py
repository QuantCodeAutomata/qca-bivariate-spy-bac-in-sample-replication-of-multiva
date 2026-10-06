"""
Tests for src/utils/matrix.py and src/utils/linalg.py.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.utils.matrix import (vech, ivech, vec, duplication_matrix,
                               elimination_matrix, vech_size)
from src.utils.linalg import (mat_sqrt, stable_log_det, safe_inv,
                               regularize_matrix, is_positive_definite)


# ---------------------------------------------------------------------------
# vech / ivech round-trip
# ---------------------------------------------------------------------------

def test_vech_ivech_roundtrip():
    """vech then ivech reconstructs the original symmetric matrix."""
    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    v = vech(A)
    A_reconstructed = ivech(v, k=2)
    assert np.allclose(A, A_reconstructed), \
        f"ivech(vech(A)) != A:\n{A}\nvs\n{A_reconstructed}"


def test_vech_ivech_roundtrip_3x3():
    """Round-trip for a 3×3 PD matrix."""
    rng = np.random.default_rng(7)
    B = rng.standard_normal((3, 3))
    A = B @ B.T + np.eye(3)   # guaranteed PD
    v = vech(A)
    A_reconstructed = ivech(v, k=3)
    assert np.allclose(A, A_reconstructed, atol=1e-14)


# ---------------------------------------------------------------------------
# vech size
# ---------------------------------------------------------------------------

def test_vech_size_k2():
    """k=2 symmetric matrix has 3 unique lower-triangular elements."""
    A = np.eye(2)
    assert len(vech(A)) == 3
    assert vech_size(2) == 3


def test_vech_size_k3():
    """k=3 gives 6 elements."""
    A = np.eye(3)
    assert len(vech(A)) == 6
    assert vech_size(3) == 6


def test_vech_size_k10():
    """k=10 gives 55 elements."""
    A = np.eye(10)
    assert len(vech(A)) == 55
    assert vech_size(10) == 55


# ---------------------------------------------------------------------------
# Duplication matrix
# ---------------------------------------------------------------------------

def _check_duplication(k):
    D = duplication_matrix(k)
    rng = np.random.default_rng(42)
    B = rng.standard_normal((k, k))
    A = 0.5 * (B + B.T)   # symmetric
    # D @ vech(A) should equal vec(A) (column-major vectorisation)
    lhs = D @ vech(A)
    rhs = vec(A)
    assert np.allclose(lhs, rhs, atol=1e-14), \
        f"D_{k} @ vech(A) != vec(A) for k={k}"


def test_duplication_matrix_k2():
    _check_duplication(2)


def test_duplication_matrix_k3():
    _check_duplication(3)


# ---------------------------------------------------------------------------
# is_positive_definite
# ---------------------------------------------------------------------------

def test_is_positive_definite_true():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    assert is_positive_definite(A) is True


def test_is_positive_definite_false_indefinite():
    A = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert is_positive_definite(A) is False


def test_is_positive_definite_false_singular():
    A = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert is_positive_definite(A) is False


def test_is_positive_definite_non_square():
    A = np.array([[1.0, 2.0, 3.0]])
    assert is_positive_definite(A) is False


# ---------------------------------------------------------------------------
# mat_sqrt
# ---------------------------------------------------------------------------

def test_mat_sqrt_squares_to_original():
    """S = sqrt(A)  =>  S @ S ≈ A for a known PD matrix."""
    A = np.array([[5.0, 2.0], [2.0, 3.0]])
    S = mat_sqrt(A)
    assert np.allclose(S @ S, A, atol=1e-12), \
        f"mat_sqrt(A)^2 != A:\n{S @ S}\nvs\n{A}"


def test_mat_sqrt_is_symmetric():
    """The symmetric square root is itself symmetric."""
    A = np.array([[4.0, 1.5], [1.5, 2.0]])
    S = mat_sqrt(A)
    assert np.allclose(S, S.T, atol=1e-14)


def test_mat_sqrt_pd():
    """The square root of a PD matrix is PD."""
    rng = np.random.default_rng(99)
    B = rng.standard_normal((4, 4))
    A = B @ B.T + np.eye(4)
    S = mat_sqrt(A)
    assert is_positive_definite(S)


# ---------------------------------------------------------------------------
# stable_log_det
# ---------------------------------------------------------------------------

def test_stable_log_det_known():
    """log|diag(2, 3)| = log(2) + log(3)."""
    A = np.diag([2.0, 3.0])
    expected = np.log(2.0) + np.log(3.0)
    result = stable_log_det(A)
    assert abs(result - expected) < 1e-12, f"{result} != {expected}"


def test_stable_log_det_matches_numpy():
    """stable_log_det agrees with numpy slogdet for a random PD matrix."""
    rng = np.random.default_rng(55)
    B = rng.standard_normal((5, 5))
    A = B @ B.T + 3 * np.eye(5)
    _, np_logdet = np.linalg.slogdet(A)
    our_logdet = stable_log_det(A)
    assert abs(our_logdet - np_logdet) < 1e-10, \
        f"stable_log_det={our_logdet:.10f} vs numpy={np_logdet:.10f}"


def test_stable_log_det_identity():
    """log|I_k| = 0 for any k."""
    for k in (2, 5, 10):
        assert abs(stable_log_det(np.eye(k))) < 1e-14
