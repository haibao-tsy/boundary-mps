"""Canonicalize a uniform MPS using square roots of transfer fixed points.

All MPS tensors have shape (chi, d, chi), with axes (left bond, physical
index, right bond). For A_s = A[:, s, :], the transfer maps are
T_L(X) = sum_s A_s.conj().T @ X @ A_s and
T_R(X) = sum_s A_s @ X @ A_s.conj().T.

The routines assume an injective MPS with a nonzero dominant transfer
eigenvalue and positive-definite left and right fixed points. Each fixed
point is normalized to unit trace and Hermitian-symmetrized before taking
its matrix square root. Explicit inversion of these square roots can
amplify numerical errors when the fixed points are ill-conditioned; no
eigenvalue cutoff or regularization is applied. The QR-based routines in
canonicalForm.py provide an alternative that avoids these inverses.

The current sparse eigensolver requires chi >= 2. mixedCanonical removes
the transfer eigenvalue scale but does not normalize the center matrix
or transform it to a diagonal Schmidt gauge.
"""

import numpy as np
from scipy.linalg import sqrtm, polar
from ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigs

def leftCanonical(A: np.ndarray, tol=1e-14):
    """Construct a left-canonical tensor from the left transfer fixed point.

    Parameters
    ----------
    A : ndarray, shape (chi, d, chi)
        Injective uniform MPS tensor with axes (left bond, physical index,
        right bond). The dominant left fixed point must be invertible.
    tol : float, optional
        Relative tolerance passed to the transfer-matrix eigensolver.

    Returns
    -------
    A_L : ndarray, shape (chi, d, chi)
        Tensor defined by A_L[:, s, :] = sqrt_l @ A[:, s, :]
        @ inv(sqrt_l) / sqrt(lam). For an accurate positive-definite fixed
        point, sum_s A_L[:, s, :].conj().T @ A_L[:, s, :] = I within
        numerical precision.
    sqrt_l : ndarray, shape (chi, chi)
        Matrix square root of the trace-normalized, Hermitian-symmetrized
        dominant fixed point l of T_L(X) = sum_s A_s.conj().T @ X @ A_s.
    lam : complex
        Dominant transfer eigenvalue, selected by largest magnitude.
        Its square root is the scale removed from each site tensor.

    Notes
    -----
    A_s denotes A[:, s, :]. The gauge relation is
    sqrt_l @ A_s = sqrt(lam) * A_L[:, s, :] @ sqrt_l.
    Small eigenvalues of l make the explicit inverse numerically unstable;
    zero eigenvalues can make inversion fail. No positivity correction
    beyond Hermitian symmetrization is performed.
    """
    chi = A.shape[0]
    Aconj = A.conj()
    def matvec(v):
        v = v.reshape(chi, chi)

        result = ncon(
            [Aconj, A, v],
            [
                [1, 3, -1],
                [2, 3, -2],
                [1, 2]
            ]
        )

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chi * chi, chi * chi),
        matvec=matvec,
        dtype=np.complex128,
    )

    eigenvalues, eigenvector = eigs(
        transferMatrix, k=1, which="LM",tol=tol
    )

    l = eigenvector[:, 0].reshape(chi, chi)
    l = l / np.trace(l)
    l = (l + l.conj().T) / 2
    lam = eigenvalues[0]
    sqrt_l = sqrtm(l)

    A_L = ncon(
        [sqrt_l, A, np.linalg.inv(sqrt_l)],
        [
            [-1, 1],
            [1, -2, 2],
            [2, -3]
        ]
    )

    A_L = A_L / np.sqrt(lam)

    return A_L, sqrt_l, lam

def rightCanonical(A: np.ndarray, tol=1e-14):
    """Construct a right-canonical tensor from the right transfer fixed point.

    Parameters
    ----------
    A : ndarray, shape (chi, d, chi)
        Injective uniform MPS tensor with axes (left bond, physical index,
        right bond). The dominant right fixed point must be invertible.
    tol : float, optional
        Relative tolerance passed to the transfer-matrix eigensolver.

    Returns
    -------
    A_R : ndarray, shape (chi, d, chi)
        Tensor defined by A_R[:, s, :] = inv(sqrt_r) @ A[:, s, :]
        @ sqrt_r / sqrt(lam). For an accurate positive-definite fixed
        point, sum_s A_R[:, s, :] @ A_R[:, s, :].conj().T = I within
        numerical precision.
    sqrt_r : ndarray, shape (chi, chi)
        Matrix square root of the trace-normalized, Hermitian-symmetrized
        dominant fixed point r of T_R(X) = sum_s A_s @ X @ A_s.conj().T.
    lam : complex
        Dominant transfer eigenvalue, selected by largest magnitude.
        Its square root is the scale removed from each site tensor.

    Notes
    -----
    A_s denotes A[:, s, :]. The gauge relation is
    A_s @ sqrt_r = sqrt(lam) * sqrt_r @ A_R[:, s, :].
    Small eigenvalues of r make the explicit inverse numerically unstable;
    zero eigenvalues can make inversion fail. No positivity correction
    beyond Hermitian symmetrization is performed.
    """
    chi = A.shape[0]
    Aconj = A.conj()
    def matvec(v):
        v = v.reshape(chi, chi)

        result = ncon(
            [Aconj, A, v],
            [
                [-2, 1, 2],
                [-1, 1, 3],
                [3, 2]
            ]
        )

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chi * chi, chi * chi),
        matvec=matvec,
        dtype=np.complex128,
    )

    eigenvalues, eigenvector = eigs(
        transferMatrix, k=1, which="LM", tol=tol
    )

    r = eigenvector[:, 0].reshape(chi, chi)
    r = r/np.trace(r)
    r = (r + r.conj().T) / 2
    sqrt_r = sqrtm(r)
    lam = eigenvalues[0]

    A_R = ncon(
        [np.linalg.inv(sqrt_r), A, sqrt_r],
        [
            [-1, 1],
            [1, -2, 2],
            [2, -3]
        ]
    )

    A_R = A_R / np.sqrt(lam)

    return A_R, sqrt_r, lam

def mixedCanonical(A, tol=1e-14):
    """Construct the mixed-canonical center tensor and connecting matrix.

    Parameters
    ----------
    A : ndarray, shape (chi, d, chi)
        Injective uniform MPS tensor with axes (left bond, physical index,
        right bond). Both dominant transfer fixed points must be invertible.
    tol : float, optional
        Relative tolerance passed to the left and right canonicalization
        eigensolvers.

    Returns
    -------
    A_C : ndarray, shape (chi, d, chi)
        Center tensor with A_C[:, s, :] = sqrt_l @ A[:, s, :] @ sqrt_r
        / sqrt(lam), using the dominant left-channel eigenvalue lam.
    C : ndarray, shape (chi, chi)
        Connecting matrix sqrt_l @ sqrt_r. It is generally non-diagonal
        and is not normalized to unit Frobenius norm.

    Notes
    -----
    With accurate fixed points, A_C[:, s, :] = A_L[:, s, :] @ C
    = C @ A_R[:, s, :] for the corresponding canonical tensors. A_L,
    A_R, and lam are computed internally but are not returned.

    The fixed points l and r have unit trace individually. In exact
    arithmetic, norm(A_C)**2 = norm(C)**2 = trace(l @ r), where norm is
    the Frobenius norm. To obtain unit-norm centers, divide both returned
    tensors by norm(C). This routine inherits the numerical limitations
    of leftCanonical and rightCanonical.
    """
    _, sqrt_l, lam = leftCanonical(A, tol=tol)
    _, sqrt_r, _ = rightCanonical(A, tol=tol)

    A_C = ncon(
        [sqrt_l, A, sqrt_r],
        [
            [-1, 1],
            [1, -2, 2],
            [2, -3]
        ]
    )

    A_C = A_C / np.sqrt(lam)

    C = sqrt_l @ sqrt_r

    return A_C, C
