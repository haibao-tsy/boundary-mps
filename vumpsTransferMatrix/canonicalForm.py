import numpy as np
from .ncon import ncon
from numpy.linalg import norm
from scipy.linalg import qr, rq, svd
from scipy.sparse.linalg import LinearOperator, eigs

def qrpos(a: np.ndarray):
    """Reduced QR with nonnegative real diagonal of R"""
    q, r = qr(a, mode="economic")
    d = np.diag(r)
    phase = np.ones_like(d)
    np.divide(d, np.abs(d), out=phase, where=(d != 0))

    q = q * phase[None, :]          # Scale columns
    r = phase.conj()[:, None] * r   # Scale rows
    return q, r

def rqpos(a: np.ndarray):
    """Reduced RQ with nonnegative real diagonal of R"""
    r, q = rq(a, mode="economic")
    d = np.diag(r)
    phase = np.ones_like(d)
    np.divide(d, np.abs(d), out=phase, where=(d != 0))

    q = phase[:, None] * q          # Scale columns
    r = r * phase.conj()[None, :]   # Scale rows
    return r, q


def leftCanonicalQR(A: np.ndarray, tol=1e-14, max_iter=1e5):
    """Return the left-canonical tensor AL, gauge matrix L, and scale lam.

    L @ A = lam * AL @ L within numerical tolerance,
    with AL.conj().T @ AL = I and norm(L) = 1.
    For an injective MPS, lam**2 is the dominant transfer-matrix eigenvalue.
    """
    chi, d = A.shape[:2]
    L = np.eye(chi, dtype=np.complex128)
    L = L / norm(L)

    AL, L_new = qrpos(
        ncon(
            [L, A],
            [
                [-1, 1],
                [1, -2, -3]
            ]
        ).reshape(chi * d, chi)
    )
    AL = AL.reshape(chi, d, chi)
    lam = norm(L_new)
    L_new = L_new / lam
    convergence = norm(L_new - L)
    iter_count = 0

    while (convergence > tol):

        def matvec(v):
            v = v.reshape(chi, chi)
            result = ncon(
                [v, AL.conj(), A],
                [
                    [1, 2],
                    [1, 3, -1],
                    [2, 3, -2]
                ]
            )
            return result.reshape(chi * chi)

        transferMatrix = LinearOperator(
            shape=(chi * chi, chi * chi),
            matvec=matvec
        )

        _, eigenvectors = eigs(
            A=transferMatrix,
            k=1,
            maxiter=max_iter,
            tol=convergence/10
        )

        L_improved = eigenvectors[:, 0].reshape(chi, chi)
        _, L_improved = qrpos(L_improved)
        L_improved /= norm(L_improved)

        AL, L = qrpos(
            ncon(
                [L_improved, A],
                [
                    [-1, 1],
                    [1, -2, -3]
                ]
            ).reshape(chi * d, chi)
        )
        AL = AL.reshape(chi, d, chi)

        lam = norm(L)
        L = L / lam
        convergence = norm(L - L_improved)

        iter_count += 1
        if iter_count > max_iter:
            raise RuntimeError("Failed to converge!")

    return AL, L, lam

def rightCanonicalQR(A: np.ndarray, tol=1e-14, max_iter=1e5):
    """Return the right-canonical tensor AR, gauge matrix R, and scale lam.

    A @ R = lam * R @ AR within numerical tolerance,
    """
    chi, d = A.shape[:2]
    R = np.eye(chi, dtype=np.complex128)
    R = R / norm(R)

    R_new, AR = rqpos(
        ncon(
            [A, R],
            [
                [-1, -2, 1],
                [1, -3]
            ]
        ).reshape(chi, d * chi)
    )
    AR = AR.reshape(chi, d, chi)
    lam = norm(R_new)
    R_new = R_new / lam
    convergence = norm(R_new - R)
    iter_count = 0

    while (convergence > tol):

        def matvec(v):
            v = v.reshape(chi, chi)
            result = ncon(
                [v, AR.conj(), A],
                [
                    [1, 2],
                    [-2, 3, 2],
                    [-1, 3, 1]
                ]
            )
            return result.reshape(chi * chi)

        transferMatrix = LinearOperator(
            shape=(chi * chi, chi * chi),
            matvec=matvec
        )

        _, eigenvectors = eigs(
            A=transferMatrix,
            k=1,
            maxiter=max_iter,
            tol=convergence/10
        )

        R_improved = eigenvectors[:, 0].reshape(chi, chi)
        R_improved, _ = rqpos(R_improved)
        R_improved /= norm(R_improved)

        R, AR = rqpos(
            ncon(
                [A, R_improved],
                [
                    [-1, -2, 1],
                    [1, -3]
                ]
            ).reshape(chi, d * chi)
        )
        AR = AR.reshape(chi, d, chi)

        lam = norm(R)
        R = R / lam
        convergence = norm(R - R_improved)

        iter_count += 1
        if iter_count > max_iter:
            raise RuntimeError("Failed to converge!")

    return AR, R, lam

def mixedCanonicalQR(A: np.ndarray, tol=1e-14, max_iter=1e5):
    """Return a normalized mixed-canonical form of a uniform MPS tensor. Using single-layer algorithm 
    (QR decomposition with Arnoldi acceleration)

    Parameters
    ----------
    A : np.ndarray, shape (chi, d, chi)
        Input tensor with axes (left bond, physical index, right bond).
        Assumes an injective MPS.
    tol : float, optional
        Convergence tolerance for the left and right canonicalizations.
    max_iter : int, optional
        Iteration limit passed to both canonicalization routines.

    Returns
    -------
    AC : np.ndarray, shape (chi, d, chi)
        Center tensor satisfying AC[:, s, :] = AL[:, s, :] @ C
        = C @ AR[:, s, :] for every physical index s, within numerical
        tolerance.
    C : np.ndarray, shape (chi, chi)
        Diagonal matrix of nonnegative Schmidt coefficients in descending
        order, normalized so that norm(C) = 1 (Frobenius norm).
    AL : np.ndarray, shape (chi, d, chi)
        Left-canonical tensor satisfying sum_s AL_s.conj().T @ AL_s = I,
        where AL_s = AL[:, s, :].
    AR : np.ndarray, shape (chi, d, chi)
        Right-canonical tensor satisfying sum_s AR_s @ AR_s.conj().T = I,
        where AR_s = AR[:, s, :].

    Notes
    -----
    The input's overall scale is removed by canonicalization and is not
    returned. The returned tensors describe the normalized state.
    """
    AL, L, lam = leftCanonicalQR(A=A, tol=tol, max_iter=max_iter)
    AR, R, _ = rightCanonicalQR(A=A, tol=tol, max_iter=max_iter)

    C = L @ R

    U, S, V = svd(C)

    C = np.diag(S) / norm(S)

    AL = ncon(
        [U.conj().T, AL, U],
        [
            [-1, 1],
            [1, -2, 2],
            [2, -3]
        ]
    )

    AR = ncon(
        [V, AR, V.conj().T],
        [
            [-1, 1],
            [1, -2, 2],
            [2, -3]
        ]
    )

    AC = ncon(
        [AL, C],
        [
            [-1, -2, 1],
            [1, -3]
        ]
    )

    return AC, C, AL, AR
