import numpy as np
from scipy.linalg import sqrtm, polar
from ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigs

def leftCanonical(A: np.ndarray, tol=1e-14):
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