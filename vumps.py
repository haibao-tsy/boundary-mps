import numpy as np
from scipy.linalg import sqrtm
from ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigs

def leftEnvironment(mpoTensor, mpsLeft, tol=1e-14):
    chi = mpsLeft.shape[0]
    d = mpsLeft.shape[1]
    D = mpoTensor.shape[0]
    mpsLeftConj = mpsLeft.conj()

    def matvec(v):
        v = v.reshape(chi, D, chi)

        result = ncon(
            [mpsLeftConj, mpoTensor, mpsLeft, v],
            [
                [1, 4, -1],
                [2, 4, 5, -2],
                [3, 5, -3],
                [1, 2, 3]
            ]
        )

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chi * D * chi, chi * D * chi),
        matvec=matvec,
        dtype=np.complex128,
    )

    eigenvalues, eigenvector = eigs(transferMatrix,k=1, which="LM", tol=tol)

    lE = eigenvector[:, 0].reshape(chi, D, chi)

    return lE, eigenvalues[0]

def rightEnvironment(mpoTensor, mpsRight, tol=1e-14):
    chi = mpsRight.shape[0]
    d = mpsRight.shape[1]
    D = mpoTensor.shape[0]
    mpsRightConj = mpsRight.conj()

    def matvec(v):
        v = v.reshape(chi, D, chi)

        result = ncon(
            [mpsRightConj, mpoTensor, mpsRight, v],
            [
                [-1, 4, 1],
                [-2, 4, 5, 2],
                [-3, 5, 3],
                [1, 2, 3]
            ]
        )

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chi * D * chi, chi * D * chi),
        matvec=matvec,
        dtype=np.complex128,
    )

    eigenvalues, eigenvector = eigs(transferMatrix,k=1, which="LM", tol=tol)

    rE = eigenvector[:, 0].reshape(chi, D, chi)

    return rE, eigenvalues[0]

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
    _, sqrt_l, lam = leftCanonical(A)
    _, sqrt_r, _ = rightCanonical(A)

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
    C = C / np.trace(C.conj().T @ C)
    A_C = A_C / np.trace(C.conj().T @ C)

    return A_C, C

def hamiltonianHAC(mpoTensor, mpsLeft, mpsRight):
    leftenv, _ = leftEnvironment(mpoTensor=mpoTensor, mpsLeft=mpsLeft)
    rightenv, _ = rightEnvironment(mpoTensor=mpoTensor, mpsRight=mpsRight)
    HAC = ncon(
        [leftenv, mpoTensor, rightenv],
        [
            [-1, 1, -4],
            [1, -2, -5, 2],
            [-3, 2, -6]
        ]
    )

    return HAC

def hamiltonianHC(mpoTensor, mpsLeft, mpsRight):
    leftenv, _ = leftEnvironment(mpoTensor=mpoTensor, mpsLeft=mpsLeft)
    rightenv, _ = rightEnvironment(mpoTensor=mpoTensor, mpsRight=mpsRight)
    HC = ncon(
        [leftenv, rightenv],
        [
            [-1, 1, -3],
            [-2, 1, -4]
        ]
    )

    return HC

def updateAC(hamiltonianHAC, tol=1e-14):
    def matvec(v):
        v = v.reshape(chi, d, chi)
        result = ncon(
            [hamiltonianHAC, v],
            [
                [-1, -2, -3, 1, 2, 3],
                [1, 2, 3]
            ]
        )
        return result.ravel()

    HAC = LinearOperator(
        shape=(chi * d * chi, chi * d * chi),
        matvec=matvec,
        dtype=np.complex128
    )

    eigenvalues, eigenvector = eigs(
        A=HAC,
        k=1,
        which="LM",
        tol=tol
    )

    return eigenvector[:, 0].reshape(chi, d, chi), eigenvalues[0]
    
def updateC(hamiltonianHC, tol=1e-14):
    def matvec(v):
        v = v.reshape(chi, chi)
        result = ncon(
            [hamiltonianHC, v],
            [
                [-1, -2, 1, 2],
                [1, 2]
            ]
        )
        return result.ravel()

    HC = LinearOperator(
        shape=(chi * chi, chi * chi),
        matvec=matvec,
        dtype=np.complex128
    )

    eigenvalues, eigenvector = eigs(
        A=HC,
        k=1,
        which="LM",
        tol=tol
    )

    return eigenvector[:, 0].reshape(chi, d, chi), eigenvalues[0]


chi = 6     # mps bond dimension
D = 6       # mpo bond dimension
d = 4       # phyical leg dimension

A = np.random.rand(chi, d, chi)
mpoTensor = np.random.rand(D, d, d, D)

A_L, _, _ = leftCanonical(A)
A_R, _, _= rightCanonical(A)
A_C, C = mixedCanonical(A)

H_A_C = hamiltonianHAC(mpoTensor=mpoTensor, mpsLeft=A_L, mpsRight=A_R)
H_C = hamiltonianHC(mpoTensor=mpoTensor, mpsLeft=A_L, mpsRight=A_R)

A_C_prime, mu_AC = updateAC(H_A_C)
C_prime, mu_C= updateAC(H_C)
