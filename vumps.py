import numpy as np
from scipy.linalg import polar
from ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigs
from canonicalForm import mixedCanonicalQR

# chi is the dimension of the MPS internal leg
# d is the dimension of the MPS physical leg
# D is the dimension of the MPO internal leg

def leftEnvironment(mpoTensor, mpsLeft, tol=1e-14):
    chi = mpsLeft.shape[0]
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
    mpo_lam = eigenvalues[0]

    return lE, mpo_lam

def rightEnvironment(mpoTensor, mpsRight, tol=1e-14):
    chi = mpsRight.shape[0]
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
    mpo_lam = eigenvalues[0]
    return rE, mpo_lam


def hamiltonianHAC(mpoTensor, leftenv, rightenv, mpo_lam):
    HAC = ncon(
        [leftenv, mpoTensor, rightenv],
        [
            [-1, 1, -4],
            [1, -2, -5, 2],
            [-3, 2, -6]
        ]
    )

    return HAC / mpo_lam

def hamiltonianHC(leftenv, rightenv):
    HC = ncon(
        [leftenv, rightenv],
        [
            [-1, 1, -3],
            [-2, 1, -4]
        ]
    )

    return HC

def updateAC(hamiltonianHAC, tol=1e-14):
    chi, d = hamiltonianHAC.shape[:2]
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
    chi = hamiltonianHC.shape[0]
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

    return eigenvector[:, 0].reshape(chi, chi), eigenvalues[0]

def recoverMixedCanonical(AC, C):
    chi, d = AC.shape[: 2]
    U_AC_left, _ = polar(
        AC.reshape(chi * d, chi), side="right"
    )
    U_AC_right, _ = polar(
        AC.reshape(chi, d * chi), side="left"
    )
    U_C, _ = polar(C)

    AL = (U_AC_left @ U_C.conj().T).reshape(chi, d, chi)
    AR = (U_C.conj().T @ U_AC_right).reshape(chi, d, chi)

    return AL, AR

chi = 30     # mps bond dimension
D = 30       # mpo bond dimension
d = 30       # phyical leg dimension

A = np.random.rand(chi, d, chi)
mpoTensor = np.random.rand(D, d, d, D)

AC, C, AL, AR = mixedCanonicalQR(A)

_, lam1 = leftEnvironment(mpoTensor=mpoTensor, mpsLeft=AL)
_, lam2 = rightEnvironment(mpoTensor=mpoTensor, mpsRight=AR)

print(lam1 - lam2)
