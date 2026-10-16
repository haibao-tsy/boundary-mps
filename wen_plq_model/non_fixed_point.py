import numpy as np
from tensor_equation_solver import tensor_equation_solver, kron_all
from vumpsTransferMatrix.vumps import vumpsMPO, normalizedEnvironment
from vumpsTransferMatrix.ncon import ncon

I = np.array([[1, 0],[0, 1]],dtype=complex)
X = np.array([[0, 1],[1, 0]],dtype=complex)
Y = np.array([[0, -1.0j],[1.0j, 0]],dtype=complex)
Z = np.array([[1, 0],[0, -1] ],dtype=complex)

chi = 2
pepsA, _ = tensor_equation_solver(
[ 
    [X, I, I.T, X.T, X],
    [I, Y, Y.T, I.T, X],
    [I, I, X.T, Y.T, Z],
    [Y, X, I.T, I.T, Z],
    [Y, I, I.T, Y.T, I]
]
)

pepsB, _ = tensor_equation_solver(
[ 
    [X, I, I.T, X.T, X],
    [I, Y, Y.T, I.T, X],
    [I, I, X.T, Y.T, Z],
    [Y, X, I.T, I.T, Z],
    [X, Y, I.T, I.T, I]
]
)

transferMatrix1 = ncon(
    [pepsA, pepsB, pepsB.conj(), pepsA.conj()],
    [
        [-6, 3, -10, -12, 4],
        [-5, -2, -9, 3, 1],
        [-3, -1, -7, 2, 1],
        [-4, 2, -8, -11, 4]
    ]
)

transferMatrix1 = transferMatrix1.reshape(chi**2, chi**4, chi**4, chi**2)

transferMatrix2 = ncon(
    [pepsB, pepsA, pepsA.conj(), pepsB.conj()],
    [
        [-6, 3, -10, -12, 4],
        [-5, -2, -9, 3, 1],
        [-3, -1, -7, 2, 1],
        [-4, 2, -8, -11, 4]
    ]
)

transferMatrix2 = transferMatrix2.reshape(chi**2, chi**4, chi**4, chi**2)
# the configuration of transfer matrix is :
#                 A B
#                 B A
transferMatrixPEPS = ncon(
    [transferMatrix1, transferMatrix2],
    [
        [-1, -3, 1, -5],
        [-2, 1, -4, -6]
    ]
)

transferMatrixPEPS = transferMatrixPEPS.reshape((chi**4, ) * 4)

transferMatrixPEPS = transferMatrixPEPS / np.linalg.norm(transferMatrixPEPS)


chiMPS = 10
AC_LEFT, C_LEFT, AL_LEFT, AR_LEFT, leftenv_LEFT, rightenv_LEFT, lam = vumpsMPO(
    mpoTensor=transferMatrixPEPS,
    chi = chiMPS,
    maxIter = 10000,
    tol=1e-6
)

AL_LEFT = AL_LEFT.reshape(chiMPS, chi, chi, chi, chi, chiMPS)

AC_RIGHT, C_RIGHT, AL_RIGHT, AR_RIGHT, leftenv_RIGHT, rightenv_RIGHT, _ = vumpsMPO(
    mpoTensor=transferMatrixPEPS.T,
    chi = chiMPS,
    maxIter = 1000,
    tol=1e-6
)

AL_RIGHT = AL_RIGHT.reshape(chiMPS, chi, chi, chi, chi, chiMPS)
mpu, _= tensor_equation_solver(
    [
        [I, I, X.T, X.T],
        [Z, X, I.T, X.T],
        [Y, I, Y.T, Z.T],
        [Y, Y, I.T, I.T]
    ]
)

D_MPU = mpu.shape[0]

transferMatrixMPS = ncon(
    [AL_LEFT, AL_RIGHT],
    [
        [-1, 1, 3, 2, 4, -3],
        [-2, 2, 4, 1, 3, -4]
    ]
)

transferMatrixMPS = transferMatrixMPS.reshape(chiMPS * chiMPS, chiMPS * chiMPS)

twistedTransferMatrixMPS = ncon(
    [AL_RIGHT, mpu, mpu, AL_LEFT],
    [
        [-1, 1, 5, 2, 6, -4],
        [-2, 2, 3, 4],
        [4, 6, 7 ,-5],
        [-3, 3, 7, 1, 5, -6]
    ]
)

twistedTransferMatrixMPS = twistedTransferMatrixMPS.reshape(D_MPU * chiMPS * chiMPS, D_MPU * chiMPS * chiMPS)


eig_vals = np.linalg.eigvals(twistedTransferMatrixMPS)
eig_vals = eig_vals / eig_vals[np.argmax(np.abs(eig_vals))]
eig_vals = eig_vals[np.argsort(np.abs(eig_vals))]
print("twisted = ",np.round(eig_vals, decimals=3)[-4:])

eig_vals = np.linalg.eigvals(transferMatrixMPS)
eig_vals = eig_vals / eig_vals[np.argmax(np.abs(eig_vals))]
eig_vals = eig_vals[np.argsort(np.abs(eig_vals))]
print("untwisted = " ,np.round(eig_vals[-4:], decimals=3))
