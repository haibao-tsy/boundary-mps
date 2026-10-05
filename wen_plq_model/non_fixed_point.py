import numpy as np
from tensor_equation_solver import tensor_equation_solver
from vumpsTransferMatrix.vumps import vumpsMPO
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
]
)

pepsB, _ = tensor_equation_solver(
[ 
    [X, I, I.T, X.T, X],
    [I, Y, Y.T, I.T, X],
    [I, I, X.T, Y.T, Z],
    [Y, X, I.T, I.T, Z],
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
transferMatrix = ncon(
    [transferMatrix1, transferMatrix2],
    [
        [-1, -3, 1, -5],
        [-2, 1, -4, -6]
    ]
)

transferMatrix = transferMatrix.reshape((chi**4, ) * 4)

transferMatrix = transferMatrix / np.linalg.norm(transferMatrix)

chiMPS = 15
AC, C, AL, AR, leftenv, rightenv, lam = vumpsMPO(
    mpoTensor=transferMatrix,
    chi = chiMPS,
    maxIter = 1000,
    tol=1e-7
)

AL = AL.reshape(chiMPS, chi, chi, chi, chi, chiMPS)

mpu, _= tensor_equation_solver(
    [
        [I, I, X.T, X.T],
        [Z, X, I.T, X.T],
        [Y, I, Y.T, Z.T],
        [Y, Y, I.T, I.T]
    ]
)

D_MPU = mpu.shape[0]
twistedTransferMatrix = ncon(
    [AL, mpu, mpu],
    [
        [-1, 1, 3, 2, 4, -3],
        [-2, 2, 1, 6],
        [6, 4, 3, -4]
    ]
)

twistedTransferMatrix = twistedTransferMatrix.reshape(D_MPU * chiMPS, D_MPU * chiMPS)

eig_vals = np.linalg.eigvals(twistedTransferMatrix)
eig_vals = eig_vals / eig_vals[0]
eig_vals = eig_vals[np.argsort(np.abs(eig_vals))[::-1]]
print(np.round(eig_vals, decimals=3))
