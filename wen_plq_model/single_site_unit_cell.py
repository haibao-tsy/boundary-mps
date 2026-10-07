import numpy as np
from tensor_equation_solver import tensor_equation_solver, kron_all
from vumpsTransferMatrix.vumps import vumpsMPO
from vumpsTransferMatrix.ncon import ncon

I = np.array([[1, 0],[0, 1]],dtype=complex)
X = np.array([[0, 1],[1, 0]],dtype=complex)
Y = np.array([[0, -1.0j],[1.0j, 0]],dtype=complex)
Z = np.array([[1, 0],[0, -1] ],dtype=complex)

d = 2
chi = 2

peps, _ = tensor_equation_solver(
    [
        [X, I, I.T, X.T, X],
        [I, Y, Y.T, I.T, X],
        [I, I, X.T, Y.T, Z],
        [Y, X, I.T, I.T, Z],
    ]
)

transferMatrix = ncon(
    [peps.conj(), peps],
    [
        [-3, -1, -5, -7, 1],
        [-4, -2, -6, -8, 1]
    ]
)

transferMatrix = transferMatrix / np.linalg.norm(transferMatrix)

transferMatrix = transferMatrix.reshape( (chi ** 2, )*4 )

chiMPS = 10
AC, C, AL, AR, leftenv, rightenv, lam = vumpsMPO(
    mpoTensor=transferMatrix,
    chi = chiMPS,
    maxIter = 1000,
    tol=1e-10
)

AL = AL.reshape(chiMPS, chi, chi, chiMPS)

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
    [AL, mpu],
    [
        [-2, 2, 1, -4],
        [-1, 1, 2, -3]
    ]
)

twistedTransferMatrix = twistedTransferMatrix.reshape(D_MPU * chiMPS, D_MPU * chiMPS)

eig_vals = np.linalg.eigvals(twistedTransferMatrix)
eig_vals = eig_vals / eig_vals[np.argmax(np.abs(eig_vals))]
eig_vals = eig_vals[np.argsort(np.abs(eig_vals))]
print(np.round(eig_vals, decimals=3))