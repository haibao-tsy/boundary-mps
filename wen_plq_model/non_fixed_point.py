import numpy as np
from tensor_equation_solver import tensor_equation_solver, kron_all
from vumpsTransferMatrix.vumps import vumpsMPO, normalizedEnvironment
from vumpsTransferMatrix.ncon import ncon

I = np.array([[1, 0],[0, 1]],dtype=complex)
X = np.array([[0, 1],[1, 0]],dtype=complex)
Y = np.array([[0, -1.0j],[1.0j, 0]],dtype=complex)
Z = np.array([[1, 0],[0, -1] ],dtype=complex)

chi = 2
tol=1e-5
chiMPS = 10

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


perturbationA, _ = tensor_equation_solver(
    [
        [X, Y, Y.T, X.T, I],
        [Y, X, X.T, Y.T, I],
        [Y, I, I.T, Y.T, I]
    ]
)
perturbationB, _ = tensor_equation_solver(
    [
        [X, Y, Y.T, X.T, I],
        [Y, X, X.T, Y.T, I],
        [X, Y, I.T, I.T, I]
    ]
)

perturbationA = perturbationA / 3.0
perturbationB = perturbationB / 3.0

pepsA = pepsA + perturbationA
pepsB = pepsB + perturbationB

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


AC_LEFT, C_LEFT, AL_LEFT, AR_LEFT, leftenv_LEFT, rightenv_LEFT, lam = vumpsMPO(
    mpoTensor=transferMatrixPEPS,
    chi = chiMPS,
    maxIter = 10000,
    tol=tol
)

MPDO_LEFT = AL_LEFT.reshape(chiMPS, chi, chi, chi, chi, chiMPS)
MPDO_LEFT = MPDO_LEFT.transpose(0, 3, 4, 1, 2, 5)

AC_RIGHT, C_RIGHT, AL_RIGHT, AR_RIGHT, leftenv_RIGHT, rightenv_RIGHT, _ = vumpsMPO(
    mpoTensor=transferMatrixPEPS.conj().transpose(0, 2, 1, 3),
    chi = chiMPS,
    maxIter = 1000,
    tol=tol
)

MPDO_RIGHT = AL_RIGHT.reshape(chiMPS, chi, chi, chi, chi, chiMPS)
MPDO_RIGHT = MPDO_RIGHT.conj()

mpu = np.array([[I, X], [1j * Y, Z]]).transpose(0, 2, 3, 1)
mpu = mpu / np.sqrt(2)
D_MPU = mpu.shape[0]

transferMatrixMPS = ncon(
    [MPDO_LEFT, MPDO_RIGHT],
    [
        [-1, 1, 3, 2, 4, -3],
        [-2, 2, 4, 1, 3, -4]
    ]
)

transferMatrixMPS = transferMatrixMPS.reshape(chiMPS * chiMPS, chiMPS * chiMPS)

twistedTransferMatrixMPS = ncon(
    [MPDO_RIGHT, mpu, mpu, MPDO_LEFT],
    [
        [-1, 1, 5, 2, 6, -4],
        [-2, 2, 3, 4],
        [4, 6, 7 ,-5],
        [-3, 3, 7, 1, 5, -6]
    ]
)

twistedTransferMatrixMPS = twistedTransferMatrixMPS.reshape(D_MPU * chiMPS * chiMPS, D_MPU * chiMPS * chiMPS)


lam_twisted = np.linalg.eigvals(twistedTransferMatrixMPS)
lam_twisted_dom = lam_twisted[np.isclose(np.abs(lam_twisted), np.abs(lam_twisted).max())]
print("twisted dominant eigenvalue = ", np.round( lam_twisted, decimals=3 )[0:3])
print("twisted dominant eigenvalue norm = ", np.round( np.abs( lam_twisted ), decimals=3 )[0:3])

lam_untwisted = np.linalg.eigvals(transferMatrixMPS)
lam_untwisted_dom = lam_untwisted[np.isclose(np.abs(lam_untwisted), np.abs(lam_untwisted).max())]
print("untwisted dominant eigenvalue = ", np.round( lam_untwisted, decimals=3 )[0:3])
print("untwisted dominant eigenvalue norm = ", np.round( np.abs( lam_untwisted ), decimals=3 )[0:3])

print("ratio = ", np.round( lam_twisted_dom[0] / lam_untwisted_dom[0], decimals=3 ))
