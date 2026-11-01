import numpy as np
from tensor_equation_solver import tensor_equation_solver
from vumpsTransferMatrix.vumps import vumpsMPO
from vumpsTransferMatrix.ncon import ncon

I = np.array([[1, 0],[0, 1]],dtype=complex)
X = np.array([[0, 1],[1, 0]],dtype=complex)
Y = np.array([[0, -1.0j],[1.0j, 0]],dtype=complex)
Z = np.array([[1, 0],[0, -1] ],dtype=complex)

chi = 2
tol=1e-6
chiMPS = 10
if_perturbation = False

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

if if_perturbation:

    perturbationA, _ = tensor_equation_solver(
        [
            [X, Y, Y.T, X.T, I],
            [Y, X, X.T, Y.T, I],
            # [Y, I, I.T, Y.T, I]
        ]
    )
    perturbationB, _ = tensor_equation_solver(
        [
            [X, Y, Y.T, X.T, I],
            [Y, X, X.T, Y.T, I],
            # [X, Y, I.T, I.T, I]
        ]
    )

    perturbationA = perturbationA / 8.0
    perturbationB = perturbationB / 8.0

    pepsA = pepsA + perturbationA
    pepsB = pepsB + perturbationB

translationMPO = np.einsum('ij, kl->ijkl', I, I)

#   A
#   B
twistedTransferMatrix1 = ncon(
    [pepsA, pepsB, translationMPO, translationMPO, pepsB.conj(), pepsA.conj()],
    [
        [-7, 5, -11, -14, 7],
        [-6, -3, -10, 5, 2],
        [-2, 1, 2, 4],
        [4, 6, 7 , -13],
        [-4, -1, -8, 3, 1],
        [-5, 3, -9, -12, 6]
    ]
)

twistedTransferMatrix1 = twistedTransferMatrix1.reshape(chi**3, chi**4, chi**4, chi**3)

#   B
#   A
twistedTransferMatrix2 = ncon(
    [pepsB, pepsA, translationMPO, translationMPO, pepsA.conj(), pepsB.conj()],
    [
        [-7, 5, -11, -14, 7],
        [-6, -3, -10, 5, 2],
        [-2, 1, 2, 4],
        [4, 6, 7 , -13],
        [-4, -1, -8, 3, 1],
        [-5, 3, -9, -12, 6]
    ]
)

twistedTransferMatrix2 = twistedTransferMatrix2.reshape(chi**3, chi**4, chi**4, chi**3)


# the configuration of transfer matrix is :
#                 A B
#                 B A
twistedTransferMatrix = ncon(
    [twistedTransferMatrix1, twistedTransferMatrix2],
    [
        [-1, -3, 1, -5],
        [-2, 1, -4, -6]
    ]
)

twistedTransferMatrix = twistedTransferMatrix.reshape(chi**6, chi**4, chi**4, chi**6)

twistedTransferMatrix = twistedTransferMatrix / np.linalg.norm(twistedTransferMatrix)


AC_LEFT, C_LEFT, AL_LEFT, AR_LEFT, leftenv_LEFT, rightenv_LEFT, lam = vumpsMPO(
    mpoTensor=twistedTransferMatrix,
    chi = chiMPS,
    maxIter = 10000,
    tol=tol
)

MPDO_LEFT = AL_LEFT.reshape(chiMPS, chi, chi, chi, chi, chiMPS)
MPDO_LEFT = MPDO_LEFT.transpose(0, 3, 4, 1, 2, 5)
