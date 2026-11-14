import numpy as np
from tensor_equation_solver import tensor_equation_solver
from vumpsTransferMatrix.vumps import vumpsMPO
from vumpsTransferMatrix.ncon import ncon

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1.0j], [1.0j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

chi = 2
tol = 1e-6
chiMPS = 10
if_perturbation = False

print(f"VUMPS tolerance = {tol}")
print(f"boundary MPS bond dimenstion = {chiMPS}\n")
pepsA, _ = tensor_equation_solver(
    [
        [X, I, I.T, X.T, X],
        [I, Y, Y.T, I.T, X],
        [I, I, X.T, Y.T, Z],
        [Y, X, I.T, I.T, Z],
        [I, X, Z.T, X.T, I],
        # [I, X, X.T, I.T, I],  # e' = e1m2 condensation
        # [X, Y, I.T, I.T, I],  # m' = m1 condensation
    ]
)

pepsB = pepsA.copy()

# pepsB, _ = tensor_equation_solver(
#     [
#         [X, I, I.T, X.T, X],
#         [I, Y, Y.T, I.T, X],
#         [I, I, X.T, Y.T, Z],
#         [Y, X, I.T, I.T, Z],
#         [I, X, Z.T, X.T, I],
#         # [X, Y, I.T, I.T, I],  # e' condensation
#         # [I, X, X.T, I.T, I],  # m' = m1 condensation
#     ]
# )


if if_perturbation:

    perturbationA, _ = tensor_equation_solver(
        [[X, Y, Y.T, X.T, I], [Y, X, X.T, Y.T, I], [Y, I, I.T, Y.T, I]]
    )
    perturbationB, _ = tensor_equation_solver(
        [[X, Y, Y.T, X.T, I], [Y, X, X.T, Y.T, I], [X, Y, I.T, I.T, I]]
    )

    perturbationA = perturbationA / 10.0
    perturbationB = perturbationB / 10.0

    pepsA = pepsA + perturbationA
    pepsB = pepsB + perturbationB

translationMPO = ncon([I, I], [[-1, -2], [-3, -4]])

#   A
#   B
twistedTransferMatrix1 = ncon(
    [pepsA, pepsB, translationMPO, translationMPO, pepsB.conj(), pepsA.conj()],
    [
        [-7, 5, -11, -14, 7],
        [-6, -3, -10, 5, 2],
        [-2, 1, 2, 4],
        [4, 6, 7, -13],
        [-4, -1, -8, 3, 1],
        [-5, 3, -9, -12, 6],
    ],
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
        [4, 6, 7, -13],
        [-4, -1, -8, 3, 1],
        [-5, 3, -9, -12, 6],
    ],
)

twistedTransferMatrix2 = twistedTransferMatrix2.reshape(chi**3, chi**4, chi**4, chi**3)


# the configuration of transfer matrix is :
#                 A B
#                 B A
twistedTransferMatrix = ncon(
    [twistedTransferMatrix1, twistedTransferMatrix2], [[-1, -3, 1, -5], [-2, 1, -4, -6]]
)

twistedTransferMatrix = twistedTransferMatrix.reshape(chi**6, chi**4, chi**4, chi**6)

twistedTransferMatrix = twistedTransferMatrix / np.linalg.norm(twistedTransferMatrix)

print("Calculating translation twisted boundary MPS: \n")
(
    twistedAC_LEFT,
    twistedC_LEFT,
    twistedAL_LEFT,
    twistedAR_LEFT,
    twistedleftenv_LEFT,
    twistedrightenv_LEFT,
    twistedlam,
) = vumpsMPO(mpoTensor=twistedTransferMatrix, chi=chiMPS, maxIter=10000, tol=tol)


# NOTE: untwisted
transferMatrix1 = ncon(
    [pepsA, pepsB, pepsB.conj(), pepsA.conj()],
    [[-6, 3, -10, -12, 4], [-5, -2, -9, 3, 1], [-3, -1, -7, 2, 1], [-4, 2, -8, -11, 4]],
)

transferMatrix1 = transferMatrix1.reshape(chi**2, chi**4, chi**4, chi**2)

transferMatrix2 = ncon(
    [pepsB, pepsA, pepsA.conj(), pepsB.conj()],
    [[-6, 3, -10, -12, 4], [-5, -2, -9, 3, 1], [-3, -1, -7, 2, 1], [-4, 2, -8, -11, 4]],
)

transferMatrix2 = transferMatrix2.reshape(chi**2, chi**4, chi**4, chi**2)
# the configuration of transfer matrix is :
#                 A B
#                 B A
transferMatrixPEPS = ncon(
    [transferMatrix1, transferMatrix2], [[-1, -3, 1, -5], [-2, 1, -4, -6]]
)

transferMatrixPEPS = transferMatrixPEPS.reshape((chi**4,) * 4)

transferMatrixPEPS = transferMatrixPEPS / np.linalg.norm(transferMatrixPEPS)

print("\nCalculating untwisted left boundary MPS: \n")
AC_LEFT, C_LEFT, AL_LEFT, AR_LEFT, leftenv_LEFT, rightenv_LEFT, lam = vumpsMPO(
    mpoTensor=transferMatrixPEPS, chi=chiMPS, maxIter=10000, tol=1e-6
)

print("\nCalculating untwisted right boundary MPS: \n")
AC_RIGHT, C_RIGHT, AL_RIGHT, AR_RIGHT, leftenv_RIGHT, rightenv_RIGHT, _ = vumpsMPO(
    mpoTensor=transferMatrixPEPS.conj().transpose(0, 2, 1, 3),
    chi=chiMPS,
    maxIter=1000,
    tol=1e-6,
)

twistedOverlap = ncon([twistedAL_LEFT, AL_RIGHT.conj()], [[-2, 1, -4], [-1, 1, -3]])

twistedOverlap = twistedOverlap.reshape(chiMPS**2, chiMPS**2)

lam_twisted = np.linalg.eigvals(twistedOverlap)
lam_twisted = lam_twisted[np.argsort(np.abs(lam_twisted))][::-1]
lam_twisted_dom = lam_twisted[
    np.isclose(np.abs(lam_twisted), np.abs(lam_twisted).max(), atol=1e-1, rtol=0)
]

untwistedOverlap = ncon([AL_LEFT, AL_RIGHT.conj()], [[-2, 1, -4], [-1, 1, -3]])

untwistedOverlap = untwistedOverlap.reshape(chiMPS**2, chiMPS**2)

lam_untwisted = np.linalg.eigvals(untwistedOverlap)
lam_untwisted = lam_untwisted[np.argsort(np.abs(lam_untwisted))][::-1]
lam_untwisted_dom = lam_untwisted[
    np.isclose(np.abs(lam_untwisted), np.abs(lam_untwisted).max(), atol=1e-5)
]

MPDO_LEFT = AL_LEFT.reshape(chiMPS, chi, chi, chi, chi, chiMPS)
MPDO_LEFT = MPDO_LEFT.transpose(0, 3, 4, 1, 2, 5)
MPDO_RIGHT = AL_RIGHT.reshape(chiMPS, chi, chi, chi, chi, chiMPS)
MPDO_RIGHT = MPDO_RIGHT.conj()

# mpu = np.array([[I, X], [1j * Y, Z]]).transpose(0, 2, 3, 1)
# mpu = mpu / np.sqrt(2)
mpu = translationMPO.copy()
D_MPU = mpu.shape[0]

transferMatrixMPS = ncon(
    [MPDO_LEFT, MPDO_RIGHT], [[-1, 1, 3, 2, 4, -3], [-2, 2, 4, 1, 3, -4]]
)

transferMatrixMPS = transferMatrixMPS.reshape(chiMPS * chiMPS, chiMPS * chiMPS)

twistedTransferMatrixMPS = ncon(
    [MPDO_RIGHT, mpu, mpu, MPDO_LEFT],
    [[-1, 1, 5, 2, 6, -4], [-2, 2, 3, 4], [4, 6, 7, -5], [-3, 3, 7, 1, 5, -6]],
)

twistedTransferMatrixMPS = twistedTransferMatrixMPS.reshape(
    D_MPU * chiMPS * chiMPS, D_MPU * chiMPS * chiMPS
)

mpu_lam_twisted = np.linalg.eigvals(twistedTransferMatrixMPS)
mpu_lam_twisted = mpu_lam_twisted[np.argsort(np.abs(mpu_lam_twisted))][::-1]
# difference larger than 0.1 are not consider close
mpu_lam_twisted_dom = mpu_lam_twisted[
    np.isclose(
        np.abs(mpu_lam_twisted), np.abs(mpu_lam_twisted).max(), atol=1e-1, rtol=0
    )
]

mpu_lam_untwisted = np.linalg.eigvals(transferMatrixMPS)
mpu_lam_untwisted_dom = mpu_lam_untwisted[
    np.isclose(np.abs(mpu_lam_untwisted), np.abs(mpu_lam_untwisted).max(), atol=1e-2)
]
