import numpy as np
from ncon import ncon
from vumps import vumpsMPO

I = np.array(
    [
        [1, 0],
        [0, 1]
    ],
    dtype=complex
)

X = np.array(
    [
        [0, 1],
        [1, 0]
    ],
    dtype=complex
)

Z = np.array(
    [
        [1, 0],
        [0, -1]
    ],
    dtype=complex
)

tau = 0.01
g = 0.8

mpoG = np.zeros((2,) * 4, dtype=complex)

mpoG[0, :, :, 0] = np.cosh(tau * g) * I
mpoG[1, :, :, 1] = np.sinh(tau * g) * I
mpoG[0, :, :, 1] = np.sqrt( np.sinh(tau * g ) * np.cosh(tau * g) ) * Z
mpoG[1, :, :, 0] = np.sqrt( np.sinh(tau * g ) * np.cosh(tau * g)) * Z

matrixH = np.cosh(tau / 2) * I + np.sinh(tau / 2) * X

mpoTrotterIsing = ncon(
    [matrixH, mpoG, matrixH],
    [
        [-2, 1],
        [-1, 1, 2, -4],
        [2, -3]
    ]
)

AC, C, AL, AR, leftenv, rightenv, mpo_lam = vumpsMPO(
    mpoTensor=mpoTrotterIsing, 
    chi=50, 
    maxIter=50,
    tol=1e-12,
    info=True
    )