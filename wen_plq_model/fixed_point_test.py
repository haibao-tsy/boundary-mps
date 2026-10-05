import numpy as np
from tensor_equation_solver import tensor_equation_solver
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

transferMatrix = ncon(
    [transferMatrix1, transferMatrix2],
    [
        [-1, -3, 1, -5],
        [-2, 1, -4, -6]
    ]
)

transferMatrix = transferMatrix.reshape((chi**4, ) * 4)
