import numpy as np
from .tensor_equation_solver import kron_all ,tensor_equation_solver, PAULI
from vumpsTransferMatrix.vumps import vumpsMPO, normalizedEnvironment
from vumpsTransferMatrix.ncon import ncon

I = np.array([[1, 0],[0, 1]],dtype=complex)
X = np.array([[0, 1],[1, 0]],dtype=complex)
Y = np.array([[0, -1.0j],[1.0j, 0]],dtype=complex)
Z = np.array([[1, 0],[0, -1] ],dtype=complex)

bell, _ = tensor_equation_solver(
[ 
    [X, X],
    [Z, Z]
]
)

print(bell / bell[0,0])