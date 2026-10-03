"""Estimate the spontaneous magnetization of the transverse-field Ising chain.

Use the Hamiltonian H = -sum_i (g Z_i Z_{i+1} + X_i), with transverse
field strength fixed to one. A symmetric second-order Trotter step for
exp(-tau H) is represented as a uniform matrix product operator (MPO).
VUMPS approximates its dominant eigenvector as a uniform matrix product
state (MPS), from which the magnitude of the Z magnetization is measured.

The result is compared with the infinite-chain, zero-temperature formula
(1 - g**(-2))**(1/8), valid in the ordered phase g > 1. Finite tau, finite
MPS bond dimension, and incomplete convergence can affect the comparison.
Running or importing this script performs the calculation and prints it.
"""
import numpy as np
from vumpsTransferMatrix.ncon import ncon
from vumpsTransferMatrix.vumps import vumpsMPO

# Single-spin identity and Pauli operators in the Z basis.
I = np.array([[1, 0],[0, 1]],dtype=complex)

X = np.array([[0, 1],[1, 0]],dtype=complex)

Z = np.array([[1, 0],[0, -1] ],dtype=complex)

g = float(input("enter the ZZ coupling g: "))  # Ferromagnetic coupling; the quantum critical point is g = 1.
tau = float(input("enter the imaginary time tau: "))  # Imaginary-time step used to construct the transfer MPO.

# MPO axes: (left virtual bond, bra spin, ket spin, right virtual bond).
# Its virtual bond dimension is two, matching the I and Z bond channels.
mpoG = np.zeros((2,) * 4, dtype=complex)

# Factor exp(tau*g*Z_i*Z_{i+1}) = cosh(tau*g) I_i I_{i+1}
# + sinh(tau*g) Z_i Z_{i+1}. Each site combines the factors from its
# adjacent bonds; mixed channels carry the geometric mean of the weights.
mpoG[0, :, :, 0] = np.cosh(tau * g) * I
mpoG[1, :, :, 1] = np.sinh(tau * g) * I
mpoG[0, :, :, 1] = np.sqrt( np.sinh(tau * g ) * np.cosh(tau * g) ) * Z
mpoG[1, :, :, 0] = np.sqrt( np.sinh(tau * g ) * np.cosh(tau * g)) * Z

# Since X**2 = I, this is the on-site half step exp((tau/2) X).
matrixH = np.cosh(tau / 2) * I + np.sinh(tau / 2) * X

# Sandwich the interaction MPO between transverse-field half steps.
# Positive ncon labels are contracted; -1 through -4 set the output axes
# to (left virtual bond, bra spin, ket spin, right virtual bond).
mpoTrotterIsing = ncon(
    [matrixH, mpoG, matrixH],
    [
        [-2, 1],
        [-1, 1, 2, -4],
        [2, -3]
    ]
)

# Find the dominant transfer-MPO eigenstate with MPS bond dimension chi.
# AC is the center-site tensor, C the center matrix, and AL/AR the
# left/right canonical tensors. The remaining outputs are the transfer
# environments and their dominant eigenvalue.
AC, C, AL, AR, leftenv, rightenv, mpo_lam = vumpsMPO(
    mpoTensor=mpoTrotterIsing, 
    chi=200, 
    maxIter=50,
    tol=1e-8,
    info=True
)

# Contract <AC|Z|AC> over both MPS bonds and the physical spin index.
# In mixed canonical form, dividing by <AC|AC> normalizes this expectation.
mz = ncon(
    [AC.conj(), Z, AC],
    [
        [1, 2, 3],
        [2, 4],
        [1, 4, 3],
    ],
) / np.vdot(AC, AC)

# Remove negligible imaginary roundoff and compare magnitudes: either
# symmetry-broken orientation can have positive or negative magnetization.
mz = np.abs( np.real_if_close(mz) )

# Exact spontaneous magnetization for this coupling convention, at g > 1.
if g > 1: 
    mz_theo = np.power(1 - np.power(g, -2), 1 / 8)
if g < 1:
    mz_theo = 0

print(f"mz = {mz: .3f}")
print(f"mz_theoretical = {mz_theo: .3f}")
# Relative discrepancy uses the numerical magnetization as the denominator.
print(f"error = {np.abs( ( mz - mz_theo ) / mz ): .3e}")
