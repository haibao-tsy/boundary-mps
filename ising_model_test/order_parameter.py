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

chi = int(input("enter bond dimension chi: "))
tol = float(input("enter tolerance for convergence: "))
AC, C, AL, AR, leftenv, rightenv, mpo_lam = vumpsMPO(
    mpoTensor=mpoTrotterIsing, 
    chi=chi, 
    maxIter=50,
    tol=tol,
    info=True
)

mz = ncon(
    [AC.conj(), Z, AC],
    [
        [1, 2, 3],
        [2, 4],
        [1, 4, 3],
    ],
) / np.vdot(AC, AC)

mz = np.abs( np.real_if_close(mz) )

if g > 1: 
    mz_theo = np.power(1 - np.power(g, -2), 1 / 8)
if g < 1:
    mz_theo = 0

print(f"order parameter = {mz: .3f}")
print(f"theoretical order patameter = {mz_theo: .3f}")
