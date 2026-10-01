"""VUMPS fixed-point iteration for a uniform transfer matrix represented as an MPO.

Tensor axes throughout this module are:

* MPS tensors: (left bond, physical index, right bond), shape (chi, d, chi).
* MPO tensors: (left bond, bra physical index, ket physical index, right
  bond), shape (D, d, d, D).
* Environments: (bra MPS bond, MPO bond, ket MPS bond), shape (chi, D, chi).
* Center matrices: (left bond, right bond), shape (chi, chi).

The effective operators target eigenvalues of largest magnitude. The
iteration assumes an injective MPS and well-defined dominant environment
fixed points. All error measures use absolute Frobenius norms.
"""

import numpy as np
from scipy.linalg import polar
from .ncon import ncon
from scipy.sparse.linalg import LinearOperator, eigs
from .canonicalForm import mixedCanonicalQR

# chi is the dimension of the MPS internal leg
# d is the dimension of the MPS physical leg
# D is the dimension of the MPO internal leg

def leftEnvironment(mpoTensor, mpsLeft, tol=1e-14):
    """Find the dominant environment of the left-to-right MPO channel.

    Parameters
    ----------
    mpoTensor : ndarray, shape (D, d, d, D)
        Transfer MPO tensor with axes (left, bra physical, ket physical,
        right).
    mpsLeft : ndarray, shape (chi, d, chi)
        Left-canonical MPS tensor, satisfying
        sum_s mpsLeft[:, s, :].conj().T @ mpsLeft[:, s, :] = I.
    tol : float, optional
        Relative tolerance passed to the channel eigensolver.

    Returns
    -------
    lE : ndarray, shape (chi, D, chi)
        Dominant channel eigenvector with axes (bra bond, MPO bond, ket
        bond), unit Frobenius norm, and arbitrary complex phase.
    mpo_lam : complex
        Corresponding channel eigenvalue, selected by largest magnitude.

    Notes
    -----
    The current sparse eigensolver requires chi * D * chi > 2.
    """
    chi = mpsLeft.shape[0]
    D = mpoTensor.shape[0]

    def matvec(v):
        v = v.reshape(chi, D, chi)

        result = ncon(
            [mpsLeft.conj(), mpoTensor, mpsLeft, v],
            [
                [1, 4, -1],
                [2, 4, 5, -2],
                [3, 5, -3],
                [1, 2, 3]
            ]
        )

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chi * D * chi, chi * D * chi),
        matvec=matvec,
        dtype=np.complex128,
    )

    eigenvalues, eigenvector = eigs(transferMatrix,k=1, which="LM", tol=tol)

    lE = eigenvector[:, 0].reshape(chi, D, chi)
    mpo_lam = eigenvalues[0]

    return lE, mpo_lam

def rightEnvironment(mpoTensor, mpsRight, tol=1e-14):
    """Find the dominant environment of the right-to-left MPO channel.

    Parameters
    ----------
    mpoTensor : ndarray, shape (D, d, d, D)
        Transfer MPO tensor with axes (left, bra physical, ket physical,
        right).
    mpsRight : ndarray, shape (chi, d, chi)
        Right-canonical MPS tensor, satisfying
        sum_s mpsRight[:, s, :] @ mpsRight[:, s, :].conj().T = I.
    tol : float, optional
        Relative tolerance passed to the channel eigensolver.

    Returns
    -------
    rE : ndarray, shape (chi, D, chi)
        Dominant channel eigenvector with axes (bra bond, MPO bond, ket
        bond), unit Frobenius norm, and arbitrary complex phase.
    mpo_lam : complex
        Corresponding channel eigenvalue, selected by largest magnitude.

    Notes
    -----
    The current sparse eigensolver requires chi * D * chi > 2.
    """
    chi = mpsRight.shape[0]
    D = mpoTensor.shape[0]

    def matvec(v):
        v = v.reshape(chi, D, chi)

        result = ncon(
            [mpsRight.conj(), mpoTensor, mpsRight, v],
            [
                [-1, 4, 1],
                [-2, 4, 5, 2],
                [-3, 5, 3],
                [1, 2, 3]
            ]
        )

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chi * D * chi, chi * D * chi),
        matvec=matvec,
        dtype=np.complex128,
    )

    eigenvalues, eigenvector = eigs(transferMatrix,k=1, which="LM", tol=tol)

    rE = eigenvector[:, 0].reshape(chi, D, chi)
    mpo_lam = eigenvalues[0]
    return rE, mpo_lam

def normalizedEnvironment(mpoTensor: np.array,
                          mpsLeft: np.array,
                          mpsRight: np.array,
                          C: np.array,
                          tol=1e-14):
    """Compute environments and normalize their contraction through C.

    Parameters
    ----------
    mpoTensor : ndarray, shape (D, d, d, D)
        Transfer MPO tensor in the module's axis convention.
    mpsLeft, mpsRight : ndarray, shape (chi, d, chi)
        Left- and right-canonical tensors of the current MPS.
    C : ndarray, shape (chi, chi)
        Current center matrix connecting the two canonical gauges.
    tol : float, optional
        Relative tolerance for both environment eigensolvers.

    Returns
    -------
    leftenv, rightenv : ndarray, shape (chi, D, chi)
        Environments with axes (bra bond, MPO bond, ket bond). Only the
        left environment is rescaled; their joint contraction satisfies
        sum_abcdm leftenv[a, m, c] * conj(C[a, b]) * C[c, d]
        * rightenv[b, m, d] = 1.
    mpo_lam : complex
        Dominant left-channel eigenvalue. For consistent mixed-canonical
        tensors, the right-channel eigenvalue agrees with it.
    """

    leftenv, mpo_lam = leftEnvironment(mpoTensor=mpoTensor, mpsLeft=mpsLeft, tol=tol)
    rightenv, _ = rightEnvironment(mpoTensor=mpoTensor, mpsRight=mpsRight, tol=tol)

    overlap = ncon(
        [leftenv, C.conj(), C, rightenv],
        [
            [1, 2, 3],
            [1, 4],
            [3, 5],
            [4, 2, 5]
        ]
    )

    leftenv /= overlap

    return leftenv, rightenv, mpo_lam

def convergenceResidual(mpoTensor, AC, C, AL,
                        leftenv, rightenv, mpo_lam):
    """Measure the left-gauge projected fixed-point residual.

    Parameters
    ----------
    mpoTensor : ndarray, shape (D, d, d, D)
        Transfer MPO tensor in the module's axis convention.
    AC : ndarray, shape (chi, d, chi)
        Current center tensor.
    C : ndarray, shape (chi, chi)
        Current center matrix.
    AL : ndarray, shape (chi, d, chi)
        Current left-canonical tensor.
    leftenv, rightenv : ndarray, shape (chi, D, chi)
        Environments computed for the current tensors and normalized by
        normalizedEnvironment.
    mpo_lam : complex
        Nonzero channel eigenvalue used to scale the center-tensor action.

    Returns
    -------
    epsilon : float
        Frobenius norm of B - AL @ K, where B = H_AC(AC) includes division
        by mpo_lam and K = H_C(C). The product acts on the MPS right bond.

    Notes
    -----
    B and K are operator applications to the current tensors; no new
    eigenvectors are computed. The projected-residual interpretation
    assumes consistent mixed-canonical tensors and converged environments.
    Check canonicalError separately. A small residual measures stationarity
    within the chosen bond dimension, not the full MPO eigenstate error.
    """

    B = ncon(
        [mpoTensor, leftenv, rightenv, AC],
        [
            [4, -2, 2, 5],
            [-1, 4, 1],
            [-3, 5, 3],
            [1, 2, 3],
        ],
    ) / mpo_lam

    K = ncon(
        [leftenv, rightenv, C],
        [
            [-1, 3, 1],
            [-2, 3, 2],
            [1, 2],
        ],
    )

    ALK = ncon(
        [AL, K],
        [[-1, -2, 1], [1, -3]],
    )

    return np.linalg.norm(B - ALK)

def canonicalError(
        AC: np.ndarray,
        C: np.ndarray,
        AL: np.ndarray,
        AR: np.ndarray):
    """Measure the mismatch between the two center factorizations.

    Parameters
    ----------
    AC : ndarray, shape (chi, d, chi)
        Center tensor to compare with AL @ C and C @ AR.
    C : ndarray, shape (chi, chi)
        Center matrix; it need not be diagonal or real.
    AL, AR : ndarray, shape (chi, d, chi)
        Left- and right-canonical MPS tensors.

    Returns
    -------
    error : float
        max(norm(AC - AL @ C), norm(AC - C @ AR)), with Frobenius norms
        over all tensor entries. Products contract the adjacent bond axes.

    Notes
    -----
    This is an absolute consistency error. It does not independently test
    the isometry of AL or AR, or the normalization of AC and C.
    """

    error1 = np.linalg.norm(
        AC - ncon([AL, C],[[-1, -2, 1], [1, -3]])
    )
    
    error2 = np.linalg.norm(
        AC - ncon([C, AR],[[-1, 1], [1, -2, -3]])
    )

    return max(error1, error2)


def updateAC(mpoTensor, leftenv, rightenv, mpo_lam, tol=1e-14):
    """Solve the effective center-tensor eigenproblem.

    Parameters
    ----------
    mpoTensor : ndarray, shape (D, d, d, D)
        Transfer MPO tensor in the module's axis convention.
    leftenv, rightenv : ndarray, shape (chi, D, chi)
        Current normalized environments with axes (bra bond, MPO bond,
        ket bond).
    mpo_lam : complex
        Nonzero channel eigenvalue. The effective operator is divided by
        this value before solving the eigenproblem.
    tol : float, optional
        Relative tolerance passed to the effective eigensolver.

    Returns
    -------
    AC_new : ndarray, shape (chi, d, chi)
        Eigenvector of the scaled effective operator with largest-magnitude
        eigenvalue, normalized to unit Frobenius norm.
    eigenvalue : complex
        Eigenvalue of that scaled effective operator, not the channel
        eigenvalue mpo_lam.

    Notes
    -----
    AC_new has an arbitrary complex phase independent of the phase chosen
    by updateC. Use recoverCanonical to obtain compatible isometries.
    The current sparse eigensolver requires chi * d * chi > 2.
    """
    chi = leftenv.shape[0]
    d = mpoTensor.shape[1]
    def matvec(v):
        v = v.reshape(chi, d, chi)
        result = ncon(
            [mpoTensor, leftenv, rightenv, v],
            [
                [4, -2, 2, 5],  
                [-1, 4, 1],     
                [-3, 5, 3],     
                [1, 2, 3]      
            ]
        )
        result /= mpo_lam
        return result.ravel()

    HAC = LinearOperator(
        shape=(chi * d * chi, chi * d * chi),
        matvec=matvec,
        dtype=np.complex128
    )

    eigenvalues, eigenvector = eigs(
        A=HAC,
        k=1,
        which="LM",
        tol=tol
    )

    return eigenvector[:, 0].reshape(chi, d, chi), eigenvalues[0]
    
def updateC(leftenv, rightenv, tol=1e-14):
    """Solve the effective center-matrix eigenproblem.

    Parameters
    ----------
    leftenv, rightenv : ndarray, shape (chi, D, chi)
        Current normalized environments with axes (bra bond, MPO bond,
        ket bond).
    tol : float, optional
        Relative tolerance passed to the effective eigensolver.

    Returns
    -------
    C_new : ndarray, shape (chi, chi)
        Eigenvector of H_C with largest-magnitude eigenvalue and unit
        Frobenius norm. Its complex phase is arbitrary; it is generally
        neither diagonal nor Hermitian.
    eigenvalue : complex
        Corresponding effective bond eigenvalue.

    Notes
    -----
    The current sparse eigensolver requires chi * chi > 2, so chi = 1
    is unsupported by this function.
    """
    chi = leftenv.shape[0]
    def matvec(v):
        v = v.reshape(chi, chi)
        result = ncon(
            [leftenv, rightenv, v],
            [
                [-1, 3, 1],
                [-2, 3, 2],
                [1, 2]
            ]
        )
        return result.ravel()

    HC = LinearOperator(
        shape=(chi * chi, chi * chi),
        matvec=matvec,
        dtype=np.complex128
    )

    eigenvalues, eigenvector = eigs(
        A=HC,
        k=1,
        which="LM",
        tol=tol
    )

    return eigenvector[:, 0].reshape(chi, chi), eigenvalues[0]

def recoverCanonical(AC, C):
    """Recover left and right isometries using polar decompositions.

    Parameters
    ----------
    AC : ndarray, shape (chi, d, chi)
        Center tensor, typically returned by updateAC.
    C : ndarray, shape (chi, chi)
        Center matrix, typically returned by updateC.

    Returns
    -------
    AL : ndarray, shape (chi, d, chi)
        Left-canonical tensor satisfying sum_s AL_s.conj().T @ AL_s = I.
    AR : ndarray, shape (chi, d, chi)
        Right-canonical tensor satisfying sum_s AR_s @ AR_s.conj().T = I.

    Notes
    -----
    Here AL_s = AL[:, s, :] and AR_s = AR[:, s, :]. Isometry holds up to
    numerical precision, but AC = AL @ C = C @ AR need not hold for
    arbitrary inputs. Use canonicalError to measure that mismatch.
    AC and C are neither modified nor brought to a diagonal Schmidt gauge.
    """
    chi, d = AC.shape[: 2]
    U_AC_left, _ = polar(
        AC.reshape(chi * d, chi), side="right"
    )
    U_AC_right, _ = polar(
        AC.reshape(chi, d * chi), side="left"
    )
    U_C, _ = polar(C)

    AL = (U_AC_left @ U_C.conj().T).reshape(chi, d, chi)
    AR = (U_C.conj().T @ U_AC_right).reshape(chi, d, chi)

    return AL, AR

def vumpsMPO(mpoTensor: np.ndarray, 
            chi: int,
            A0: np.ndarray=None, 
            tol: float=1e-14,
            eigs_tol: float=1e-14,
            maxIter: int=1e5,
            info: bool=True):
    """Seek a dominant uniform-MPS fixed point of a transfer MPO.

    Parameters
    ----------
    mpoTensor : ndarray, shape (D, d, d, D)
        Uniform transfer MPO tensor with axes (left bond, bra physical
        index, ket physical index, right bond).
    chi : int
        MPS bond dimension for random initialization; must be at least 2
    A0 : ndarray, shape (chi, d, chi), optional
        Initial injective MPS tensor. If omitted, use uniform random real
    tol : float, optional
        Tolerance for convergence
    maxIter : int, optional
        Maximum number of outer iterations (default 100000).

    Returns
    -------
    AC : ndarray, shape (chi, d, chi)
        Final center tensor with unit Frobenius norm.
    C : ndarray, shape (chi, chi)
        Final center matrix with unit Frobenius norm. It is generally
        complex and non-diagonal after updates.
    AL, AR : ndarray, shape (chi, d, chi)
        Final left- and right-canonical tensors. On convergence,
        AC = AL @ C = C @ AR holds within tol in Frobenius norm.
    leftenv, rightenv : ndarray, shape (chi, D, chi)
        Normalized environments with axes (bra bond, MPO bond, ket bond).
        They correspond to the returned MPS when convergence is detected.
    mpo_lam : complex
        Dominant left-channel eigenvalue from the final environment
        calculation, representing the transfer eigenvalue per site at
        the MPS fixed point.
    """
    d = mpoTensor.shape[1]
    if A0 is None: 
        A0 = np.random.rand(chi, d, chi)

    AC, C, AL, AR = mixedCanonicalQR(A=A0, tol=tol)

    iter_count = 0
    while iter_count < maxIter:
        iter_count += 1

        leftenv, rightenv, mpo_lam = normalizedEnvironment(
                                    mpoTensor=mpoTensor,
                                    mpsLeft=AL,
                                    mpsRight=AR,
                                    C=C,
                                    tol=tol)

        epsilon = convergenceResidual(
            mpoTensor=mpoTensor,
            AC=AC,
            C=C,
            AL=AL,
            leftenv=leftenv,
            rightenv=rightenv,
            mpo_lam=mpo_lam)

        error = canonicalError(AC=AC, C=C, AL=AL, AR=AR)

        if epsilon < tol and error < tol: 
            break

        if iter_count == maxIter: 
            raise RuntimeError(f"Failed to converge below tolerance of {tol} after {iter_count} iterations")

        AC_new, _= updateAC(
                mpoTensor=mpoTensor, 
                leftenv=leftenv, 
                rightenv=rightenv, 
                mpo_lam=mpo_lam,
                tol=eigs_tol
                )

        C_new, _ = updateC(
                leftenv=leftenv, 
                rightenv=rightenv, 
                tol=eigs_tol)

        AL_new, AR_new = recoverCanonical(AC=AC_new, C=C_new)
        AL = AL_new
        AR = AR_new
        AC = AC_new
        C = C_new

        if info:
            print(
                f"{iter_count}: ", 
                f"eigenvalue: {mpo_lam:.3f}",
                f"residual: {epsilon:.3e}",
                f"canonical error{error:.3e}")


    return AC, C, AL, AR, leftenv, rightenv, mpo_lam
