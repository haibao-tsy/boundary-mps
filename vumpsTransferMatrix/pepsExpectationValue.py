"""Estimate long-string observables from uniform PEPS boundary MPS tensors."""

import numpy as np
from .ncon import ncon
from scipy.sparse.linalg import eigs, LinearOperator
import warnings


def pepsExpectationValue(
    mpsLEFT: np.ndarray,
    mpsRIGHT: np.ndarray,
    leftEnd: np.ndarray,
    stringBulk: np.ndarray,
    rightEnd: np.ndarray,
    tol: int = 1e-14,
    string_length: int = 1e3,
    return_untwisted_lam: bool = False,
):
    """Estimate a normalized long-string matrix element between uniform MPS.

    Insert ``leftEnd[0] stringBulk[1] ... stringBulk[n] rightEnd[n+1]``,
    where ``n = string_length``, between the bra MPS ``mpsRIGHT`` and ket
    MPS ``mpsLEFT``, and normalize by their overlap. The infinite exterior
    and the bulk string are represented by dominant transfer eigenmodes.
    Identity endpoints give a plain string observable; identity bulk gives
    a long-distance two-point correlation.

    Parameters
    ----------
    mpsLEFT : ndarray, shape (chi, d, chi)
        Uniform ket MPS tensor with axes (left bond, physical, right bond).
    mpsRIGHT : ndarray, shape (chi, d, chi)
        Uniform MPS tensor for the bra, conjugated internally. Both MPS
        tensors must have the same bond and physical dimensions; their
        names distinguish the boundary states, not canonical gauges.
    leftEnd, stringBulk, rightEnd : ndarray, shape (d, d)
        Single-site operators with axes (bra physical, ket physical).
    tol : float, optional
        Relative tolerance passed to each sparse eigensolve (default 1e-14).
    string_length : int, optional
        Number of bulk sites between the endpoints (default 1000).

    Returns
    -------
    exp_val : complex
        Approximate normalized matrix element, including the bulk factor
        ``(lambda_twisted / lambda_untwisted)**string_length``.

    Notes
    -----
    Assumes isolated dominant eigenmodes for both transfer matrices, a
    nonzero untwisted eigenvalue, and nonzero left/right environment
    overlaps. Subleading bulk modes are omitted, so short strings are
    generally not evaluated exactly. The MPS need not be pre-normalized.

    Left environments are eigenvectors of the transfer matrix transpose,
    not its conjugate transpose. Their overlaps with right environments
    therefore use bilinear contractions without complex conjugation.
    """
    chiMPS = mpsLEFT.shape[0]

    # Environments have axes (bra bond, ket bond); flatten only for eigs.
    def matvec(v):
        """Apply the twisted transfer matrix to an environment on the right."""
        v = v.reshape(chiMPS, chiMPS)

        # Positive ncon labels are contracted; -1 and -2 order the open bonds.
        result = ncon(
            [mpsRIGHT.conj(), stringBulk, mpsLEFT, v],
            [[-1, 1, 3], [1, 2], [-2, 2, 4], [3, 4]],
        )

        return result.ravel()

    # Use a matrix-free action instead of storing chiMPS**4 matrix entries.
    transferMatrix = LinearOperator(
        shape=(chiMPS * chiMPS, chiMPS * chiMPS), matvec=matvec, dtype=np.complex128
    )

    # The largest-magnitude eigenmode determines long-string propagation.
    eigenvalues, eigenvectors = eigs(A=transferMatrix, k=1, which="LM", tol=tol)
    twisted_lam = eigenvalues[0]
    twisted_right_env = eigenvectors[:, 0].reshape(chiMPS, chiMPS)

    # Swap the open and contracted bond pairs to apply the transpose.
    def matvec(v):
        """Apply the twisted transpose to an environment on the left."""
        v = v.reshape(chiMPS, chiMPS)

        result = ncon(
            [mpsRIGHT.conj(), stringBulk, mpsLEFT, v],
            [[3, 1, -1], [1, 2], [4, 2, -2], [3, 4]],
        )

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chiMPS * chiMPS, chiMPS * chiMPS), matvec=matvec, dtype=np.complex128
    )

    _, eigenvectors = eigs(A=transferMatrix, k=1, which="LM", tol=tol)
    twisted_left_env = eigenvectors[:, 0].reshape(chiMPS, chiMPS)

    # Omitting stringBulk gives the untwisted channel for the MPS overlap.
    def matvec(v):
        """Apply the untwisted transfer matrix to an environment on the right."""
        v = v.reshape(chiMPS, chiMPS)

        result = ncon([mpsRIGHT.conj(), mpsLEFT, v], [[-1, 1, 2], [-2, 1, 3], [2, 3]])

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chiMPS * chiMPS, chiMPS * chiMPS), matvec=matvec, dtype=np.complex128
    )

    eigenvalues, eigenvectors = eigs(A=transferMatrix, k=1, which="LM", tol=tol)
    untwisted_lam = eigenvalues[0]
    untwisted_right_env = eigenvectors[:, 0].reshape(chiMPS, chiMPS)

    def matvec(v):
        """Apply the untwisted transpose to an environment on the left."""
        v = v.reshape(chiMPS, chiMPS)

        result = ncon([mpsRIGHT.conj(), mpsLEFT, v], [[2, 1, -1], [3, 1, -2], [2, 3]])

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chiMPS * chiMPS, chiMPS * chiMPS), matvec=matvec, dtype=np.complex128
    )

    _, eigenvectors = eigs(A=transferMatrix, k=1, which="LM", tol=tol)
    untwisted_left_env = eigenvectors[:, 0].reshape(chiMPS, chiMPS)

    # Divide out the overlap growth per site before raising to string_length.
    twisted_lam /= untwisted_lam
    if np.abs(twisted_lam) > 1.0 + 1e-10:
        warnings.warn(
            f"Twisted bulk transfer matrix eigenvalue {np.abs(twisted_lam):.3f} > 1!",
            RuntimeWarning,
            stacklevel=2,
        )
    # Join the untwisted exterior to the twisted bulk at the left endpoint.
    exp_val = ncon(
        [untwisted_left_env, mpsRIGHT.conj(), leftEnd, mpsLEFT, twisted_right_env],
        [[1, 2], [1, 3, 5], [3, 4], [2, 4, 6], [5, 6]],
    )

    # Join the twisted bulk back to the untwisted exterior at the right end.
    exp_val *= ncon(
        [twisted_left_env, mpsRIGHT.conj(), rightEnd, mpsLEFT, untwisted_right_env],
        [[1, 2], [1, 3, 5], [3, 4], [2, 4, 6], [5, 6]],
    )

    # Cancel arbitrary eigenvector scales with bilinear environment overlaps.
    normalization = ncon([untwisted_left_env, untwisted_right_env], [[1, 2], [1, 2]])

    twisted_normalization = np.sum(twisted_left_env * twisted_right_env)

    # The two endpoint sites each contribute one untwisted eigenvalue.
    exp_val /= normalization * twisted_normalization * untwisted_lam**2
    # Retain only the dominant twisted mode across the bulk string.
    exp_val *= twisted_lam**string_length

    return exp_val
