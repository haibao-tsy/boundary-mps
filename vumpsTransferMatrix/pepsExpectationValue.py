import numpy as np
from .ncon import ncon
from scipy.sparse.linalg import eigs, LinearOperator

def pepsExpectationValue(
    mpsLEFT: np.ndarray,
    mpsRIGHT: np.ndarray,
    leftEnd: np.ndarray,
    stringBulk: np.ndarray,
    rightEnd: np.ndarray,
    tol: int=1e-14,
    string_length: int=1e3
):
    chiMPS = mpsLEFT.shape[0]

    def matvec(v):
        v = v.reshape(chiMPS, chiMPS)

        result = ncon(
            [mpsRIGHT.conj(), stringBulk, mpsLEFT, v],
            [
                [-1, 1, 3],
                [1, 2],
                [-2, 2, 4],
                [3, 4]
            ]
        )

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chiMPS * chiMPS, chiMPS * chiMPS),
        matvec=matvec,
        dtype=np.complex128
    )

    eigenvalues, eigenvectors = eigs(
        A=transferMatrix,
        k=1,
        which='LM',
        tol=tol
    )
    twisted_lam = eigenvalues[0]
    twisted_right_env = eigenvectors[:, 0].reshape(chiMPS, chiMPS)

    def matvec(v):
        v = v.reshape(chiMPS, chiMPS)

        result = ncon(
            [mpsRIGHT.conj(), stringBulk, mpsLEFT, v],
            [
                [3, 1, -1],
                [1, 2],
                [4, 2, -2],
                [3, 4]
            ]
        )

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chiMPS * chiMPS, chiMPS * chiMPS),
        matvec=matvec,
        dtype=np.complex128
    )

    _, eigenvectors = eigs(
        A=transferMatrix,
        k=1,
        which='LM',
        tol=tol
    )
    twisted_left_env = eigenvectors[:, 0].reshape(chiMPS, chiMPS)
    
    def matvec(v):
        v = v.reshape(chiMPS, chiMPS)

        result = ncon(
            [mpsRIGHT.conj(), mpsLEFT, v],
            [
                [-1, 1, 2],
                [-2, 1, 3],
                [2, 3]
            ]
        )

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chiMPS * chiMPS, chiMPS * chiMPS),
        matvec=matvec,
        dtype=np.complex128
    )

    eigenvalues, eigenvectors = eigs(
        A=transferMatrix,
        k=1,
        which='LM',
        tol=tol
    )
    untwisted_lam = eigenvalues[0]
    untwisted_right_env = eigenvectors[:, 0].reshape(chiMPS, chiMPS)

    def matvec(v):
        v = v.reshape(chiMPS, chiMPS)

        result = ncon(
            [mpsRIGHT.conj(), mpsLEFT, v],
            [
                [2, 1, -1],
                [3, 1, -2],
                [2, 3]
            ]
        )

        return result.ravel()

    transferMatrix = LinearOperator(
        shape=(chiMPS * chiMPS, chiMPS * chiMPS),
        matvec=matvec,
        dtype=np.complex128
    )

    _, eigenvectors = eigs(
        A=transferMatrix,
        k=1,
        which='LM',
        tol=tol
    )
    untwisted_left_env = eigenvectors[:, 0].reshape(chiMPS, chiMPS)

    twisted_lam /= untwisted_lam

    exp_val = ncon(
        [untwisted_left_env, mpsRIGHT.conj(), leftEnd, mpsLEFT, twisted_right_env],
        [
            [1, 2],
            [1, 3, 5],
            [3, 4],
            [2, 4, 6],
            [5, 6]
        ]
    )

    exp_val *= ncon(
        [twisted_left_env, mpsRIGHT.conj(), rightEnd, mpsLEFT, untwisted_right_env],
        [
            [1, 2],
            [1, 3, 5],
            [3, 4],
            [2, 4, 6],
            [5, 6]
        ]
    )

    normalization = ncon(
        [untwisted_left_env, untwisted_right_env],
        [
            [1, 2],
            [1, 2]
        ]
    )

    twisted_normalization = np.sum(
        twisted_left_env * twisted_right_env
    )

    exp_val /= (
        normalization * twisted_normalization * untwisted_lam**2
    )
    exp_val *= twisted_lam ** string_length

    return exp_val