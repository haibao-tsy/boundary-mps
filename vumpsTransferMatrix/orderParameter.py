import numpy as np 
from .ncon import ncon
from .vumps import normalizedEnvironment

def orderParameter(
        leftEnd: np.ndarray,
        stringBulk: np.ndarray,
        rightEnd: np.ndarray,
        AC: np.ndarray,
        C: np.ndarray,
        AL: np.ndarray,
        AR: np.ndarray,
        stringLength: int = 1e3
):
    """Estimate a long-string expectation value in a uniform MPS.

    Approximates <leftEnd[0] stringBulk[1] ... stringBulk[n] rightEnd[n+1]>,
    where n = stringLength, using the dominant twisted transfer eigenmode.
    Identity endpoints give a plain string; identity bulk gives a
    long-distance two-point correlation.

    Parameters
    ----------
    leftEnd, stringBulk, rightEnd : ndarray, shape (d, d)
        Single-site operators with axes (bra physical, ket physical).
    AC, AL, AR : ndarray, shape (chi, d, chi)
        Center, left-canonical, and right-canonical MPS tensors, with axes
        (left bond, physical index, right bond).
    C : ndarray, shape (chi, chi)
        Center matrix satisfying AC[:, s, :] = AL[:, s, :] @ C
        = C @ AR[:, s, :] for each physical index s.
    stringLength : int, optional
        Number of bulk sites between the endpoints (default 1000).

    Returns
    -------
    order_para : complex
        Approximate expectation value, including twisted_lam**stringLength.
    twisted_lam : complex
        Dominant eigenvalue of the transfer channel with stringBulk inserted.

    Notes
    -----
    Assumes an injective, consistent mixed-canonical MPS, an isolated
    dominant twisted eigenmode, and nonzero environment overlap through C.
    Subleading transfer modes are omitted, so short strings are generally
    not evaluated exactly.
    """
    chi = AC.shape[0]
    d = AC.shape[1]

    twistedLeftEnv, twistedRightEnv, twisted_lam = normalizedEnvironment(
        mpoTensor=stringBulk.reshape(1, d, d, 1),
        mpsLeft=AL,
        mpsRight=AR,
        C=C
    )

    twistedLeftEnv = twistedLeftEnv.reshape(chi, chi)
    twistedRightEnv = twistedRightEnv.reshape(chi, chi)

    order_para = ncon(
        [AC.conj(), leftEnd, AC, twistedRightEnv],
        [
            [1, 2, 3],
            [2, 4],
            [1, 4, 5],
            [3, 5]
        ]
    )

    order_para *= ncon(
        [AC.conj(), rightEnd, AC, twistedLeftEnv],
        [
            [1, 2, 3],
            [2, 4],
            [5, 4, 3],
            [1, 5]
        ]
    )

    order_para /= np.vdot(AC, AC)
    order_para *= twisted_lam ** stringLength

    return order_para, twisted_lam
