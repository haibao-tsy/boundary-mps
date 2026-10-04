from .order_parameter import *
from vumpsTransferMatrix.vumps import normalizedEnvironment


mpoX = X.reshape(1, 2, 2, 1)

twistedLeftEnv, twistedRightEnv, twisted_lam = normalizedEnvironment(
    mpoTensor=mpoX, 
    mpsLeft=AL,
    mpsRight=AR,
    C=C
)

twistedLeftEnv = twistedLeftEnv.reshape(chi, chi)
twistedRightEnv = twistedRightEnv.reshape(chi, chi)

disorder_para = ncon(
    [C.conj(), C, twistedRightEnv],
    [
        [1, 2],
        [1, 3],
        [2, 3]
    ]
)

disorder_para *= ncon(
    [C.conj(), C, twistedLeftEnv],
    [
        [1, 2],
        [3, 2],
        [1, 3]
    ]
)

disorder_para /= np.vdot(AC, AC)
if g < 1: disorder_para_theo = np.power(1 - np.power(g, 2), 1 / 4)
if g > 1: disorder_para_theo = 0

length_of_string = int(input("enter the length of string operator: "))

# Allow complex dtype promotion before removing negligible imaginary roundoff.
disorder_para = np.real_if_close(
    disorder_para * np.power(twisted_lam, length_of_string)
)
print(f"disorder parameter = {disorder_para: .3f}")
print(f"theoretical disorder parameter = {disorder_para_theo: .3f}")
print(f"twisted transfer matrix dominant eigenvalue λ = {twisted_lam: .3f}")
