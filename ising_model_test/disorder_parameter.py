from order_parameter import *
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

length_of_string = 10000

disorder_para = np.real_if_close(
    disorder_para
)

disorder_para = disorder_para * twisted_lam ** length_of_string
print(f"disorder parameter = {disorder_para.real: .3f}")
print(f"theoretical disorder parameter = {disorder_para_theo: .3f}")

disorder_test, _= orderParameter(
    leftEnd=I,
    stringBulk=X,
    rightEnd=I,
    AC=AC,
    C=C,
    AL=AL,
    AR=AR
)

print(f"test = {disorder_test.real: .3f}")
