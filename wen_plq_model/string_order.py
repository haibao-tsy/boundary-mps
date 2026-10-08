from non_fixed_point import *
mpsAL = AL.reshape(chiMPS, chi ** 4, chiMPS)
mpsAR = AR.reshape(chiMPS, chi ** 4, chiMPS)
mpsAC = AC.reshape(chiMPS, chi ** 4, chiMPS)

mpoZZ = (kron_all(Z, Z, Z, Z)).reshape(1, chi ** 4, chi ** 4, 1)

ZtwistedLeftEnv, ZtwistedRightEnv, Ztwisted_lam = normalizedEnvironment(
    mpoTensor=mpoZZ,
    mpsLeft=mpsAL,
    mpsRight=mpsAR,
    C=C
)

ZtwistedLeftEnv = ZtwistedLeftEnv.reshape(chiMPS, chiMPS)
ZtwistedRightEnv = ZtwistedRightEnv.reshape(chiMPS, chiMPS)

string_order = ncon(
    [mpsAC.conj(), kron_all(X, Z, X, Z), mpsAC, ZtwistedRightEnv],
    [
        [1, 2, 3],
        [2, 4],
        [1, 4, 5],
        [3, 5]
    ]
)

string_order *= ncon(
    [mpsAC.conj(), kron_all(Y, Z, Y, Z), mpsAC, ZtwistedLeftEnv],
    [
        [1, 2, 3],
        [2, 4],
        [5, 4, 3],
        [1, 5]
    ]
)

print(Ztwisted_lam)
string_order /= np.vdot(mpsAC, mpsAC)
string_order *= Ztwisted_lam ** 1000

print(string_order)
