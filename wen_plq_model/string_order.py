from non_fixed_point import *
from vumpsTransferMatrix.orderParameter import orderParameter
mpsAL = AL_LEFT.reshape(chiMPS, chi ** 4, chiMPS)
mpsAR = AR_LEFT.reshape(chiMPS, chi ** 4, chiMPS)
mpsAC = AC_LEFT.reshape(chiMPS, chi ** 4, chiMPS)

mpoZZ = (kron_all(Z, Z, Z, Z)).reshape(1, chi ** 4, chi ** 4, 1)

ZtwistedLeftEnv, ZtwistedRightEnv, Ztwisted_lam = normalizedEnvironment(
    mpoTensor=mpoZZ,
    mpsLeft=mpsAL,
    mpsRight=mpsAR,
    C=C_LEFT
)

ZtwistedLeftEnv = ZtwistedLeftEnv.reshape(chiMPS, chiMPS)
ZtwistedRightEnv = ZtwistedRightEnv.reshape(chiMPS, chiMPS)

# string_order = ncon(
#     [mpsAC.conj(), kron_all(X, Z, X, Z), mpsAC, ZtwistedRightEnv],
#     [
#         [1, 2, 3],
#         [2, 4],
#         [1, 4, 5],
#         [3, 5]
#     ]
# )

# string_order *= ncon(
#     [mpsAC.conj(), kron_all(Y, Z, Y, Z), mpsAC, ZtwistedLeftEnv],
#     [
#         [1, 2, 3],
#         [2, 4],
#         [5, 4, 3],
#         [1, 5]
#     ]
# )

string_order = orderParameter(
    leftEnd=kron_all(X, Z, X, Z),
    stringBulk=kron_all(Z, Z, Z, Z),
    rightEnd=kron_all(Y, Z, Y, Z),
    AC=mpsAC,
    C=C_LEFT,
    AL=mpsAL,
    AR=mpsAR,
    stringLength=1e5
)
# print(np.real_if_close(Ztwisted_lam))

print(np.real_if_close(string_order))
