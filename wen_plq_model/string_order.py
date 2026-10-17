from non_fixed_point import *
from vumpsTransferMatrix.orderParameter import orderParameter
mpsAL_LEFT = AL_LEFT.reshape(chiMPS, chi ** 4, chiMPS)
mpsAR_LEFT = AR_LEFT.reshape(chiMPS, chi ** 4, chiMPS)
mpsAC_LEFT = AC_LEFT.reshape(chiMPS, chi ** 4, chiMPS)

mpsAL_RIGHT = AL_RIGHT.reshape(chiMPS, chi ** 4, chiMPS)
mpsAR_RIGHT = AR_RIGHT.reshape(chiMPS, chi ** 4, chiMPS)
mpsAC_RIGHT = AC_RIGHT.reshape(chiMPS, chi ** 4, chiMPS)

mpoZZ = (kron_all(Z, Z, Z, Z)).reshape(1, chi ** 4, chi ** 4, 1)

