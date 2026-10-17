from non_fixed_point import *
from vumpsTransferMatrix.pepsExpectationValue import pepsExpectationValue
mpsAL_LEFT = AL_LEFT.reshape(chiMPS, chi ** 4, chiMPS)

mpsAL_RIGHT = AL_RIGHT.reshape(chiMPS, chi ** 4, chiMPS)

ffCorrelator = pepsExpectationValue(
    mpsLEFT=mpsAL_LEFT,
    mpsRIGHT=mpsAL_RIGHT,
    leftEnd=kron_all(X, Z, X, Z),
    stringBulk=kron_all(Z, Z, Z, Z),
    rightEnd=kron_all(Z, Y, Z, Y)
)

print(f"ffCorrelator = {ffCorrelator}")

fbarfbarCorrelator = pepsExpectationValue(
    mpsLEFT=mpsAL_LEFT,
    mpsRIGHT=mpsAL_RIGHT,
    leftEnd=kron_all(Y, Z, Y, Z),
    stringBulk=kron_all(Z, Z, Z, Z),
    rightEnd=kron_all(Z, X, Z, X)
)

print(f"fbarfbarCorrelator = {fbarfbarCorrelator}")
