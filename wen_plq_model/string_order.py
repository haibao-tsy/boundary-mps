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

print(f"ffCorrelator = {ffCorrelator:.3e}")

fbarfbarCorrelator = pepsExpectationValue(
    mpsLEFT=mpsAL_LEFT,
    mpsRIGHT=mpsAL_RIGHT,
    leftEnd=kron_all(Y, Z, Y, Z),
    stringBulk=kron_all(Z, Z, Z, Z),
    rightEnd=kron_all(Z, X, Z, X)
)

print(f"fbarfbarCorrelator = {fbarfbarCorrelator:.3e}")

eeCorrelator = pepsExpectationValue(
    mpsLEFT=mpsAL_LEFT,
    mpsRIGHT=mpsAL_RIGHT,
    leftEnd=kron_all(I, X, I, X),
    stringBulk=kron_all(Y, X, Y, X),
    rightEnd=kron_all(Y, I, Y, I)
)

print(f"eeCorrelator = {eeCorrelator:.3e}")

eCorrelator = pepsExpectationValue(
    mpsLEFT=mpsAL_LEFT,
    mpsRIGHT=mpsAL_RIGHT,
    leftEnd=kron_all(I, X, I, I),
    stringBulk=kron_all(Y, X, I, I),
    rightEnd=kron_all(Y, I, I, I)
)

print(f"eCorrelator = {eCorrelator:.3e}")

mmCorrelator = pepsExpectationValue(
    mpsLEFT=mpsAL_LEFT,
    mpsRIGHT=mpsAL_RIGHT,
    leftEnd=kron_all(X, Y, X, Y),
    stringBulk=kron_all(X, Y, X, Y),
    rightEnd=kron_all(X, Y, X, Y)
)

print(f"mmCorrelator = {mmCorrelator:.3e}")
mCorrelator = pepsExpectationValue(
    mpsLEFT=mpsAL_LEFT,
    mpsRIGHT=mpsAL_RIGHT,
    leftEnd=kron_all(X, Y, I, I),
    stringBulk=kron_all(X, Y, I, I),
    rightEnd=kron_all(X, Y, I, I)
)

print(f"mCorrelator = {mCorrelator:.3e}")

ebarebarCorrelator = pepsExpectationValue(
    mpsLEFT=mpsAL_LEFT,
    mpsRIGHT=mpsAL_RIGHT,
    leftEnd=kron_all(I, Y, I, Y),
    stringBulk=kron_all(X, Y, X, Y),
    rightEnd=kron_all(X, I, X, I)
)

print(f"ebarebarCorrelator = {ebarebarCorrelator:.3e}")

ebarCorrelator = pepsExpectationValue(
    mpsLEFT=mpsAL_LEFT,
    mpsRIGHT=mpsAL_RIGHT,
    leftEnd=kron_all(I, Y, I, I),
    stringBulk=kron_all(X, Y, I, I),
    rightEnd=kron_all(X, I, I, I)
)

print(f"ebarCorrelator = {ebarCorrelator:.3e}")

mbarmbarCorrelator = pepsExpectationValue(
    mpsLEFT=mpsAL_LEFT,
    mpsRIGHT=mpsAL_RIGHT,
    leftEnd=kron_all(Y, X, Y, X),
    stringBulk=kron_all(Y, X, Y, X),
    rightEnd=kron_all(Y, X, Y, X)
)

print(f"mbarmbarCorrelator = {mbarmbarCorrelator:.3e}")

mbarCorrelator = pepsExpectationValue(
    mpsLEFT=mpsAL_LEFT,
    mpsRIGHT=mpsAL_RIGHT,
    leftEnd=kron_all(Y, X, I, I),
    stringBulk=kron_all(Y, X, I, I),
    rightEnd=kron_all(Y, X, I, I)
)

print(f"mbarCorrelator = {mbarCorrelator:.3e}")
