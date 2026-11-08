from physical_translation import *
from tensor_equation_solver import kron_all
from vumpsTransferMatrix.pepsExpectationValue import pepsExpectationValue

ffCorrelator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(X, Z, X, Z),
    stringBulk=kron_all(Z, Z, Z, Z),
    rightEnd=kron_all(Z, Y, Z, Y),
)


fbarfbarCorrelator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(Y, Z, Y, Z),
    stringBulk=kron_all(Z, Z, Z, Z),
    rightEnd=kron_all(Z, X, Z, X),
)


eeCorrelator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(I, X, I, X),
    stringBulk=kron_all(Y, X, Y, X),
    rightEnd=kron_all(Y, I, Y, I),
)


eCorrelator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(I, X, I, I),
    stringBulk=kron_all(Y, X, I, I),
    rightEnd=kron_all(Y, I, I, I),
)


mmCorrelator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(X, Y, X, Y),
    stringBulk=kron_all(X, Y, X, Y),
    rightEnd=kron_all(X, Y, X, Y),
)

mCorrelator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(X, Y, I, I),
    stringBulk=kron_all(X, Y, I, I),
    rightEnd=kron_all(X, Y, I, I),
)


ebarebarCorrelator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(I, Y, I, Y),
    stringBulk=kron_all(X, Y, X, Y),
    rightEnd=kron_all(X, I, X, I),
)


ebarCorrelator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(I, Y, I, I),
    stringBulk=kron_all(X, Y, I, I),
    rightEnd=kron_all(X, I, I, I),
)


mbarmbarCorrelator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(Y, X, Y, X),
    stringBulk=kron_all(Y, X, Y, X),
    rightEnd=kron_all(Y, X, Y, X),
)


mbarCorrelator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(Y, X, I, I),
    stringBulk=kron_all(Y, X, I, I),
    rightEnd=kron_all(Y, X, I, I),
)

e1Correlator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(I, I, I, Y),
    stringBulk=kron_all(I, I, I, I),
    rightEnd=kron_all(I, I, X, I),
)

e1e1Correlator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(I, Y, I, Y),
    stringBulk=kron_all(I, I, I, I),
    rightEnd=kron_all(X, I, X, I),
)

e2Correlator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(I, X, I, I),
    stringBulk=kron_all(I, I, I, I),
    rightEnd=kron_all(Y, I, I, I),
)

e2e2Correlator = pepsExpectationValue(
    mpsLEFT=AL_LEFT,
    mpsRIGHT=AL_RIGHT,
    leftEnd=kron_all(I, X, I, X),
    stringBulk=kron_all(I, I, I, I),
    rightEnd=kron_all(Y, I, Y, I),
)
