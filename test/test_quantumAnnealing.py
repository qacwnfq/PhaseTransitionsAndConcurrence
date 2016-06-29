import math
import numpy as np
from qutip import jmat
# qeye, sigmax, sigmay, sigmaz, tensor, 

from code.quantumAnnealing import (kronecker, gen_m, mSxn, mSzn,
                                   Sx, Sz, H0, Vtf, Vaff,
                                   H0plusVtf, H0plusVaffplusVtf,
                                   diagonalize, cardaniac)


def test_cardaniac():
    A = 1
    B = 0
    C = 0
    D = 0
    sol = cardaniac(A, B, C, D)
    expected = [0, 0, 0]
    assert((sol == expected))
    A = 1
    B = -4
    C = 5
    D = -2
    sol = cardaniac(A, B, C, D)
    expected = [1, 1, 2]
    assert((sol == expected))
    A = 1
    B = 0
    C = -1
    D = 0
    sol = cardaniac(A, B, C, D)
    expected = [-1, 0, 1]
    np.testing.assert_almost_equal(sol, expected)
    A = 1
    B = -7.5
    C = 17.75
    D = -13.125
    sol = cardaniac(A, B, C, D)
    expected = [1.5, 2.5, 3.5]
    np.testing.assert_almost_equal(sol, expected)
    A = 1
    B = 2.5
    C = -7.25
    D = -13.125
    sol = cardaniac(A, B, C, D)
    expected = [-3.5, -1.5, 2.5]
    np.testing.assert_almost_equal(sol, expected)
    A = 1
    B = 38.1
    C = 205.918
    D = -789.416
    sol = cardaniac(A, B, C, D)
    expected = [-30.5, -10.15, 2.55]
    np.testing.assert_almost_equal(sol, expected, decimal=1)


def test_kronecker():
    m = [1, 11, -345, 3451, 0, 0.5, 123, 1.5, 1.5, 1.7, 55.5]
    n = [1, 10, 24, 123, 0, 0.5, 123, 1.5, 2.5, 1.5, 55.5]
    expected = [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1]
    actual = []
    reflectiontest = []
    for i in range(len(m)):
        actual.append(kronecker(m[i], n[i]))
        reflectiontest.append(kronecker(n[i], n[i]))
    assert(actual == expected)
    assert(reflectiontest == [1 for i in range(len(n))])


def test_gen_m():
    actual = gen_m(5)
    expected = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    assert(actual == expected)


def test_mSxn():
    assert(0 == mSxn(0, 3, 6))
    assert(0.5*math.sqrt(4**2+4-2) == mSxn(1, 2, 4))
    assert(0.5*math.sqrt(10**2+10-2) == mSxn(1, 2, 10))
    assert(0.5*math.sqrt(100**2+100-99*100) == mSxn(99, 100, 100))
    assert(1 == mSxn(2, 1, 2))
    assert(1 == mSxn(-2, -1, 2))
    assert(0.5*math.sqrt(2) == mSxn(0, 1, 1))


def test_mSzn():
    assert(0 == mSzn(0, 1, 5))
    assert(0 == mSzn(99, 6, 100))
    assert(100 == mSzn(100, 100, 100))
    assert(100 == mSzn(100, 100, 1090))
    assert(3 == mSzn(3, 3, 10))
    assert(-10 == mSzn(-10, -10, 100))
    assert(0 == mSzn(0, 0, 10000))


def test_Sx():
    # test one particle
    expected = np.zeros(shape=(2, 2))
    expected[1][0] = 0.5
    expected[0][1] = 0.5
    expected = np.asmatrix(expected)
    actual = Sx(1)
    assert((expected == actual).all())
    # test two particles
    expected = np.zeros(shape=(3, 3))
    expected[0][1] = 1/math.sqrt(2)
    expected[1][0] = 1/math.sqrt(2)
    expected[1][2] = 1/math.sqrt(2)
    expected[2][1] = 1/math.sqrt(2)
    expected = np.asmatrix(expected)
    actual = Sx(2)
    np.testing.assert_almost_equal(actual, expected)
    # test three particles
    expected = np.zeros(shape=(4, 4))
    expected[0][1] = math.sqrt(3)
    expected[1][0] = math.sqrt(3)
    expected[1][2] = 2
    expected[2][1] = 2
    expected[2][3] = math.sqrt(3)
    expected[3][2] = math.sqrt(3)
    expected = expected*0.5
    expected = np.asmatrix(expected)
    actual = Sx(3)
    np.testing.assert_almost_equal(actual, expected)
    # test four particles
    expected = np.zeros(shape=(5, 5))
    expected[0][1] = 2
    expected[1][0] = 2
    expected[1][2] = math.sqrt(6)
    expected[2][1] = math.sqrt(6)
    expected[2][3] = math.sqrt(6)
    expected[3][2] = math.sqrt(6)
    expected[3][4] = 2
    expected[4][3] = 2
    expected = expected*0.5
    expected = np.asmatrix(expected)
    actual = Sx(4)
    np.testing.assert_almost_equal(actual, expected)
    # test 64 particles
    N = 64
    expected = np.zeros(shape=(N+1, N+1))
    S = float(N)/2
    # with this max spin following m exist
    m = [-S + i for i in range(N+1)]
    for i in range(0, N):
        expected[i][i+1] = mSxn(m[i], m[i+1], S)
        expected[i+1][i] = mSxn(m[i+1], m[i], S)
    expected = np.asmatrix(expected)
    actual = Sx(N)
    np.testing.assert_almost_equal(actual, expected)


def test_Sz():
    Ns = [2, 3, 4, 64]
    for N in Ns:
        expected = np.zeros(shape=(N+1, N+1))
        S = float(N)/2
        # with this max spin following m exist
        m = [-S + i for i in range(N+1)]
        for i in range(len(m)):
            expected[i][i] = mSzn(m[i], m[i], S)
        expected = np.asmatrix(expected)
        actual = Sz(N)
        np.testing.assert_almost_equal(actual, expected)


def test_H0():
    Ns = [2, 3, 4, 64]
    for N in Ns:
        p = 5
        S = float(N)/2
        m = gen_m(N)
        expected = np.zeros(shape=(N+1, N+1))
        for i in range(N+1):
            expected[i][i] = -N*((m[i]/(S))**p)
        actual = H0(N, p=p)
        np.testing.assert_almost_equal(actual, expected)


def test_Vtf():
    Ns = [2, 3, 4, 64]
    for N in Ns:
        S = float(N)/2
        m = gen_m(N)
        expected = np.zeros(shape=(N+1, N+1))
        for i in range(0, N):
            expected[i][i+1] = -N*mSxn(m[i], m[i+1], S)/S
            expected[i+1][i] = -N*mSxn(m[i+1], m[i], S)/S
        expected = np.asmatrix(expected)
        actual = Vtf(N)
        np.testing.assert_almost_equal(actual, expected)


def test_Vaff():
    # per hand testing to find bugs
    expected = np.zeros(shape=(3, 3))
    expected[0][0] = 1
    expected[2][2] = 1
    expected[0][2] = 1
    expected[2][0] = 1
    expected[1][1] = 2
    expected = np.asmatrix(expected)
    actual = Vaff(2)
    np.testing.assert_almost_equal(actual, expected)
    # this keeps new bugs from sneaking in
    Ns = [2, 3, 4, 64]
    for N in Ns:
        S = float(N)/2
        m = gen_m(N)
        expected = np.zeros(shape=(N+1, N+1))
        for i in range(0, N):
            expected[i][i+1] = (mSxn(m[i], m[i+1], S)/S)
            expected[i+1][i] = (mSxn(m[i+1], m[i], S)/S)
        expected = np.asmatrix(expected)
        expected = N*expected**2
        actual = Vaff(N)
        np.testing.assert_almost_equal(actual, expected)


def test_H0plusVtf():
    N = 32
    s = 0.33
    expected = s*H0(N)+(1-s)*Vtf(N)
    actual = H0plusVtf(N, s)
    assert((actual == expected).all())


def test_H0plusVaffplusVtf():
    N = 31
    s = 0.67
    l = 0.21
    expected = s*l*H0(N) + s*(1-l)*Vaff(N) + (1-s)*Vtf(N)
    actual = H0plusVaffplusVtf(N, s, l)
    assert((actual == expected).all())


def test_diagonalize():
    N = 2
    s = 1
    expected = [-2, 0, 2]
    actual, e = diagonalize(H0plusVtf(N, s))
    assert((actual == expected).all())
    N = 2
    s = 0
    expected = [-2, 0, 2]
    actual, e = diagonalize(H0plusVtf(N, s))
    np.testing.assert_almost_equal(actual, expected)
    N = 8
    S = float(N)/2
    s = 0.5
    l = 0.2
    # spin operators for qutip to compare
    S_list = jmat(S)
    Sx = S_list[0]
    Sz = S_list[2]
    # construct the hamiltonian
    Vtf = -N*(Sx/S)
    Vaff = N*(Sx/S)**2
    H0 = -N*(Sz/S)**5
    H = s*l*H0 + s*(1-l)*Vaff + (1-s)*Vtf
    expected = H.eigenenergies()
    actualH = H0plusVaffplusVtf(N, s, l)
    actual, e = diagonalize(actualH)
    np.testing.assert_almost_equal(actual, expected)
