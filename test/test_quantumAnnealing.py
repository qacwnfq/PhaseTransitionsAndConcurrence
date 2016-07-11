import math
import numpy as np
from qutip import jmat
# qeye, sigmax, sigmay, sigmaz, tensor,

from code.quantumMechanics import gen_m, mSxn
# imports the functions to be tested
from code.quantumAnnealing import (H0, Vtf, Vaff,
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
