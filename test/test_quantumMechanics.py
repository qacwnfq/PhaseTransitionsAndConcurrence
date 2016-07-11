import math
import numpy as np

from code.quantumMechanics import (kronecker, gen_m,
                                   mSxn, mSzn, Sx,
                                   Sz, state, ketBra,
                                   zeeman, dm)


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


def test_states_and_ketBra():
    # 1) case
    s1 = state(['u', 'd', 'd', 'u'])
    s2 = state(['u', 'u', 'd', 'u'])
    expected = 0
    actual = s1.scalar_product(s2)
    assert(actual == expected)

    actual = s2.scalar_product(s2)
    expected = 1
    assert(actual == expected)

    actual = s1.scalar_product(s1)
    expected = 1
    assert(actual == expected)

    actual = s1.tensor_product(s2)
    expected = ketBra(s1, s2)
    assert(actual == expected)
    assert(expected.bra == s2)
    assert(expected.ket == s1)

    ketBra1 = ketBra(s1, s2)
    ketBra2 = ketBra(s1, s2)
    actual = ketBra1.multiply_with(ketBra2)
    actual = 0 if actual.amplitude == 0 else actual
    expected = 0
    assert(actual == expected)

    ketBra1 = ketBra(s1, s2)
    ketBra2 = ketBra(s2, s1)
    actual = ketBra1.multiply_with(ketBra2)
    actual = 0 if actual.amplitude == 0 else actual
    expected = ketBra(s1, s1)
    assert(actual == expected)

    # 2) case
    s1 = state(['u', 'd'])
    s2 = state(['u', 'd'])
    expected = 1
    actual = s1.scalar_product(s2)
    assert(actual == expected)

    # 3) case
    s1 = state(['u', 'd', 'd'])
    s2 = state(['u', 'd', 'd'])
    expected = 1
    actual = s1.scalar_product(s2)
    assert(actual == expected)

    # 4) case
    s1 = state([])
    s2 = state([])
    expected = 1
    actual = s1.scalar_product(s2)
    assert(actual == expected)

    # 5) case
    k1 = state(['u', 'd'])
    k2 = state(['d', 'u'])
    s1 = k1 + k2
    s2 = k2 + k1

    actual = s1.scalar_product(k2)
    expected = 1./math.sqrt(2)
    assert(actual == expected)

    actual = s1.scalar_product(s1)
    expected = 1
    np.testing.assert_almost_equal(actual, expected)

    actual = s1.scalar_product(s2)
    expected = 1
    np.testing.assert_almost_equal(actual, expected)


def test_zeeman():
    # 1) case
    actual = zeeman(2, 1, 1)
    expected = state(['u', 'u'])
    assert(actual == expected)
    assert(actual.state == expected.state)

    # 2) case
    actual = zeeman(2, 1, 0)
    expected = state(['d', 'u']) + state(['u', 'd'])
    assert(actual == expected)
    assert(actual.state == expected.state)
    np.testing.assert_almost_equal(actual.nm, expected.nm)

    # 3) case
    actual = zeeman(2, 1, -1)
    expected = state(['d', 'd'])
    assert(actual == expected)
    assert(actual.state == expected.state)

    # 4) case
    actual = zeeman(3, 1.5, -1.5)
    expected = state(['d', 'd', 'd'])
    assert(actual == expected)

    # 5) case
    actual = zeeman(4, 2, 0)
    expected = (state(['u', 'u', 'd', 'd']) +
                state(['d', 'd', 'u', 'u']) +
                state(['u', 'd', 'd', 'u']) +
                state(['d', 'u', 'u', 'd']) +
                state(['u', 'd', 'u', 'd']) +
                state(['d', 'u', 'd', 'u']))
    assert(actual == expected)


def test_dm():
    # 1) case
    expected = np.zeros(shape=(3, 3))
    numbers = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for i in range(3):
        for j in range(3):
            expected[i][j] = numbers[i][j]
    actual = dm(expected, 2)
    actual = actual.nparray()
    assert((actual == expected).all())

    # 2) case
    rho = np.zeros(shape=(3, 3))
    numbers = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for i in range(3):
        for j in range(3):
            rho[i][j] = numbers[i][j]
    actual = dm(rho, 2)
    actual = actual.ptrace(0)
    actual = actual.nparray()
    expected = np.zeros(shape=(2, 2))
    expected[0][0] = rho[0][0] + rho[1][1]/2
    expected[0][1] = 1/math.sqrt(2)*(rho[0][1]+rho[1][2])
    expected[1][0] = 1/math.sqrt(2)*(rho[1][0]+rho[2][1])
    expected[1][1] = rho[2][2] + rho[1][1]/2
    print(actual)
    print(expected)
    np.testing.assert_almost_equal(actual, expected)
