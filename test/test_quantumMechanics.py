import math
import numpy as np

from code.quantumMechanics import state, ketBra, zeeman


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
