from code.quantumMechanics import state, ketBra


def test_state():
    # 1) case
    s1 = state(['u', 'd', 'd', 'u'])
    s2 = state(['u', 'u', 'd', 'u'])
    expected = 0
    actual = s1.scalar_product(s2)
    assert(actual == expected)

    expected = 1
    actual = s2.scalar_product(s2)
    assert(actual == expected)
    expected = 1
    actual = s1.scalar_product(s1)
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
    print(ketBra1)
    print(ketBra2)
    actual = ketBra1.multiply_with(ketBra2)
    actual = 0 if actual.amplitude == 0 else actual
    expected = ketBra(s1, s1)
    print("actual")
    print(actual)
    print(expected)
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
