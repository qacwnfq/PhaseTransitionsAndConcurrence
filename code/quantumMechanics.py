# Author Fredrik Jadebeck
#
# This is a collection of classes and functions revolving
# density matrices and their partial traces as well as other
# basic quantum mechanical functions.

import math
import numpy as np
import pandas as pd


def gen_m(N):
    assert(N > 0)
    S = float(N)/2
    return [-S + i for i in range(N+1)]


def kronecker(m, n):
    """returns kronecker_{m,n}"""
    if n != m:
        return 0
    elif n == m:
        return 1
    else:
        raise RuntimeError("This can not happen.")


def mSxn(m, n, S):
    """returns matrixelement m,n of Sx for spin S"""
    assert(S > 0)
    return (kronecker(m, n+1) + kronecker(m+1, n))*0.5*math.sqrt(
        S*(S+1) - m*n)


def mSzn(m, n, S):
    """returns matrixelement m,n of Sz for spin S"""
    assert(S > 0)
    return kronecker(m, n)*m


def Sx(N):
    assert(N > 0)
    """constructs and returns Sx matrix block for N particles and
    maximal Spin"""
    S = float(N)/2
    # with this max spin following m exist
    m = gen_m(N)
    # most of the matrix are zeroes
    Sx = np.zeros(shape=(N+1, N+1))
    for i in range(0, N):
        Sx[i][i+1] = mSxn(m[i], m[i+1], S)
        Sx[i+1][i] = mSxn(m[i+1], m[i], S)
    return np.asmatrix(Sx)


def Sz(N):
    """constructs and returns Sz matrix block for N particles and
    maximal Spin"""
    assert(N > 0)
    S = float(N)/2
    m = gen_m(N)
    Sz = np.zeros(shape=(N+1, N+1))
    for i in range(N+1):
        Sz[i][i] = mSzn(m[i], m[i], S)
    return np.asmatrix(Sz)


class dm:
    def __init__(self, rho, N):
        """takes a np.ndarray and constructs density matrix.
        The density matrix is a dictionary where they keys are
        the ketbras and the values are the amplitudes."""
        self.rho = {}
        self.N = N
        S = N/2
        ms = gen_m(N)
        self.z = []
        for n in range(len(ms)):
            self.z.append(zeeman(N, S, ms[n]))
        for m in range(len(ms)):
            for n in range(len(ms)):
                self.rho[(ketBra(self.z[n], self.z[m]))] = rho[n][m]

    def nparray(self):
        result = np.zeros(shape=(self.N+1, self.N+1))
        ms = gen_m(self.N)
        for m in range(len(ms)):
            for n in range(len(ms)):
                result[n][m] = (self.rho[
                    (ketBra(self.z[n],
                            self.z[m]))])
        return result

    def ptrace(self, k):
        """traces out the kth spin"""
        result = {}
        ms = gen_m(self.N-1)
        self.z = []
        for n in range(len(ms)):
            self.z.append(zeeman(self.N-1, (self.N-1)/2, ms[n]))
        for m in range(len(ms)):
            for n in range(len(ms)):
                result[(ketBra(self.z[n], self.z[m]))] = 0
        # calculates which terms add up
        for key, value in self.rho.items():
            for ud in ['d', 'u']:
                newKet = None
                newKetFactor = 1
                newBra = None
                newBraFactor = 1
                newVal = 0
                for i in range(len(key.ket.state)):
                    if key.ket.state[i][k] == ud:
                        if newKet is None:
                            newstate = key.ket.state[i]
                            newstate = [newstate[i] for i
                                        in range(len(newstate)) if i != k]
                            newKet = state(newstate)
                        else:
                            newstate = key.ket.state[i]
                            newstate = [newstate[i] for i
                                        in range(len(newstate)) if i != k]
                            newKet += state(newstate)
                if newKet is None:
                    continue
                for i in range(len(key.bra.state)):
                    if key.bra.state[i][k] == ud:
                        if newBra is None:
                            newstate = key.bra.state[i]
                            newstate = [newstate[i] for i in
                                        range(len(newstate)) if i != k]
                            newBra = state(newstate)
                        else:
                            newstate = key.bra.state[i]
                            newstate = [newstate[i] for i in
                                        range(len(newstate)) if i != k]
                            newBra += state(newstate)
                if newKet is not None and newBra is not None:
                    newKetFactor = newKet.nm/key.ket.nm
                    newBraFactor = newBra.nm/key.bra.nm
                    newVal = value * newKetFactor * newBraFactor
                    result[(ketBra(newKet, newBra))] += newVal
        # updates N
        self.N -= 1
        self.rho = result
        return self

    def __str__(self):
        s = ""
        for i, v in self.rho.items():
            if math.isclose(self.rho[i], 0):
                continue
            else:
                s += (" + " + str(v) + "(" +
                      str(i.amplitude) + str(i) + ")")
        return s


class state:
    """class which describe qm state vector also known as ket"""
    def __init__(self, spins):
        """takes list of u and d like 'uud' to create a state
        of spin up up and down"""
        self.state = [[]]
        for i in range(len(spins)):
            # makes sure we are in up down basis
            assert(spins[i] == 'u' or spins[i] == 'd')
            self.state[0].append(spins[i])
        self.numVectors = 1
        self.nm = math.sqrt(1)

    def norm(self):
        norm = math.sqrt(self.numVectors)
        self.norm = norm
        return self.norm

    def scalar_product(self, ket):
        endResult = 0
        for i in self.state:
            for j in ket.state:
                result = 1
                if i != j:
                    result = 0
                endResult += result
        return endResult/(self.nm*ket.nm)

    def tensor_product(self, bra):
        assert(len(self.state) == len(bra.state))
        return ketBra(self, bra)

    def m(self):
        m = 0
        for val in self.state.values():
            if val == 'u':
                m += 0.5
            elif val == 'd':
                m -= 0.5
        return m

    def __eq__(self, other):
        result = False
        if math.isclose(self.scalar_product(other), 1.):
            result = True
        return result

    def __str__(self):
        return str(self.state)

    def __add__(self, other):
        newstate = state([])
        newstate.state = self.state + other.state
        newstate.numVectors = self.numVectors + other.numVectors
        newstate.nm = newstate.norm()
        return newstate


class ketBra:
    def __init__(self, ket, bra, amplitude=None):
        self.ket = ket
        self.bra = bra
        if amplitude is None:
            self.amplitude = 1./(self.ket.nm*self.bra.nm)

    def multiply_with(self, other):
        result = self.ket.tensor_product(other.bra)
        result.amplitude *= self.bra.scalar_product(other.ket)
        return result

    def __str__(self):
        return ("|" + str(self.ket) + "><" + str(self.bra) + "|")

    def __eq__(self, other):
        assert(type(self) == type(other))
        result = False
        if self.ket == other.ket and self.bra == other.bra:
            result = True
        return result

    def __hash__(self):
        h = self.ket.state[0] + self.bra.state[0]
        t = tuple(h)
        j = hash(t)
        return j


def zeeman(N, S, m):
    """returns a ket in the zeeman basis for N spins with total Spin
    S and magnetic order m"""
    # this function will only do the S=N/2 case for now.
    ps = zeeman_ext(N, S, m)
    zstate = state(ps[0])
    for i in range(1, len(ps)):
        zstate += state(ps[i])
    return zstate


def zeeman_ext(N, S, m):
    """Reads the zeeman basis data, which was created externally
    by a c++ function."""
    zeemanBasis = pd.DataFrame.from_csv("data/zeemanbasis.csv", sep=',')
    # Note that index is the number of spins by design.
    permutations = []
    # Iterates over the columns in a row.
    for index, row in zeemanBasis.iterrows():
        # Pandas reads index as str, but to compare N and index
        # they should both have the same type.
        if str(N) == index:
            # Converts the pandas.Series to a list.
            permutations = row.tolist()
            permutations = [x for x in permutations if str(x) != 'nan']
            break
    # Selects the permutations which belong to m.
    permutations = permutations[gen_m(N).index(m)]
    # Adjusts format to what is expected in the zeeman method above.
    permutations = permutations.split(" ")
    permutations.remove("")
    permutations = [list(i) for i in permutations]
    return permutations


if __name__ == "__main__":
    zeeman_ext(4, 2, 0)
