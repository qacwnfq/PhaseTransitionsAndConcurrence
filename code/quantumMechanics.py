# Author Fredrik Jadebeck
#
# This is a collection of classes and functions revolving
# density matrices and their partial traces.
from itertools import permutations
import math
import numpy as np

from code.quantumAnnealing import gen_m


class dm:
    def __init__(self, rho, N):
        """takes a np.ndarray and constructs density matrix.
        The density matrix is a dictionary where they keys are
        the ketbras and the values are the amplitudes."""
        self.rho = {}
        self.N = N
        S = N/2
        ms = gen_m(N)
        for m in range(len(ms)):
            for n in range(len(ms)):
                self.rho[(ketBra(zeeman(N, S, ms[n]),
                                 zeeman(N, S, ms[m])))] = rho[n][m]

    def nparray(self):
        result = np.zeros(shape=(self.N+1, self.N+1))
        ms = gen_m(self.N)
        for m in range(len(ms)):
            for n in range(len(ms)):
                result[n][m] = (self.rho[
                    (ketBra(zeeman(self.N, self.N/2, ms[n]),
                            zeeman(self.N, self.N/2, ms[m])))])
        return result

    def ptrace(self, k):
        """traces out the kth spin"""
        result = {}
        ms = gen_m(self.N-1)
        for m in range(len(ms)):
            for n in range(len(ms)):
                result[(ketBra(zeeman(self.N-1, (self.N-1)/2, ms[n]),
                               zeeman(self.N-1, (self.N-1)/2, ms[m])))] = 0
        # calculates which terms add up
        for key, value in self.rho.items():
            # up
            newKet = None
            newKetFactor = 1
            newBra = None
            newBraFactor = 1
            for i in range(len(key.ket.state)):
                if key.ket.state[i][k] == 'd':
                    if newKet is None:
                        newstate = key.ket.state[i]
                        del newstate[k]
                        newKet = state(newstate)
                    else:
                        newstate = key.ket.state[i]
                        del newstate[k]
                        newKet += state(newstate)
            for i in range(len(key.bra.state)):
                if key.bra.state[i][k] == 'd':
                    if newBra is None:
                        newstate = key.bra.state[i]
                        del newstate[k]
                        newBra = state(newstate)
                    else:
                        newstate = key.bra.state[i]
                        del newstate[k]
                        newBra += state(newstate)
            if newKet is not None and newBra is not None:
                newKetFactor = newKet.nm/key.ket.nm
                newBraFactor = newBra.nm/key.bra.nm
                print(newKetFactor)
                print(newBraFactor)
                print(ketBra(newKet, newBra))
                result[(ketBra(newKet, newBra))] += (value *
                                                     newKetFactor *
                                                     newBraFactor)
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
                for k in range(len(i)):
                    if i[k] != j[k]:
                        result = 0
                        break
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
    assert(S == N/2)
    s = state(N*['u'])
    n = int(S - m)
    for i in range(n):
        s.state[0][i] = 'd'
    ps = []
    for p in permutations(s.state[0]):
        if p not in ps:
            ps.append(p)
    zstate = state(ps[0])
    for i in range(1, len(ps)):
        zstate += state(ps[i])
    return zstate

if __name__ == "__main__":
    print(gen_m(4))
    pass
