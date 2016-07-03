# Author Fredrik Jadebeck
#
# This is a collection of classes and functions revolving
# density matrices and their partial traces.
from itertools import permutations
import math

from code.quantumAnnealing import gen_m


class dm:
    def __init__(self, rho, N):
        """takes a np.ndarray and constructs density matrix.
        The density matrix is a dictionary where they keys are
        the ketbras and the values are the amplitudes."""
        self.rho = {}
        S = N/2
        for m in gen_m(N):
            for n in gen_m(N):
                # Todo figure out mapping of rho onto zeeman
                self.rho[ketBra(zeeman(N, S, n), zeeman(N, S, m))] = rho

    def ptrace(self, k):
        """traces out the kth spin"""
        pass


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
    def __init__(self, ket, bra, amplitude=1):
        self.ket = ket
        self.bra = bra
        self.amplitude = amplitude

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


def zeeman(N, S, m):
    """returns a ket in the zeeman basis for N spins with total Spin
    S and magnetic order m"""
    # this function will only do the special case for now for simplicity
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
    # for p in ps:
    #    zstate += state(p)
    for i in range(1, len(ps)):
        zstate += state(ps[i])

    # print(str(1./zstate.nm) + " " + str(zstate))
    # print("")
    return zstate

if __name__ == "__main__":
    zeeman(3, 1.5, 1.5)
    zeeman(3, 1.5, .5)
    zeeman(3, 1.5, -.5)
    zeeman(3, 1.5, -1.5)
