# Author Fredrik Jadebeck
#
# This is a collection of classes and functions revolving
# density matrices and their partial traces.
from itertools import combinations
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
            # Todo figure out mapping of rho onto zeeman
            self.rho[zeeman(N, S, m)] = rho

    def ptrace(self, k):
        """traces out the kth spin"""
        pass


class state:
    """class which describe qm state vector also known as ket"""
    def __init__(self, spins):
        """takes list of u and d like 'uud' to create a state
        of spin up up and down"""
        if len(spins) == 0:
            print("No spins.")
        # self.state = {}
        self.state = []
        for i in range(len(spins)):
            # makes sure we are in up down basis
            assert(spins[i] == 'u' or spins[i] == 'd')
            # self.state[i] = spins[i]
            self.state.append(spins[i])
        self.norm = math.sqrt(1)

    def scalar_product(self, ket):
        assert(len(self.state) == len(ket.state))
        result = 1
        for i in range(len(self.state)):
            if self.state[i] != ket.state[i]:
                result = 0
                break
        return result

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
        if self.scalar_product(other) == 1:
            result = True
        return result

    def __str__(self):
        return str(self.state)


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
    # takes string and returns string.
    # find this function for lists or
    # replace used lists by strings. Second option might be easier
    s = state(['u', 'd', 'u'])
    print(s.stat)
    p = list(combinations([0, 1, 1], 2))
    print(p)


if __name__ == "__main__":
    zeeman(4, 2, 4)
