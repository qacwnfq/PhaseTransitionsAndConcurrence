# Author Fredrik Jadebeck
#
# This is a collection of functions used to
# simulate quantum annealing with a transverse field
# and antiferromagnetic fluctuations.

import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from qutip import qeye, sigmax, sigmaz, entropy, states, tensor, ptrace
from scipy import optimize


from code.quantumMechanics import dm, Sx, Sz

# defines
P = 5


def listToDataFrame(column_name, l):
    """Wrapper to pandas dataframe"""
    df = pd.DataFrame({column_name: l})
    return df


def qutipConcurrence(s, lam, N=2, p=P):
    si = qeye(2)
    sx = sigmax()
    sz = sigmaz()
    sx_list = []
    sz_list = []
    for n in range(N):
        op_list = [si] * N
        op_list[n] = sx
        sx_list.append(tensor(op_list))
        op_list[n] = sz
        sz_list.append(tensor(op_list))
    # construct the hamiltonian
    VTF = 0
    # traverse field
    for n in range(N):
        VTF += -sx_list[n]
    H0 = 0
    # ferromagnet p model for p = 5
    for n in range(N):
        H0 += (1/N*sz_list[n])
    H0 = H0**p
    H0 = -H0*N
    # AFF
    VAFF = 0
    for n in range(N):
        VAFF += 1/N * sx_list[n]
    VAFF = N*VAFF**2
    H = s*lam*H0 + s*(1-lam)*VAFF + (1-s)*VTF
    evals, ekets = H.groundstate()
    rho = states.ket2dm(ekets)
    if N != 2:
        rho = ptrace(rho, [0, 1])
    c = entropy.concurrence(rho)
    return c


def cardaniac(A, B, C, D):
    """returns solutions in list ordered of A*x**3+B*x**2+C*x+D.
    This is used to compute 2 spin eigenenergies.
    """
    Delta = (27*A**2*D**2+4*B**3*D-18*A*B*C*D+4*A*C**3-B**2*C**2)/(108*A**4)
    p = (9*A*C-3*B**2)/(9*A**2)
    q = (2*B**3-9*A*B*C+27*A**2*D)/(27*A**3)
    if Delta == 0:
        if p == 0:
            solution = [-B/(3*A) for i in range(0, 3)]
        else:
            x1 = (B**3-4*A*B*C+9*A**2*D)/(3*A**2*C-A*B**2)
            x2 = (A*B*C-9*A**2*D)/(6*A**2*C-2*A*B**2)
            solution = [x1, x2, x2]
    elif Delta < 0:
        x1 = math.sqrt(-4./3*p)*math.cos(
            1./3*math.acos(-q/2*math.sqrt(-27/p**3))) - B/(3*A)
        x2 = -math.sqrt(-4./3*p)*math.cos(
            1./3*math.acos(-q/2*math.sqrt(-27/p**3)) + math.pi/3) - B/(3*A)
        x3 = -math.sqrt(-4./3*p)*math.cos(
            1./3*math.acos(-q/2*math.sqrt(-27/p**3)) - math.pi/3) - B/(3*A)
        solution = [x1, x2, x3]
    else:
        raise ValueError("Complex solutions not implemented yet.")
    solution.sort()
    return solution


def H0(N, p=P):
    """constructs and returns H0 block for N particles and
    maximal Spin"""
    assert(N > 0)
    S = float(N)/2
    H0 = -N*(Sz(N)/S)**p
    return H0


def Vtf(N):
    """constructs and returns transverse field Vtf block for N particles and
    maximal Spin"""
    assert(N > 0)
    S = float(N)/2
    Vtf = -N*(Sx(N)/S)
    return Vtf


def Vaff(N):
    """constructs and returns antiferromagnetic Vaff block for N particles and
    maximal Spin"""
    assert(N > 0)
    S = float(N)/2
    Vaff = N*(Sx(N)/S)**2
    return Vaff


def H0plusVtf(N, s):
    """Creates Hamiltonian from H0 and Vtf"""
    assert(N > 0)
    return s*H0(N) + (1-s)*Vtf(N)


def H0plusVaffplusVtf(N, s, l):
    """Creates Hamiltonian but with antiferromagnetic term"""
    assert(N > 0)
    return s*l*H0(N) + s*(1-l)*Vaff(N) + (1-s)*Vtf(N)


def diagonalize(H):
    """Wrapper for numpy method"""
    # use np to get only eigvals not eigenvectors of
    # symmetric or hermitian matrix. (H is always hermitian)
    return np.linalg.eigh(H)


# http://stackoverflow.com/questions/19820921/a-simple-matlab-like-way-of-finding-the-null-space-of-a-small-matrix-in-numpy
def null(a, rtol=1e-5):
    """Find null space of matrix. This is used to get eigenvectors
    from eigenvalues"""
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()


def getConcurrenceFromEigenvectors(eigenvalues, eigenvectors):
    """Returns concurrence from eigenvalues and eigenvectors"""
    rho, N = cket2dm(eigenvalues, eigenvectors)
    # N dimensions of eigenvectors means there are N-1 particles!!
    partial_rho = extract2qubitDm(rho, N-1)
    trace = np.trace(partial_rho)
    np.testing.assert_almost_equal(trace, 1.)
    c, lambdas = concurrence(partial_rho)
    return c


def concurrence(rho):
    """Takes a 2-particle density matrix and returns concurrence"""
    temp = rho.copy()
    rho[0][0] = -rho[0][2]
    rho[1][0] = -rho[1][2]
    rho[2][0] = -rho[2][2]
    rho[0][2] = -temp[0][0]
    rho[1][2] = -temp[1][0]
    rho[2][2] = -temp[2][0]
    rho = np.asmatrix(rho)
    R = rho*rho
    ev = np.linalg.eigvals(R)
    ev = np.real(ev)
    for e in ev:
        if abs(e) < 1.e-7:
            e = 0
    ev = [math.sqrt(abs(i)) for i in ev]
    ev = sorted(ev)
    ev2 = ev
    ev = max(0, ev[3] - ev[2] - ev[1] - ev[0])
    return ev, ev2


def extract2qubitDm(rho, N):
    """Calls partial trace on rho until it is a 2-particle
    density matrix and returns it."""
    partial_rho = 0
    if N == 2:
        partial_rho = rho
    else:
        partial_rho = dm(rho, N)
        for i in range(N-2):
            partial_rho = partial_rho.ptrace(0)
        partial_rho = partial_rho.nparray()
        temp = np.zeros(shape=(4, 4))
        for i in range(3):
            for j in range(3):
                temp[i][j] = partial_rho[i][j]
        partial_rho = temp
    return partial_rho


def cket2dm(ev, ket):
    """Custom implementation of qutips ket2dm"""
    ket = ket[:, 0]  # gets eigenvector corresponding to groundstate
    N = len(ket)
    rho = np.zeros(shape=(N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            try:
                rho[i][j] = ket[i]*ket[j]
            except:
                rho[i][j] = 0.
    return rho, N


def run(funH, N, s, l=None):
    """Get eigenvalues for N particles of the Hamiltonian H
    which is created by the function funH"""
    if l is None:
        H = funH(N, s)
    else:
        H = funH(N, s, l)
    ev, evs = diagonalize(H)
    return ev, evs


def classicalEnergy(s, l=None, p=P):
    """This function returns the results obtained by analytical calculation of
    the groundstate energy for the thermodynamic limit."""
    if l is None:
        x = optimize.minimize_scalar(
            lambda x: -s*(math.cos(x))**p - (1-s)*math.sin(x))
        if x.success is True:
            x = x.x
        else:
            raise RuntimeError("Minimizing failed.")
        return -s*(math.cos(x))**p - (1-s)*math.sin(x)
    else:
        # energy for qp phase
        qp = 2*s-s*l-1
        # energy for fm phase
        fm = 1000
        if l == 0 and s > 1./3:
            fm = -s*l - (1-s)**2/(4*s)
        elif s != 0:
            x = np.linspace(0, math.pi, 1001)
            f = [e(i, s, l, p) for i in x]
            fm = min(f)
        return min(qp, fm)


def e(x, s, l, p=P):
    """x is the minimum of the classical groundstate energy."""
    return (-s*l*math.sin(x)**p + s*(1-l)*math.cos(x)**2 -
            (1-s)*math.cos(x))


def gamma(x, s, l, p=P):
    """x is the minimum of the classical groundstate energy."""
    return (-0.5*s*l*p*(p-1)*math.sin(x)**(p-2)*math.cos(x)**2 +
            s*(1-l)*math.sin(x)**2)


def delta(x, s, l, p=P):
    """x is the minimum of the classical groundstate energy."""
    return (
        -s*l*(p*(p-1)*math.sin(x)**(p-2)*math.cos(x)**2-2*p*math.sin(x)**p) +
        s*(1-l)*(2*math.sin(x)**2-4*math.cos(x)**2) +
        2*(1-s)*math.cos(x))


def epsilon(x, s, l, p=P):
    """x is the minimum of the classical groundstate energy."""
    return (-2.*gamma(x, s, l, p) / delta(x, s, l, p))


def Delta(x, s, l, p=P):
    """x is the minimum of the classical groundstate energy."""
    return delta(x, s, l, p)*math.sqrt(1-epsilon(x, s, l, p)**2)


def calculateGap(s, l, p=P):
    """Uses analytical formula to calculate the Gap in the classical limit."""
    x = np.linspace(0, math.pi, 10001)
    f = [e(i, s, l, p) for i in x]
    f = {}
    for i in x:
        f[i] = e(i, s, l, p)
    fm = min(f, key=f.get)
    try:
        return Delta(fm, s, l, p)
    except ValueError:
        return 0


def calculateConcurrenceInLimit(s, l, p=P):
    """Returns the concurrence in the classical limit."""
    x = np.linspace(0, math.pi, 10001)
    f = [e(i, s, l, p) for i in x]
    f = {}
    for i in x:
        f[i] = e(i, s, l, p)
    fm = min(f, key=f.get)
    eps = epsilon(fm, s, l, p=p)
    alpha = math.sqrt((1-eps)/(1+eps))
    return (1-alpha)


def lambdaOne():
    """Calculates the groundstate energy for lambda=1"""
    s_list = np.linspace(0, 1, 1001)
    N_list = [2, 4, 8, 16, 32, 62]
    energies = {}
    for N in N_list:
        print(N)
        energies[N] = []
        for s in s_list:
            eigenvalues, eigenvectors = run(H0plusVtf, N, s)
            energies[N].append(sorted(eigenvalues)[0]/N)
    energies["oo"] = []
    for s in s_list:
        eigenvalues = classicalEnergy(s)
        energies["oo"].append(eigenvalues)
    for N in N_list:
        plt.plot(s_list, energies[N], label=str(N) + " spins")
    plt.plot(s_list, energies["oo"], label="Classical Large N Limit")
    plt.title("Energy e0 per spin")
    plt.xlabel("s")
    plt.ylabel("e0")
    plt.legend()
    plt.show()


def lambdaOneGap():
    """Calculates the energy Gap for lambda=1"""
    s_list = np.linspace(0, 1, 1001)
    N_list = [2, 4, 8, 16, 32, 62]
    energies = {}
    for N in N_list:
        print(N)
        energies[N] = []
        for s in s_list:
            eigenvalues, eigenvectors = run(H0plusVtf, N, s)
            energies[N].append((sorted(eigenvalues)[1] -
                                sorted(eigenvalues)[0])/N)
    energies["oo"] = []
    for s in s_list:
        eigenvalues = classicalEnergy(s)
        energies["oo"].append(eigenvalues)
    for N in N_list:
        plt.plot(s_list, energies[N], label=str(N) + " spins")
    # Figure out classical gap size
    # plt.plot(s_list, energies["oo"], label="Classical Large N Limit")
    plt.title("Gapsize g for p=" + str(P))
    plt.xlabel("s")
    plt.ylabel("g")
    plt.legend()
    plt.show()


def lambdaOneConcurrence():
    """ Calculates the concurrence for lambda=1"""
    s_list = np.linspace(0, 1, 101)
    N_list = [i for i in range(2, 4)]
    concurrence = {}
    concurrenceLimit = []
    concurrence3 = {}
    title = "../results/concurrence/p5/lambda1.csv"
    resultdf = pd.DataFrame()
    try:
        resultdf = pd.DataFrame.from_csv(title)
    except:
        pass
    for N in N_list:
        print("Running for " + str(N) + " spins.")
        concurrence[N] = []
        concurrence3[N] = []
        for s in s_list:
            eigenvalues, eigenvectors = run(H0plusVtf, N, s)
            c = getConcurrenceFromEigenvectors(eigenvalues, eigenvectors)
            concurrence[N].append(c*(N-1))
            # c3 = qutipConcurrence(s, 1, N=N)
            # concurrence3[N].append(c3)
        resultdf = resultdf.copy()
        resultdf[str(N)] = concurrence[N]
        resultdf.to_csv(title, sep=',')
    for s in s_list:
        concurrenceLimit.append(calculateConcurrenceInLimit(s, 1, p=P))
    for N in N_list:
        plt.plot(s_list, concurrence[N], label=str(N) + " spins", marker='^')
        # plt.plot(s_list, concurrence3[N], label=str(N) + " spins qutip",
        #          marker='v')
    plt.plot(s_list, concurrenceLimit, label="Classical Limit", marker='x')
    plt.title("Rescaled Concurrence Cr")
    plt.xlabel("s")
    plt.ylabel("c")
    plt.legend()
    plt.show()


def lambdaNotOne():
    """ Calculates the groundstate energy for lambda!=1"""
    s_list = np.linspace(0, 1, 1001)
    N_list = [2, 4, 8, 16, 32, 62]
    l_list = np.linspace(0, 1, 11)
    for l in l_list:
        energies = {}
        for N in N_list:
            print(N)
            energies[N] = []
            for s in s_list:
                eigenvalues, eigenvectors = run(H0plusVaffplusVtf, N, s, l)
                energies[N].append(min(eigenvalues)/N)
        energies["oo"] = []
        for s in s_list:
            eigenvalues = classicalEnergy(s, l=l)
            energies["oo"].append(eigenvalues)
        for N in N_list:
            plt.plot(s_list, energies[N], label=str(N))
        plt.plot(s_list, energies["oo"], label="inf")
        plt.title("Groundstate energy e0 per Spin for p= " +
                  str(P) + " and lambda=" + str(l))
        plt.xlabel("s")
        plt.ylabel("e0")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        plt.cla()


def lambdaNotOneGap():
    """ Calculates the energy Gap for lambda!=1"""
    s_list = np.linspace(0, 1, 1001)
    N_list = [2, 4, 8, 16, 32, 62]
    l_list = np.linspace(0, 1, 11)
    for l in l_list:
        energies = {}
        for N in N_list:
            print(N)
            energies[N] = []
            for s in s_list:
                eigenvalues, eigenvectors = run(H0plusVaffplusVtf, N, s, l)
                eigenvalues.sort()
                energies[N].append(eigenvalues[1]-eigenvalues[0])
        energies["oo"] = []
        for s in s_list:
            eigenvalues = calculateGap(s, l=l, p=P)
            energies["oo"].append(eigenvalues)
        for N in N_list:
            plt.plot(s_list, energies[N], label=str(N))
        plt.plot(s_list, energies["oo"], label="inf")
        plt.title("Gap g for p=" + str(P) + " and lambda=" + str(l))
        plt.xlabel("s")
        plt.ylabel("g")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        plt.cla()


def lambdaNotOneConcurrence():
    """ Calculates the concurrence for lambda!=1"""
    s_list = np.linspace(0, 1, 101)
    N_list = [i for i in range(8, 11)]
    l_list = np.linspace(0., 1, 6)
    l_list = [0.8]
    title = "../results/concurrence/p5/lambdaNot1.csv"
    resultdf = pd.DataFrame()
    try:
        resultdf = pd.DataFrame.from_csv(title)
    except:
        pass
    for l in l_list:
        concurrence = {}
        concurrence3 = {}
        for N in N_list:
            print(N)
            concurrence[N] = []
            concurrenceLimit = []
            concurrence3[N] = []
            for s in s_list:
                eigenvalues, eigenvectors = run(H0plusVaffplusVtf, N, s, l)
                c = getConcurrenceFromEigenvectors(eigenvalues, eigenvectors)
                concurrence[N].append(c*(N-1))
                # c3 = qutipConcurrence(s, l, N=N)
                # concurrence3[N].append(c3)
            resultdf = resultdf.copy()
            resultdf[str(N)] = concurrence[N]
            resultdf.to_csv(title, sep=',')
        for s in s_list:
            concurrenceLimit.append(calculateConcurrenceInLimit(s, l, p=P))
        maxC = 0
        for N in N_list:
            plt.plot(s_list, concurrence[N], label=str(N), marker="v",
                     linewidth=1)
            maxC = max(maxC, max(concurrence[N]))
            # if l != 0:
            # plt.plot(s_list, concurrence3[N], label=str(N), marker="^",
            #         linewidth=1)
        # plt.plot(s_list, concurrenceLimit, label="Limit", marker="x")
        # x = 1./(3-2*l)
        # plt.plot((x, x), (0, maxC), label=str("1/(3-2*lambda)"))
        plt.title("Concurrence for lambda=" + str(l))
        plt.xlabel("s")
        plt.ylabel("Rescaled Concurrence Cr")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        plt.cla()


def lambdaOne2Spins():
    """Compares numerical and analytical results for 2 spins and lambda=1."""
    s_list = np.linspace(0, 1, 1001)
    N_list = [2]
    energies = {}
    for N in N_list:
        print(N)
        energies[N] = []
        for s in s_list:
            eigenvalues, eigenvectors = run(H0plusVtf, N, s)
            energies[N].append(eigenvalues[0]/N)
    energies["a"] = []
    energies["oo"] = []
    analytical = []
    for s in s_list:
        l = 1.
        A = 1
        B = -((s+s*l)+2*s*(1-l)+(s-3*s*l))
        C = -(s**2*(1-l)**2+4*(s-1)**2-2*s*(1-l)*(s+s*l) -
              (s-3*s*l)*(s+s*l)-2*s*(1-l)*(s-3*s*l))
        D = (-2*s*(1-l)*(s-3*l*s)*(s+s*l)-4*(s-1)**2*s*(1-l) +
             2*s**3*(1-l)**3+2*(s-1)**2*(s-3*s*l)+2*(s+s*l)*(s-1)**2)
        analytical.append(cardaniac(A, B, C, D)[0]/2)
        eigenvalues = classicalEnergy(s)
        energies["oo"].append(eigenvalues)
        energies["a"].append(-math.sqrt(1-2*s+2*s**2))
    for N in N_list:
        plt.plot(s_list, energies[N], label=str(N) + " spins")
    plt.plot(s_list, energies["oo"], label="Classical Large N Limit")
    plt.plot(s_list, analytical, label="Analytical")
    plt.plot(s_list, energies["a"], label="Analytical2")
    plt.title("")
    plt.xlabel("s")
    plt.ylabel("Energy Per Spin")
    plt.legend()
    plt.show()


def lambdaNotOne2Spins():
    """Compares numerical and analytical results for 2 spins and lambda!=1."""
    s_list = np.linspace(0, 1, 1001)
    N_list = [2]
    l_list = np.linspace(0, 1, 11)
    for l in l_list:
        energies = {}
        analytical = {}
        for N in N_list:
            print(N)
            energies[N] = []
            for s in s_list:
                eigenvalues, eigenvectors = run(H0plusVaffplusVtf, N, s, l)
                energies[N].append(min(eigenvalues)/N)
        analytical[l] = []
        for s in s_list:
            A = 1
            B = -((s+s*l)+2*s*(1-l)+(s-3*s*l))
            C = -(s**2*(1-l)**2+4*(s-1)**2-2*s*(1-l)*(s+s*l) -
                  (s-3*s*l)*(s+s*l)-2*s*(1-l)*(s-3*s*l))
            D = (-2*s*(1-l)*(s-3*l*s)*(s+s*l)-4*(s-1)**2*s*(1-l) +
                 2*s**3*(1-l)**3+2*(s-1)**2*(s-3*s*l)+2*(s+s*l)*(s-1)**2)
            analytical[l].append(cardaniac(A, B, C, D)[0]/2)
        for N in N_list:
            plt.plot(s_list, energies[N], label=str(N))
        plt.plot(s_list, analytical[l], label="analytical")
        plt.xlabel("s")
        plt.ylabel("e0")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        plt.cla()


def getConcurrenceLimit(l):
    s_list = np.linspace(0, 1, 501)
    concurrenceLimit = []
    for s in s_list:
        concurrenceLimit.append(max(0, calculateConcurrenceInLimit(
            s, l, p=P)))
    title = ("../results/concurrence/p7/lambda" + str(l) +
             "limit.csv")
    resultdf = pd.DataFrame()
    try:
        resultdf = pd.DataFrame.from_csv(title)
    except:
        pass
    resultdf["s"] = s_list
    resultdf["cR"] = concurrenceLimit
    resultdf.to_csv(title)


if __name__ == "__main__":
    # lambdaOne()
    lambdaNotOne()
    # lambdaOne2Spins()
    # lambdaNotOne2Spins()
    # lambdaOneGap()
    lambdaNotOneGap()
    # lambdaOneConcurrence()
    # l_list = np.linspace(0., 1, 6)
    # for l in l_list:
    #     print(l)
    #     if l == 0:
    #         continue
    #     getConcurrenceLimit(l)
    # lambdaNotOneConcurrence()
