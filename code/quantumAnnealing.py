# Author Fredrik Jadebeck
#
# This is a collection of functions used to
# to calculate limit behaviour of the model.
# They will be ported to C++ in the future.

import math
import numpy as np
import pandas as pd
from scipy import optimize

# defines
P = 5


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
        # Calculates energy for qp phase
        qp = 2*s-s*l-1
        # Finds energy for fm phase by minimizing.
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
    return (-0.5*(s*l*p*(p-1)*math.sin(x)**(p-2)*math.cos(x)**2 -
            2*s*(1-l)*math.sin(x)**2))


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


def getConcurrenceLimit(l):
    s_list = np.linspace(0, 1, 501)
    concurrenceLimit = []
    for s in s_list:
        concurrenceLimit.append(max(0, calculateConcurrenceInLimit(
            s, l, p=P)))
    title = ("../results/concurrence/p" + str(P) + "/lambda" + str(l) +
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
    l_list = np.linspace(0., 1, 6)
    l_list = list(l_list) + [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    print(l_list)
    for l in l_list:
        print(l)
        if l == 0:
            continue
        getConcurrenceLimit(l)
