from matplotlib import pyplot as plt
import csv
import numpy as np

l_list = np.linspace(0, 1, 11)
for l in l_list:
    if l == 0:
        continue
    if l == 1:
        l = str(1)
    file = "../results/magnetization/MagnetizationLambda" + str(l) + "p5.csv"
    spamReader = csv.reader(open(file), delimiter=',', quotechar='|')
    mx = []
    mz = []
    s = []
    next(spamReader)
    next(spamReader)
    for row in spamReader:
        s.append(row[0])
        mx.append(row[1])
        mz.append(row[2])

    plt.grid(True)
    plt.plot(s, mx, marker='.', ls='None', label="mx, l=" + str(l))
    # plt.plot(s, mz, marker='.', ls='None', label="mz, l=" + str(l))
    plt.xlabel("s")

plt.title("Magnetization for p=5")
plt.legend()
plt.show()
