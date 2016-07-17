from matplotlib import pyplot as plt
import csv
file = "../results/magnetization/MagnetizationLambda0.8p5.csv"

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

plt.plot(s, mx, "mx")
plt.plot(s, mz, "")
plt.xlabel("s")
plt.ylabel("mx")
plt.title("lambda 0.8")
plt.show()
