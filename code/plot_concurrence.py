from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

file = "../results/concurrence/p5/lambdaNot1.csv"

df = pd.DataFrame.from_csv(file)
s = np.linspace(0, 1, 101)

print(df.columns.values)
for N in df.columns.values:
    print(N)
    plt.plot(s, df[N].tolist())
plt.xlabel("s")
plt.ylabel("Rescaled concurrence")
plt.title("lambda 1")
plt.show()
