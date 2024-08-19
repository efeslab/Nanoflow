import sys

filename = sys.argv[1]

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(filename, header=None, names=['prefill', 'decode'])

df["sum"] = df["prefill"] + df["decode"]

df = df.drop(columns=["decode"])

df.plot()

# set y limit to 0 to 2100
plt.ylim(0, 2100)

plt.savefig(f"{filename}.png")