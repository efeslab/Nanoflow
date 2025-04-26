import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

df = pd.read_csv('./allgather/8.csv')
x_df = df["size"].values
y_df = df["avg_time"].values

f_cubic_df = interp1d(x_df, y_df, kind='linear', fill_value='extrapolate')
xnew_df = np.linspace(min(x_df)/2, max(x_df), num=100, endpoint=True)
ynew_cubic_df = f_cubic_df(xnew_df)


plt.figure(figsize=(10, 6))
plt.plot(x_df, y_df, 'o', label='Original data')
plt.plot(xnew_df, ynew_cubic_df, '-', label='Cubic spline interpolation')
plt.legend()
plt.xlabel('Size')
plt.ylabel('Avg Time')

plt.savefig('interpolate.png')