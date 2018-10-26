import numpy as np
import matplotlib.pyplot as pl
import fluence


exp = 0.03
r = 0.1
u = np.array([0.4, 0.26])


b = np.linspace(0.05, 1.5, 300)
data = fluence.compute(b, r, u, exp)
f = data[:, 0]
qf = data[:, 1]
tf2 = data[:, 2]
tf4 = data[:, 3]
tf6 = data[:, 4]
tf8 = data[:, 5]

fig, ax = pl.subplots(2, figsize=(8, 8))
ax[0].plot(b, f)
ax[0].plot(b, tf2)
ax[0].plot(b, tf4)
ax[0].plot(b, tf6)
ax[0].plot(b, tf8)
ax[0].plot(b, qf, 'k', lw=1)
ax[0].set_ylim(0.9875 , 1.0005)

ax[1].plot(b, np.abs(f - qf))
ax[1].plot(b, np.abs(tf2 - qf))
ax[1].plot(b, np.abs(tf4 - qf))
ax[1].plot(b, np.abs(tf6 - qf))
ax[1].plot(b, np.abs(tf8 - qf))
ax[1].set_yscale("log")

for axis in ax:
    axis.axvline(1 - r - exp / 2., ls='--', color='k', lw=1)
    axis.axvline(1 - r + exp / 2., ls='--', color='k', lw=1)
    axis.axvline(1 + r - exp / 2., ls='--', color='k', lw=1)
    axis.axvline(1 + r + exp / 2., ls='--', color='k', lw=1)

pl.show()
