import numpy as np
import matplotlib.pyplot as pl
import fluence
from scipy.integrate import quad
from tqdm import tqdm


def taylorFluence(exp, f, f2, f4=None, f6=None, f8=None):
    return f + (1 / 24.) * exp ** 2 * f2 + \
               ((1 / 1920.) * exp ** 4 * f4 if f4 is not None else 0) + \
               ((1 / 322560.) * exp ** 6 * f6 if f6 is not None else 0) + \
               ((1 / 92897280.) * exp ** 8 * f8 if f8 is not None else 0)


def riemannFluence(exp, b, f):
    brange = b[-1] - b[0]
    nbins = int(brange / exp)
    ff = f.reshape(nbins, -1)
    bb = b.reshape(nbins, -1)
    fluence = np.mean(ff, axis=1)
    return np.mean(bb, axis=1), fluence


exp = 0.75
r = 0.05
u = np.array([0.4, 0.26])
b = np.linspace(-1.5, 1.5, 300)
data = fluence.compute(b, r, u, exp)
f = data[:, 0]
qf = data[:, 1]
f2 = data[:, 2]
f4 = data[:, 3]
f6 = data[:, 4]
f8 = data[:, 5]

tf2 = taylorFluence(exp, f, f2)
tf4 = taylorFluence(exp, f, f2, f4)
tf6 = taylorFluence(exp, f, f2, f4, f6)
tf8 = taylorFluence(exp, f, f2, f4, f6, f8)

fig, ax = pl.subplots(2, figsize=(8, 8))
ax[0].plot(b, f)
ax[0].plot(b, tf2)
ax[0].plot(b, tf4)
ax[0].plot(b, tf6)
ax[0].plot(b, tf8)
ax[0].plot(b, qf, 'k', lw=1)
ax[0].set_ylim(0.9965, 1.0005)

ax[1].plot(b, np.abs(f - qf))
ax[1].plot(b, np.abs(tf2 - qf))
ax[1].plot(b, np.abs(tf4 - qf))
ax[1].plot(b, np.abs(tf6 - qf))
ax[1].plot(b, np.abs(tf8 - qf))
ax[1].set_yscale("log")

for axis in ax:
    axis.axvline(-1 - r - exp / 2., ls='--', color='k', lw=1)
    axis.axvline(-1 + r + exp / 2., ls='--', color='k', lw=1)
    axis.axvline(1 - r - exp / 2., ls='--', color='k', lw=1)
    axis.axvline(1 + r + exp / 2., ls='--', color='k', lw=1)

pl.show()
