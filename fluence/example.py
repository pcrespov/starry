import numpy as np
import matplotlib.pyplot as pl
import fluence


exp = 0.45
r = 0.1
u = np.array([0.4, 0.26])


b = np.linspace(0.21, 1.5, 300)
f = fluence.flux(b, r, u)

qf = fluence.fluence(b, r, u, exp, 99, 1)
tf21 = fluence.fluence(b, r, u, exp, 2, 1)
tf24 = fluence.fluence(b, r, u, exp, 2, 4)
tf28 = fluence.fluence(b, r, u, exp, 2, 8)

fig, ax = pl.subplots(2, figsize=(8, 8))
ax[0].plot(b, f)
ax[0].plot(b, tf21)
ax[0].plot(b, tf24)
ax[0].plot(b, tf28)
ax[0].plot(b, qf, 'k', lw=1)
ax[0].set_ylim(0.9875 , 1.0005)

ax[1].plot(b, np.abs(f - qf))
ax[1].plot(b, np.abs(tf21 - qf))
ax[1].plot(b, np.abs(tf24 - qf))
ax[1].plot(b, np.abs(tf28 - qf))
ax[1].set_yscale("log")

# Bounds:
e = 0.5 * exp
P = 1 - r
Q = 1 + r
A = P - e
B = P + e
C = Q - e
D = Q + e
for axis in ax:
    axis.axvspan(A, B, color='r', zorder=-99, alpha=0.05)
    axis.axvspan(C, D, color='r', zorder=-99, alpha=0.05)
    axis.axvline(P, color='k', ls='--', lw=1, alpha=0.5)
    axis.axvline(Q, color='k', ls='--', lw=1, alpha=0.5)

pl.show()
