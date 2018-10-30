import numpy as np
import matplotlib.pyplot as pl
import fluence


exp = 0.45
r = 0.1
u = np.array([0.4, 0.26])


b = np.linspace(-1.5, 1.5, 100)
f = fluence.flux(b, r, u)

fl = fluence.exact_fluence(b, r, u, exp)
rf = fluence.riemann_fluence(b, r, u, exp, 50)
trf = fluence.trapezoid_fluence(b, r, u, exp, 50)
sf = fluence.simpson_fluence(b, r, u, exp, 8)
tf2 = fluence.taylor_fluence(b, r, u, exp, 2)
lrf = fluence.left_riemann_fluence(b, r, u, exp, 50)

fig, ax = pl.subplots(2, figsize=(8, 8))
ax[0].plot(b, f)
ax[0].plot(b, tf2)
ax[0].plot(b, rf)
ax[0].plot(b, trf)
ax[0].plot(b, sf)
ax[0].plot(b, lrf)
ax[0].plot(b, fl, 'k', lw=1)
ax[0].set_ylim(0.9875 , 1.0005)

ax[1].plot(b, np.abs(f - fl))
ax[1].plot(b, np.abs(tf2 - fl))
ax[1].plot(b, np.abs(rf - fl))
ax[1].plot(b, np.abs(trf - fl))
ax[1].plot(b, np.abs(sf - fl))
ax[1].plot(b, np.abs(lrf - fl))
ax[1].set_yscale("log")

# Plot the bounds

pl.show()
