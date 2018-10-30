import numpy as np
import matplotlib.pyplot as pl
import fluence


exp = 0.45
r = 0.1
u = np.array([0.4, 0.26])
b = np.linspace(-1.5, 1.5, 100)

f = fluence.flux(b, r, u)
fl = fluence.exact_fluence(b, r, u, exp)

fig, ax = pl.subplots(1, figsize=(8, 5))
ax.plot(b, f)
ax.plot(b, fl, 'k', lw=1)
ax.set_ylim(0.9875 , 1.0005)
pl.show()
