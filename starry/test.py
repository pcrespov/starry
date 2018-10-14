import numpy as np
import matplotlib.pyplot as pl

b, f, f2, f4, f6 = np.loadtxt("test.txt").transpose()
fig, ax = pl.subplots(4, figsize=(5, 8))
ax[0].plot(b, f)
ax[1].plot(b, f2)
ax[2].plot(b, f4)
ax[3].plot(b, f6)
for axis in ax:
    axis.set_yscale("log")
pl.show()
