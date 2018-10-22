import numpy as np
import matplotlib.pyplot as pl
import starry


def F(b, r):
    if (b >= 1 + r):
        sphi = 0
        slam = 0
        cphi = 1
        clam = 1
        phi = -np.pi / 2
        lam = np.pi / 2
    elif (b <= 1 - r):
        sphi = 0
        slam = 0
        cphi = 1
        clam = 1
        phi = np.pi / 2
        lam = np.pi / 2
    else:
        sphi = (1 - r ** 2 - b ** 2) / (2 * b * r)
        slam = (1 - r ** 2 + b ** 2) / (2 * b)
        cphi = np.sqrt(1 - sphi ** 2)
        clam = np.sqrt(1 - slam ** 2)
        phi = np.arcsin(sphi)
        lam = np.arcsin(slam)
    return (np.pi / 2 + lam + clam * slam - r ** 2 * (np.pi / 2 + phi + cphi * sphi)) / np.pi

r = 0.3
x = np.linspace(-1.75, 1.75, 1000)
y = np.linspace(-0.3, 0.3, 1000)
b = np.sqrt(x ** 2 + y ** 2)
f = [F(bi, r) for bi in b]
pl.plot(x, f)

pl.show()
