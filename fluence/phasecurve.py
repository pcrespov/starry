import numpy as np
import matplotlib.pyplot as pl
import fluence
import starry

# Time array
time = np.linspace(0, 2, 1000)

# Flux
map = starry.Map(lmax=20)
map.load_image("earth")
map.axis = [0, 1, 0]
t0 = 0
theta0 = 0
per = 1
theta = theta0 + 360. / per * (time - t0)
f = map.flux(theta=theta)

# Fluence
exp = 0.01
fl = fluence.phase_curve_fluence(time, np.array(map.y), np.array(map.axis), per, t0, theta0, exp)

# Plot
fig, ax = pl.subplots(2, figsize=(8, 8))
ax[0].plot(time, f, lw=1)
ax[0].plot(time, fl, 'k', lw=1)

ax[1].plot(time, f - fl)
pl.show()
