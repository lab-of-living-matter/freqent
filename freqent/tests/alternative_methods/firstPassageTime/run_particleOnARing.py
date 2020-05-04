import numpy as np
import matplotlib.pyplot as plt
from particleOnARing import *

dt = 1e-5
x0 = np.random.rand() - 0.5
v = 1
D = 0.008
nsteps = 1e5
nsims = 20

p = particleOnARing(x0=x0, dt=dt, nsteps=nsteps)

fig, ax = plt.subplots()
for ii in range(nsims):
    p.runSimulation(v=v, D=D)
    ax.plot(p.t, p.pos, 'k-', alpha=0.1)

plt.show()
