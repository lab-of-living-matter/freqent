# Dissipation in frequency space
This repository contains code to calculate entropy production rates from times series using a formulation based on correlation functions in frequency space. The code to actually calculate the entropy is written as a module for easy use and modularity. The module is easily installed using `pip` as follows (this is done in the terminal):

```bash
cd path/to/this/repo
pip install -e .
```

The entropy calculation requires that you first have a data set ready in the form of an _NxM_ numpy array. _N_ is the number of variables and _M_ is the length of the time series for each variable. In other words, if the array `x` contains the data, `x[n]` gives the time series of the _nth_ variable.

There is also a simulation of a Brownian particle in a non-conservative force field to test the entropy calculations with. The following example will get you going on running the simulations and calculating the entropy production rate estimated from the generated trajectories

Example:
```python
import numpy as np
import matplotlib.pyplot as plt
import frequent.frequent as fe
import os

os.chdir('path/to/this/repo')
os.chdir('tests/spinOsc')
from spinOscSimulation import spinOscLangevin

# Simulation environment parameters chosen to keep particle within 1 um of origin
gamma = 2e-8  # drag on 1um particle in water in kg/s
dt = 1e-3  # time step of simulation in seconds
nsteps = 1e4  # number of simulation steps
kT = 4e-9  # thermal energy in kg um^2 / s^2
r0 = np.random.rand(2) - 0.5  # starting xy position in um

# create object
colloid = spinOscLangevin(dt=dt, nsteps=nsteps, kT=kT, gamma=gamma, r0=r0)

# Run a simulation
k = 2 * r.gamma  # strength of harmonic potential. Scale to size of drag
alpha = 2 * r.gamma  # strength of rotating force
colloid.runSimulation(k=k, alpha=alpha)

# colloid.pos is a 2x(nsteps) array with the (x,y) positions of the simulation particle
fig, ax = plt.subplots()
ax.plot(colloid.pos[0], colloid.pos[1])

# calculate the frequency space correlation function
colloid_corrMat = fe.corr_matrix(colloid.pos,
                                 sample_spacing=colloid.dt,
                                 mode='full',
                                 method='auto',
                                 return_fft=True)

# calculate the entropy production rate
dsdt = fe.entropy(colloid_corrMat, sample_spacing=colloid.dt)
```
