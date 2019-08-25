# Driven Brownian particle

These are simulations of an $N \geq 2$ dimensional driven Brownian particle. The simulation solves the following system of Langevin equations:

$$
\begin{aligned}
    \dot{\mathbf{x}} &= F\mathbf{x} + \sqrt{2D} \boldsymbol{\xi}, \\
    F &=
    \begin{pmatrix}
        -k & -\alpha & 0 & \ldots &  \\
        \alpha & -k & 0 &  &  \\
        0 & 0 & -k &  &  \\
        \vdots &  &  &  \ddots & \\
        &  &  &  & -k
    \end{pmatrix},
\end{aligned}
$$

The simulations are non-dimensionalized with time scale $\tau = 1/k$ and length scale $\lambda = \sqrt{D/k}$, so the harmonic potential is always set to strength $k=1$ and the only free parameter to set is the stength of the driving, $\alpha$. See below for an example of how to run a simulation, plot the trajectory, calculate the entropy production rate and epr density of the resulting trajectory and plot it.

```python
from spinOscSimulation import *
import numpy as np
import matplotlib.pyplot as plt
import freqent.freqent as fe

ndim = 2  # run a 2 dimensional simulation
r0 = np.random.rand(ndim)  # set initial condition of particle
dt = 1e-3  # set size of time step for simulation
nsteps = 1e6  # number of time steps to take. Total time of simulation is dt * nsteps

# create an instance of the simulation class
r = spinOscLangevin(dt=dt, r0=r0, nsteps=nsteps)

# run a simulation with rotational strength alpha=2
a = 2
r.runSimulation(alpha=a)

# trajectory information contained in r.pos numpy array
fig, ax = plt.subplots()
ax.plot(r.pos[0], r.pos[1], 'k', lw=1)
ax.set(xlabel='x', ylabel='y', title=r'Tajectory for $\alpha =${a}'.format(a=a))

# calculate epr and epr density
s, s_density, freqs = fe.entropy(data=r.pos, sample_spacing=dt, return_density=True)

# print epr measured
print(s)

# plot epr density
fig2, ax2 = plt.subplots()
ax2.plot(freqs, s_density, 'k')




```
