# Driven Brownian particle

These are simulations of an $`N \geq 2`$ dimensional driven Brownian particle. The simulation solves the following system of Langevin equations:

```math
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
```

The simulations are non-dimensionalized with time scale $`\tau = 1/k`$ and length scale $`\lambda = \sqrt{D/k}`$, so the harmonic potential is always set to strength $`k=1`$ and the only free parameter to set is the stength of the driving, $`\alpha`$. See below for an example of how to run a simulation, plot the trajectory, calculate the entropy production rate $`\dot{S}`$ and entropy production factor $`\mathcal{E}`$, and plot $`\mathcal{E}`$.

```python
from spinOscSimulation import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import freqent.freqent as fe

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

ndim = 2  # run a 2 dimensional simulation
r0 = np.random.rand(ndim)  # set initial condition of particle
dt = 1e-3  # set size of time step for simulation
nsteps = 1e6  # number of time steps to take. Total time of simulation is dt * nsteps

# create an instance of the simulation class
r = spinOscLangevin(dt=dt, r0=r0, nsteps=nsteps)

# run a simulation with rotational strength alpha=2
alpha = 2
r.runSimulation(alpha=alpha)

# trajectory information contained in r.pos numpy array
fig, [ax_traj, ax_epf] = plt.subplots(1, 2)
ax_traj.plot(r.pos[0, 1000:3001], r.pos[1, 1000:3001])
ax_traj.plot(r.pos[0, 1000], r.pos[1, 1000], 'X',
             markersize=10, markeredgecolor='k', color='r',
             label='start position')
ax_traj.plot(r.pos[0, 3000], r.pos[1, 3000], 'o',
             markersize=15, markeredgecolor='k', color=(0.9, 0.9, 0.9),
             label='current position')
ax_traj.set(xlabel='x', ylabel='y',
            title=r'Tajectory for $\alpha =${a}'.format(a=a))
ax_traj.set_aspect('equal')
ax_traj.legend()

# calculate epr and epr density, smoothing the correlation functions with a Gaussian
# with a standard deviation of sigma = 30 * dk, where dk = 2 * pi / T is the
# spacing between frequency bins, i.e. dk = np.diff(freqs)[0]
s, epf, w = fe.entropy(data=r.pos, sample_spacing=dt,
                       return_epf=True, sigma=30)

# print epr measured
print('Theoretical EPR: {s:0.2f}'.format(s=2 * a**2))
print('Measured EPR: {s:0.2f}'.format(s=s))

# Calculate theoretical entropy production factor
w0 = np.sqrt(1 + a**2)
epf_thry = (8 * alpha**2 * w**2) / ((w**2 - w0**2)**2 + 4 * w**2)
dw = np.diff(w)[0]

# plot epf and mark the rotational frequency used to run the simulation

ax_epf.plot(w, epf, lw=2, label=r'$\hat{\mathcal{E}}$')
ax_epf.plot(w, epf_thry * dw / (2 * np.pi), 'k--', lw=2,
         label=r'$\mathcal{E}^\mathrm{DBP} \times d\omega / 2 \pi$')
ax_epf.set(xlabel=r'$\omega$', ylabel=r'$\mathcal{E}$', xlim=[-30, 30])
ax_epf.legend()

```
