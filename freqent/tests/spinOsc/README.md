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

The simulations are non-dimensionalized with time scale $`\tau = 1/k`$ and length scale $`\lambda = \sqrt{D/k}`$, so the harmonic potential is always set to strength $`k=1`$ and the only free parameter to set is the stength of the driving, $`\alpha`$. See below for an example of how to run a simulation, plot the trajectory, estimate the entropy production rate $`\hat{\dot{S}}`$ and entropy production factor $`\hat{\mathcal{E}}`$, and plot $`\hat{\mathcal{E}}`$ alongside its analytical form. The analytic results are


```math
\mathcal{E}^\mathrm{DBP} = \frac{8 \alpha^2 \omega^2}{\left( \omega^2 - \omega_0^2 \right)^2 + (2 \omega)^2}, \qquad \dot{S}^\mathrm{DBP} = 2 \alpha^2
```

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

# plot subsection of trajectory
fig, [ax_traj, ax_epf] = plt.subplots(1, 2, figsize=(9, 4))
ax_traj.plot(r.pos[0, 1000:3001], r.pos[1, 1000:3001])
ax_traj.plot(r.pos[0, 1000], r.pos[1, 1000], 'X',
             markersize=10, markeredgecolor='k', color='r',
             label='start position')
ax_traj.plot(r.pos[0, 3000], r.pos[1, 3000], 'o',
             markersize=15, markeredgecolor='k', color=(0.9, 0.9, 0.9),
             label='current position')
ax_traj.set(xlabel='x', ylabel='y')
ax_traj.legend(loc='lower right')

# calculate epr and epf, smoothing the correlation functions with a Gaussian
# with a standard deviation of sigma = 30 * dk, where dk = 2 * pi / T is the
# spacing between frequency bins, i.e. dk = np.diff(freqs)[0]
epr, epf, w = fe.entropy(data=r.pos, sample_spacing=dt,
                         return_epf=True, sigma=30)

# print epr measured
print('Theoretical EPR: {s:0.2f}'.format(s=2 * alpha**2))
print('Measured EPR: {s:0.2f}'.format(s=epr))

# Calculate theoretical entropy production factor for the driven brownian particle
w0 = np.sqrt(1 + alpha**2)  # peak frequency of epf
epf_dbp = (8 * alpha**2 * w**2) / ((w**2 - w0**2)**2 + 4 * w**2)
dw = np.diff(w)[0]

# plot epf and mark the rotational frequency used to run the simulation

ax_epf.plot(w, epf, lw=2, label=r'$\hat{\mathcal{E}}$')
# rescale epf_dbp by the integration measure to compare with estimate
ax_epf.plot(w, epf_dbp * dw / (2 * np.pi), 'k--', lw=2,
              label=r'$\mathcal{E}^\mathrm{DBP}$')
ax_epf.set(xlabel=r'$\omega$', ylabel=r'$\mathcal{E}$',
           xlim=[-20, 20])
ax_epf.legend(loc='upper right')

plt.tight_layout()
plt.show()
```

Here is an example of the output from running the above as a script.

```python
Theoretical EPR: 8.00
Measured EPR: 8.52
```

![image](/freqent/tests/spinOsc/readme_example_alpha2.png)
