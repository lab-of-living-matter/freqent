# Driven Gaussian Fields

These are simulations of 2 coupled, 1 dimensional fields, $`(\phi(x, t), \psi(x, t))`$ evolving using stochastic Model A dynamics with non-equilibrium coupling parametrized by $`\alpha`$. The equations of motion are

```math
    \partial_t \phi(\mathbf{x}, t) = -D (r - \nabla^2) \phi - \alpha \psi + \sqrt{2D} \xi_\phi \\
    \partial_t \psi(\mathbf{x}, t) = -D (r - \nabla^2) \psi + \alpha \phi + \sqrt{2D} \xi_\psi
```
with $`\langle \xi_i(\mathbf{x}, t) \xi_j(\mathbf{x}', t') \rangle = \delta_{ij} \delta(t-t') \delta^d(\mathbf{x}-\mathbf{x}')`$, $`D`$ is a relaxation constant, and $`r`$ is a spring-like constant that penalizes large amplitudes of the fields.

The simulation is run using the class in `gaussianFieldSimulation.py`. The simulations are non-dimensionalized in both space and time by setting $`D = r = 1`$. Below is a snippet of code that illustrates how to load the class, run some simulations, plot an example trajectory, and calculate $`\hat{\dot{s}}`$ and $`\hat{\mathcal{E}}`$. We compare the estimations with their analytical counterparts, which are given by:

```math
\mathcal{E}^{\mathrm{DGF}}=\frac{8 \alpha^{2} \omega^{2}}{\left(\omega^{2}-\omega_{0}^{2}(q)\right)^{2}+\left(2 D\left(r+q^{2}\right) \omega\right)^{2}}, \quad \dot{s}^{\mathrm{DGF}}=\frac{\alpha^{2}}{D \sqrt{r}}
```

Below is an example of how to run a simulation and calculate the entropy production rate. The simulation takes approximately 10 seconds to run on a 2016 Macbook Air.

```python
import numpy as np
import matplotlib.pyplot as plt
from gaussianFieldSimulation import gaussianFields1D
import matplotlib as mpl
import freqent.freqentn as fen

mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


np.random.seed(9842590)  # set seed for reproducibility

dt = 0.001  # time step of simulation
nsteps = 20000  # number of simulation time steps
dx = 0.1  # lattice spacing
nsites = 128  # number of lattice sites
ic = np.random.randn(2, nsites)  # random initial conditions
alpha = 7.5  # strength of coupling
nsim = 5  # number of simulations to run

# load the class
f = gaussianFields1D(dt=dt, dx=dx, ic=ic, nsteps=nsteps)

# run the simulations and concatenate the trajectories
traj = np.zeros((nsim, 2, nsteps + 1, nsites))
for n in range(nsim):
    f.runSimulation(alpha=alpha)
    traj[n] = f.pos

# plot the last simulation
# plot every 10th time step between 0.2 and 0.8 of the total time
f.plotTrajectory(tmin_frac=0.2, tmax_frac=0.8, delta=10)
```

![trajectory](/freqent/tests/gaussfield/readme_example_traj.png)

```python
# Calculate EPR and EPF
# use second half of simulation to ensure steady state
t_epr = f.t > f.t // 2
epr, epf, w = fen.entropy(traj[:, :, t_epr, :], sample_spacing=[dt, dx],
                          sigma=[3, 2], return_epf=True, many_traj=True)

# print epr measured
print('Theoretical EPR: {s:0.2f}'.format(s=alpha**2))
print('Measured EPR: {s:0.2f}'.format(s=epr))
```

```python
Theoretical EPR: 56.25
Measured EPR: 67.81
```

```python
# Plot epf and its analytical form side-by-side
kk, ww = np.meshgrid(w[1], w[0])
epf_thry = (8 * alpha**2 * ww**2 / (((1 + kk**2 + ww * 1j)**2 + alpha**2) * ((1 + kk**2 - ww * 1j)**2 + alpha**2))).real
# get frequency spacing to multiply epf_thry by integration measure
dw = [np.diff(f)[0] for f in w]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
a0 = ax[0].pcolormesh(kk, ww, epf_thry * np.prod(dw) / (4 * np.pi**2),
                      rasterized=True)
fig.colorbar(a0, ax=ax[0])
ax[0].set(ylim=[-10 * np.pi, 10 * np.pi], ylabel=r'$\omega$',
          xlabel=r'$q$', title=r'$\mathcal{E}^\mathrm{DGF}$')
ax[0].set_aspect('equal')
a1 = ax[1].pcolormesh(kk, ww, epf, rasterized=True)
fig.colorbar(a1, ax=ax[1])
ax[1].set(ylim=[-10 * np.pi, 10 * np.pi], ylabel=r'$\omega$',
          xlabel=r'$q$', title=r'$\hat{\mathcal{E}}$')
ax[1].set_aspect('equal')

plt.tight_layout()
plt.show()

```
![epf](/freqent/tests/gaussfield/readme_example_epf.png)
