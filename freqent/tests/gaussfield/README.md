# Driven Gaussian Fields

These are simulations of 2 coupled, 1 dimensional fields, $`(\phi(x, t), \psi(x, t))`$ evolving using stochastic Model A dynamics with non-equilibrium coupling parametrized by $`\alpha`$. The equations of motion are

```math
\begin{align}
    \partial_t \phi(\mathbf{x}, t) &= -D (r - \nabla^2) \phi - \alpha \psi + \sqrt{2D} \xi_\phi \\
    \partial_t \psi(\mathbf{x}, t) &= -D (r - \nabla^2) \psi + \alpha \phi + \sqrt{2D} \xi_\psi.
\end{align}
```
with $`\langle \xi_i(\mathbf{x}, t) \xi_j(\mathbf{x}', t') \rangle = \delta_{ij} \delta(t-t') \delta^d(\mathbf{x}-\mathbf{x}')`$, $`D`$ is a relaxation constant, and $`r`$ is a spring-like constant that penalizes large amplitudes of the fields.

The simulation is run using the class in `gaussianFieldSimulation.py`. The simulations are non-dimensionalized in both space and time by setting $`D = r = 1`$. Below is a snippet of code that illustrates how to load the class, run a simulation, and plot the output

```python
import numpy as np
import matplotlib.pyplot as plt
from gaussianFieldSimulation import gaussianFields1D

# time step of simulation
dt = 0.01
# number of simulation time steps, here chosen for a final time of 100
nsteps = 1e4
# lattice spacing
dx = 0.1
# number of lattice sites
nsites = 128
# random initial conditions
ic = np.random.randn(2, nsites)
# strength of coupling
alpha = 2

# load the class
f = gaussianFields1D(dt=dt, dx=dx, ic=ic, nsteps=nsteps)
# run the simulation
f.runSimulation(alpha=alpha)
# plot the resulting simulation, only plot every 10th time step between 0.2 and 0.8 of the total time
f.plotTrajectory(savepath='your/favorite/path', tmin_frac=0.2, tmax_frac=0.8, delta=10)
```
