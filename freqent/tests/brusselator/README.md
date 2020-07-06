# Brusselator reactions

These are stochastic simulations of a reversible Brusselator model. The reactions are

```math
    A \underset{k^-_1}{\overset{k^+_1}{\rightleftharpoons}} X; \quad
    B + X \underset{k^-_2}{\overset{k^+_2}{\rightleftharpoons}} Y + C; \quad
    2X + Y \underset{k^-_3}{\overset{k^+_3}{\rightleftharpoons}} 3X;
```
In these simulations, we consider $`A, B, C`$ as external chemostats with constant concentrations and monitor how $`X, Y`$ change numbers.

The simulations are run using a Gillespie algorithm, contained in the class in `brusselator_gillespie.py`. An example of how to load, run, and plot the simulation are given below

```python
import numpy as np
import matplotlib.pyplot as plt
from brusselator_gillespie import brusselatorStochSim

# reaction rates in format [k1plus, k1minus, k2plus, k2minus, k3plus, k3minus]
rates = [1, 0.5, 2, 1, 2, 1]
# simulation time steps are not evenly distributed.
# Give array of evenly-spaced time points to output the simulation status
t_points = np.arange(0, 100, 0.1)
# set size of reaction volume. The lower the value, the stronger the noise
V = 100
# set initial values of (X, Y)
pop_init = np.random.rand(2) * V

bruss = brusselatorStochSim(population_init=pop_init, V=V, t_points=t_points, rates=rates)
bruss.runSimulation()

fig, ax = plt.subplots()
ax.plot(bruss.pop[:, 0], bruss.pop[:, 1])
ax.set(xlabel='X', ylabel='Y')

```
