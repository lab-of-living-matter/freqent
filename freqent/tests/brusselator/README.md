# Brusselator reactions

These are stochastic simulations of a reversible Brusselator model. The reactions are

```math
    A \underset{k^-_1}{\overset{k^+_1}{\rightleftharpoons}} X; \quad
    B + X \underset{k^-_2}{\overset{k^+_2}{\rightleftharpoons}} Y + C; \quad
    2X + Y \underset{k^-_3}{\overset{k^+_3}{\rightleftharpoons}} 3X;
```
In these simulations, we consider $`A, B, C`$ as external chemostats with constant concentrations and monitor how $`X, Y`$ change numbers.

The simulations are run using a Gillespie algorithm, contained in the class in `brusselator_gillespie.py`.

Below is an example of how to run the simulation and calculate the entropy production rate. The simulation takes approximately 3 minutes to run on a 2016 Macbook Air.

```python
import numpy as np
import matplotlib.pyplot as plt
from brusselator_gillespie import brusselatorStochSim
import matplotlib as mpl
import freqent.freqent as fe
from scipy import stats

# Plotting preferences
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

seed = 86018373  # set random seed for reproducibility

# reaction rates in format
# [k1plus, k1minus, k2plus, k2minus, k3plus, k3minus]
rates = [1, 0.5, 2, 1, 2, 1]

# Give array of evenly-spaced time points to output from the simulation
dt = 0.01
t_final = 100
t_points = np.arange(0, t_final, dt)

# set size of reaction volume. The lower the value, the stronger the noise
V = 100

# set non-equilibrium driving strength, which sets chemostat values
mu = 6.2
A = V
B = np.exp(mu / 2) * V / (rates[2] * rates[4])
C = np.exp(-mu / 2) * V / (rates[3] * rates[5])

# set initial values of (X, Y) as some noise around the steady state
# to allow fast equilibration. Also make sure (X, Y) are integers
xss = A * rates[0] / rates[1]
yss = (rates[2] * B * xss + rates[5] * xss**3) / (rates[3] * C + rates[4] * xss**2)
X0, Y0 = [int(xss + np.random.randn() * xss / 10),
          int(yss + np.random.randn() * yss / 10)]

pop_init = [X0, Y0, A, B, C]

# run simulation
bruss = brusselatorStochSim(population_init=pop_init,
                            V=V,
                            t_points=t_points,
                            rates=rates,
                            seed=seed)
bruss.runSimulation()

# Calculate true, blind, and estimated entropy production rates
t_ss = t_points > 10
sdot, sdot_intercept, _, _, _ = stats.linregress(t_points[t_ss], bruss.ep[t_ss])
sdot_blind, sdot_blind_intercept, _, _, _ = stats.linregress(t_points[t_ss], bruss.ep_blind[t_ss])
# for estimated entropy, must have first dimension index variable, not time
epr, epf, w = fe.entropy(bruss.population[t_ss].T, sample_spacing=dt,
                         return_epf=True, sigma=1)

# find occupancy map of steady state trajectory
binsize = 5  # size of each bin
edges = [np.arange(bruss.population[t_ss, 0].min() - 5.5,
                   bruss.population[t_ss, 0].max() + 5.5,
                   binsize),
         np.arange(bruss.population[t_ss, 1].min() - 5.5,
                   bruss.population[t_ss, 1].max() + 5.5,
                   binsize)]
prob_map, edges = np.histogramdd(bruss.population[t_ss], bins=edges)
prob_map /= prob_map.sum()

# plot trajectory
fig, ax = plt.subplots(figsize=(6, 4))
prob_ax = ax.pcolormesh(edges[0][1:], edges[1][1:],
                        np.log(prob_map.T), cmap='Blues')
ax.plot(bruss.population[t_ss, 0][1000:2001],
        bruss.population[t_ss, 1][1000:2001],
        'k', alpha=0.5, label=r'$\Delta \mu = ${m}'.format(m=mu))
ax.plot(bruss.population[t_ss, 0][2000],
        bruss.population[t_ss, 1][2000],
        'o', markersize=15, markeredgecolor='k', color=(0.9, 0.9, 0.9))
ax.set(xlabel='X', ylabel='Y', ylim=[200, 1300], xlim=[-100, 1000])
ax.set_aspect('equal')
ax.legend(loc='upper right')
cbar = fig.colorbar(prob_ax, ax=ax)
cbar.set_label(r'$\ln(p(X,Y))$')
plt.tight_layout()
```

![traj](/freqent/tests/brusselator/readme_example_traj.png)

```python
# plot entropy production info
fig1, ax1 = plt.subplots(1, 3, figsize=(10, 3))

# true entropy production and epr
ax1[0].plot(t_points, bruss.ep, 'k')
ax1[0].plot(t_points[t_ss], t_points[t_ss] * sdot + sdot_intercept,
            'r--', label=r'$\dot{{S}}_\mathrm{{true}} = ${0:0.2f}'.format(sdot))
ax1[0].set(xlabel='t', ylabel=r'$\Delta S_\mathrm{true}$')
ax1[0].legend(loc='lower right')
ax1[0].set_aspect(np.diff(ax1[0].set_xlim())[0] / np.diff(ax1[0].set_ylim())[0])

# blind entropy production and epr
ax1[1].plot(t_points, bruss.ep_blind, 'k')
ax1[1].plot(t_points[t_ss], t_points[t_ss] * sdot_blind + sdot_blind_intercept,
            'r--', label=r'$\dot{{S}}_\mathrm{{blind}} = ${0:0.2f}'.format(sdot_blind))
ax1[1].set(xlabel='t', ylabel=r'$\Delta S_\mathrm{blind}$')
ax1[1].legend(loc='lower right')
ax1[1].set_aspect(np.diff(ax1[1].set_xlim())[0] / np.diff(ax1[1].set_ylim())[0])

# epf and estimated epr
ax1[2].loglog(w[w > 0], epf[w > 0],
              label=r'$\hat{{\dot{{S}}}} = ${0:0.2f}'.format(epr))
ax1[2].set(xlabel=r'$\omega$', ylabel=r'$\hat{\mathcal{E}}$')
ax1[2].legend(loc='lower left')

plt.tight_layout()
plt.show()

```

![epr](/freqent/tests/brusselator/readme_example_epr+epf.png)
