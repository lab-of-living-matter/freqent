# Reaction-diffusion Brusselator

These are stochastic simulations of a 1-dimensional, periodic reaction-diffusion system with reversible Brusselator reactions. The reactions are

```math
    A_i \underset{k^-_1}{\overset{k^+_1}{\rightleftharpoons}} X_i; \quad
    B_i + X_i \underset{k^-_2}{\overset{k^+_2}{\rightleftharpoons}} Y_i + C_i; \quad
    2X_i + Y_i \underset{k^-_3}{\overset{k^+_3}{\rightleftharpoons}} 3X_i;
```
augmented by stochastic hopping between lattice sites $`i \rightleftarrows i \pm 1`$ with rates $`d_i = D_\mu / h^2`$, where $`D_\mu`$ is the diffusion constant for chemical species $`\mu = \lbrace X, Y \rbrace`$ and $`h`$ is the spacing between lattice sites.

In these simulations, we consider $`A_i, B_i, C_i`$ as external chemostats with constant concentrations and monitor how $`X_i, Y_i`$ change numbers.

The simulations are run using a Gillespie algorithm, contained in the class in `brussfield_gillespie.py`.

Below is an example of how to run a simulation and calculate the entropy production rate. The simulation takes approximately **15 minutes** to run on a 2016 Macbook air.

```python
import numpy as np
import matplotlib.pyplot as plt
from brussfield_gillespie import brusselator1DFieldStochSim
import matplotlib as mpl
import freqent.freqentn as fen
from scipy import stats

# Plotting preferences
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


seed = 34901340  # set seed for reproducibility

# reaction rates in format
# [k1plus, k1minus, k2plus, k2minus, k3plus, k3minus]
rates = [1, 0.5, 2, 1, 2, 1]

# Give array of evenly-spaced time points to output from the simulation
dt = 0.01
t_final = 50
t_points = np.arange(0, t_final, dt)

n_compartments = 25  # number of compartments
V = 100  # reaction volume of compartments
DX, DY = [1.0, 0.1]  # diffusion constant of X and Y
h = 1  # spacing between lattice sites

# set non-equilibrium driving strength, which sets chemostat values
mu = 6.2
A = V
B = np.exp(mu / 2) * V / (rates[2] * rates[4])
C = np.exp(-mu / 2) * V / (rates[3] * rates[5])

# set initial values of (X, Y) as some noise around the steady state
# to allow fast equilibration. Also make sure (X, Y) are integers
xss = A * rates[0] / rates[1]
yss = (rates[2] * B * xss + rates[5] * xss**3) / (rates[3] * C + rates[4] * xss**2)
X0, Y0 = np.stack((np.around(xss + np.random.randn(n_compartments) * xss / 10),
                   np.around(yss + np.random.randn(n_compartments) * yss / 10))).astype(int)

# run simulation
brussfield = brusselator1DFieldStochSim(XY_init=[X0, Y0],
                                        ABC=[A, B, C],
                                        rates=rates,
                                        t_points=t_points,
                                        V=V,
                                        D=[DX, DY],
                                        n_subvolumes=n_compartments,
                                        l_subvolumes=h,
                                        seed=seed)
brussfield.runSimulation()

# Calculate true, blind, and estimated entropy production rates
t_ss = t_points > 10
sdot, sdot_intercept, _, _, _ = stats.linregress(t_points[t_ss], brussfield.ep[t_ss])
sdot_blind, sdot_blind_intercept, _, _, _ = stats.linregress(t_points[t_ss], brussfield.ep_blind[t_ss])
# for estimated entropy, must have first dimension index variable, not time
epr, epf, w = fen.entropy(np.moveaxis(brussfield.population[t_ss], 1, 0),
                          sample_spacing=[dt, h],
                          return_epf=True, sigma=1)

# plot trajectory
L = np.arange(n_compartments)

cmap_name = 'cividis'
cmap = mpl.cm.get_cmap(cmap_name)
vmax = abs(brussfield.population[t_ss]).max()
vmin = 0
normalize = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# colors = [cmap(normalize(value)) for value in np.ravel(traj)]


fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].pcolormesh(L, t_points[t_ss],
                 brussfield.population[t_ss, 0],
                 cmap=cmap_name, vmin=vmin, vmax=vmax,
                 rasterized=True)
ax[1].pcolormesh(L, t_points[t_ss],
                 brussfield.population[t_ss, 1],
                 cmap=cmap_name, vmin=vmin, vmax=vmax,
                 rasterized=True)

ax[0].set(xlabel=r'$r$', title=r'$X$', ylabel=r'$t \ [1/k^+_1]$')
ax[0].tick_params(which='both', direction='in')
ax[1].set(xlabel=r'$r $', title=r'$Y$')
ax[1].tick_params(which='both', direction='in')

cax, _ = mpl.colorbar.make_axes(ax)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
cbar.ax.tick_params(which='both', direction='in')
```

![traj](/freqent/tests/brussfield/readme_example_traj.png)

```python
# plot entropy production info
fig1, ax1 = plt.subplots(1, 3, figsize=(10, 4))

# true entropy production and epr
ax1[0].plot(t_points, brussfield.ep, 'k')
ax1[0].plot(t_points[t_ss], t_points[t_ss] * sdot + sdot_intercept,
            'r--', label=r'$\dot{{s}}_\mathrm{{true}} = ${0:0.2f}'.format(sdot / (n_compartments * h)))
ax1[0].set(xlabel='t', ylabel=r'$\Delta S_\mathrm{true}$')
ax1[0].legend(loc='lower right')
ax1[0].set_aspect(np.diff(ax1[0].set_xlim())[0] / np.diff(ax1[0].set_ylim())[0])

# blind entropy production and epr
ax1[1].plot(t_points, brussfield.ep_blind, 'k')
ax1[1].plot(t_points[t_ss], t_points[t_ss] * sdot_blind + sdot_blind_intercept,
            'r--', label=r'$\dot{{s}}_\mathrm{{blind}} = ${0:0.2f}'.format(sdot_blind / (n_compartments * h)))
ax1[1].set(xlabel='t', ylabel=r'$\Delta S_\mathrm{blind}$')
ax1[1].legend(loc='lower right')
ax1[1].set_aspect(np.diff(ax1[1].set_xlim())[0] / np.diff(ax1[1].set_ylim())[0])

# epf and estimated epr
ax1[2].pcolormesh(w[1], w[0], epf)
ax1[2].set(xlabel=r'$q$', ylabel=r'$\omega$', ylim=[-np.pi, np.pi])
ax1[2].text(2, 2, r'$\hat{\mathcal{E}}$', color='w', fontsize=15)
ax1[2].text(0.5, -2.5, r'$\hat{{\dot{{s}}}} = ${0:0.2f}'.format(epr), color='w')
ax1[2].set_aspect('equal')

plt.tight_layout()
plt.show()

```

![epr](/freqent/tests/brussfield/readme_example_epr+epf.png)
