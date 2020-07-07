'''
Harmonic trap test

Now we calculate the correlation function for a particle trapped in a harmonic potential only. This should lead to a correlation function given by

< r_\mu(t) r_\nu(t + \tau) > = \delta_{\mu \nu} \dfrac{D\gamma}{k} e^{-k |\tau|/\gamma}$$
'''

import numpy as np
import matplotlib.pyplot as plt
from spinOscSimulation import spinOscLangevin
import freqent.freqent as fe
import matplotlib as mpl
import os

mpl.rcParams['pdf.fonttype'] = 42

savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/frequencySpaceDissipation/tests/spinOsc/'

# Simulation environment parameters
gamma = 2e-8  # drag on 1um particle in water in kg/s
dt = 1e-3  # time step of simulation in seconds
nsteps = 1e4  # number of simulation steps
kT = 4e-9  # thermal energy in kg um^2 / s^2
r0 = np.random.rand(2) - 0.5  # starting xy position in um

# create object
r = spinOscLangevin(dt=dt, nsteps=nsteps, kT=kT, gamma=gamma, r0=r0)

# forcing parameters
k = 2 * r.gamma  # spring
nsim = 50
c_all = np.zeros((int(2 * nsteps + 1), 2, 2, nsim))

for ii in range(nsim):
    txt = 'simulation {n} of {N}'.format(n=ii, N=nsim)
    print(txt, end='\r')
    r.reset()
    r.runSimulation(k=k, alpha=0)
    c_all[..., ii], tau = fe.corr_matrix(r.pos,
                                         sample_spacing=r.dt,
                                         mode='full',
                                         method='auto',
                                         return_fft=False)

fig, ax = plt.subplots(2, 2, sharex=True)

c_thry = np.zeros((int(2 * nsteps + 1), 2, 2))

c_thry[:, 0, 0] = np.exp(-k * np.abs(tau) / r.gamma) * r.D * r.gamma / k
c_thry[:, 1, 1] = c_thry[:, 0, 0]

for ii in range(nsim):
    ax[0, 0].plot(tau, c_all[:, 0, 0, ii], 'k-', alpha=0.2)
    ax[0, 1].plot(tau, c_all[:, 0, 1, ii], 'k-', alpha=0.2)
    ax[1, 0].plot(tau, c_all[:, 1, 0, ii], 'k-', alpha=0.2)
    ax[1, 1].plot(tau, c_all[:, 1, 1, ii], 'k-', alpha=0.2)

ax[0, 0].plot(tau, np.mean(c_all[:, 0, 0, :], axis=-1), 'r-', linewidth=3)
ax[0, 1].plot(tau, np.mean(c_all[:, 0, 1, :], axis=-1), 'r-', linewidth=3)
ax[1, 0].plot(tau, np.mean(c_all[:, 1, 0, :], axis=-1), 'r-', linewidth=3)
ax[1, 1].plot(tau, np.mean(c_all[:, 1, 1, :], axis=-1), 'r-', linewidth=3, label='ensemble mean')

ax[0, 0].plot(tau, c_thry[:, 0, 0], 'c-')
ax[0, 1].plot(tau, c_thry[:, 0, 1], 'c-')
ax[1, 0].plot(tau, c_thry[:, 1, 0], 'c-')
ax[1, 1].plot(tau, c_thry[:, 1, 1], 'c-', label='theory')

ax[0, 0].set(ylabel=r'$C_{xx}$')
ax[0, 1].set(ylabel=r'$C_{xy}$')
ax[1, 0].set(xlabel=r'lag time, $\tau$ [s]', ylabel=r'$C_{yx}$')
ax[1, 1].set(xlabel=r'lag time, $\tau$ [s]', ylabel=r'$C_{yy}$')


plt.legend(loc='lower center')
plt.tight_layout()

fig.savefig(os.path.join(savepath, 'harmonic_correlation_test.pdf'), format='pdf')
plt.show()
