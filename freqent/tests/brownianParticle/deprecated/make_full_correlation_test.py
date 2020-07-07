'''
Harmonic trap test

Now we calculate the correlation function for a particle subject to the for

< r_mu(t) r_nu(t + tau) > = delta_{mu nu} dfrac{Dgamma}{k} e^{-k |tau|/gamma}$$
'''

import numpy as np
import matplotlib.pyplot as plt
from spinOscSimulation import spinOscLangevin
import freqent.freqent as fe
import matplotlib as mpl
import os
import argparse

mpl.rcParams['pdf.fonttype'] = 42

parser = argparse.ArgumentParser(description=('Perform simulations of Brownian particles'
                                              ' in a harmonic potential plus a rotating'
                                              ' force.'))
parser.add_argument('--save', type=bool, default=False,
                    help='Boolean of whether to save outputs')
parser.add_argument('--savepath', type=str, default='',
                    help='Path to save outputs if save')
parser.add_argument('--filename', type=str, default='',
                    help='Name of image file to save at savepath')
parser.add_argument('--gamma', type=float, default=2e-8,
                    help='drag on 1um particle in water in kg/s')
parser.add_argument('--dt', type=float, default=1e-3,
                    help='time step of simulation in seconds')
parser.add_argument('--nsteps', type=int, default=int(1e4),
                    help='number of simulation steps')
parser.add_argument('--kT', type=float, default=4e-9,
                    help='thermal energy in kg um^2 / s^2')
parser.add_argument('--r0', type=float, default=np.random.rand(2) - 0.5,
                    help='starting xy position in um')
parser.add_argument('--nsim', type=int, default=50,
                    help='number of simulations to run')
parser.add_argument('--k_multiple', type=float, default=2,
                    help='Spring constant of harmonic potential in units of gamma')
parser.add_argument('--alpha_multiple', type=float, default=2,
                    help='Rotational force strength in units of gamma')

args = parser.parse_args()
# create object
r = spinOscLangevin(dt=args.dt, nsteps=args.nsteps, kT=args.kT, gamma=args.gamma,
                    r0=args.r0)

# forcing parameters
# equilibriationFrames = int(nsteps/2);
k = args.k_multiple * r.gamma
alpha = args.alpha_multiple * r.gamma
c_all = np.zeros((int(2 * (args.nsteps) + 1), 2, 2, args.nsim))

for ii in range(args.nsim):
    txt = 'simulation ' + str(ii)
    print(txt, end='\r')
    r.reset()
    r.runSimulation(k=k, alpha=alpha)
    c_all[..., ii], tau = fe.corr_matrix(r.pos,
                                         sample_spacing=r.dt,
                                         mode='full',
                                         method='auto',
                                         return_fft=False)

fig, ax = plt.subplots(2, 2, sharex=True)

c_thry = np.zeros((int(2 * (args.nsteps) + 1), 2,2))

c_thry[:, 0, 0] = (r.D * r.gamma / k) * np.exp(-k * np.abs(tau) / r.gamma) +
                  (alpha**2 * r.D * r.gamma / (2 * k**3)) * ((k * np.abs(tau) / r.gamma) + 1) *
                  np.exp(-k * np.abs(tau) / r.gamma)
c_thry[:, 1, 1] = c_thry[:, 0, 0]
c_thry[:, 0, 1] = -alpha * r.D * r.gamma * (k * tau * np.exp(-k * np.abs(tau) / r.gamma) / r.gamma) / k**2
c_thry[:, 1, 0] = -c_thry[:, 0, 1]

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

ax[0, 0].set(ylabel=r'$C_{xx}$')#, ylim=[-0.25, 0.25], xlim = [-5, 5])
ax[0, 1].set(ylabel=r'$C_{xy}$')#, ylim=[-0.25, 0.25], xlim = [-5, 5])
ax[1, 0].set(xlabel=r'lag time, $\tau$ [s]', ylabel=r'$C_{yx}$')#, ylim=[-0.25, 0.25], xlim = [-5, 5])
ax[1, 1].set(xlabel=r'lag time, $\tau$ [s]', ylabel=r'$C_{yy}$')#, ylim=[-0.25, 0.25], xlim = [-5, 5])

# ax[0, 0].set_aspect(np.diff(ax[0, 0].set_xlim()) / np.diff(ax[0, 0].set_ylim()))
# ax[0, 1].set_aspect(np.diff(ax[0, 1].set_xlim()) / np.diff(ax[0, 1].set_ylim()))
# ax[1, 0].set_aspect(np.diff(ax[1, 0].set_xlim()) / np.diff(ax[1, 0].set_ylim()))
# ax[1, 1].set_aspect(np.diff(ax[1, 1].set_xlim()) / np.diff(ax[1, 1].set_ylim()))

plt.tight_layout()
plt.legend()

fig.savefig(os.path.join(savepath, 'full_correlation_test_alpha10.pdf'), format='pdf')
