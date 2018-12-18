'''
Now we calculate the correlation function for a Brownian particle with dynamics given by

gamma dr/dt = -kr + alpha(cross(z, r)) + xi

correlation function defined as
< r_mu(t) r_nu(t + tau) >
'''

import numpy as np
import matplotlib.pyplot as plt
from spinOscSimulation import spinOscLangevin
import freqent.freqent as fe
import matplotlib as mpl
import os
import argparse
import multiprocessing
from datetime import datetime
import csv
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
# parser.add_argument('--r0', type=float, nargs=3, default=np.random.rand(2) - 0.5,
#                     help='starting xy position in um')
parser.add_argument('--nsim', type=int, default=50,
                    help='number of simulations to run')
parser.add_argument('-k', '--k_multiple', type=float, default=2,
                    help='Spring constant of harmonic potential in units of gamma')
parser.add_argument('-a', '--alpha_multiple', type=float, default=2,
                    help='Rotational force strength in units of gamma')
parser.add_argument('--norm', '-norm', type=str, default='biased',
                    help=('Normalization of correlation function to use. Options are '
                          '"biased", "unbiased", and "none"'))
parser.add_argument('--seed_type', type=str, default='time',
                    help=('a string to decide what seed to use when generating '
                          'trajectories. use "time" to use current microsecond or '
                          '"ints" to use the integers 1,2,...,nsim as seeds.'))
parser.add_argument('--ndim', type=int, default=2,
                    help='Number of dimensions of simulation')

args = parser.parse_args()
r0 = np.random.randn(args.ndim)
# create object
r = spinOscLangevin(dt=args.dt, nsteps=args.nsteps, kT=args.kT, gamma=args.gamma,
                    r0=r0)

# forcing parameters
# equilibriationFrames = int(nsteps/2);
k = args.k_multiple * r.gamma
alpha = args.alpha_multiple * r.gamma


def get_corr_mat(seed):
    '''
    helper function to pass to multiprocessing pool
    '''
    np.random.seed(seed)
    r.reset()
    r.runSimulation(k=k, alpha=alpha)
    c, t = fe.corr_matrix(r.pos,
                          sample_spacing=r.dt,
                          mode='full',
                          method='auto',
                          norm=args.norm,
                          return_fft=False)
    return c


def get_corr_mat_fft(seed):
    '''
    helper function to pass to multiprocessing pool
    '''
    np.random.seed(seed)
    r.reset()
    r.runSimulation(k=k, alpha=alpha)
    c_fft, omega = fe.corr_matrix(r.pos,
                                  sample_spacing=r.dt,
                                  mode='full',
                                  method='auto',
                                  norm=args.norm,
                                  return_fft=True)
    return c_fft


if str(args.seed_type) == 'time':
    seeds = np.arange(args.nsim) + datetime.now().microsecond
elif str(args.seed_type) == 'ints':
    seeds = np.arange(args.nsim)
else:
    ValueError('Expected seed_type = {''time'', ''ints''}, received {0}.'.format(int(args.seed)))

with multiprocessing.Pool(processes=5) as pool:
    result_real = pool.map(get_corr_mat, seeds)
    result_freq = pool.map(get_corr_mat_fft, seeds)

c_all = np.asarray(result_real)
c_all_fft = np.asarray(result_freq)

# get lag time vector
maxTau = args.dt * args.nsteps
tau = np.linspace(-maxTau, maxTau, 2 * args.nsteps + 1)
c_thry = np.zeros((int(2 * (args.nsteps) + 1), 2, 2))

# get frequency vector
omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(tau), d=args.dt))
c_thry_fft = np.zeros(c_thry.shape, dtype=complex)

# c0 = (r.D / (k / r.gamma)) * np.exp(-(k / r.gamma) * np.abs(tau))
# c1 = (r.D / (k / r.gamma)**2) * (k / r.gamma) * tau * np.exp(-(k / r.gamma) * np.abs(tau))
# c2 = ((r.D / (2 * (k / r.gamma)**3)) *
#       ((k / r.gamma) * np.abs(tau) + 1) * np.exp(-(k / r.gamma) * np.abs(tau)))

c_thry[:, 0, 0] = (r.D / (k / r.gamma)) * np.exp(-(k / r.gamma) * np.abs(tau)) * np.cos((alpha / r.gamma) * tau)
c_thry[:, 1, 1] = (r.D / (k / r.gamma)) * np.exp(-(k / r.gamma) * np.abs(tau)) * np.cos((alpha / r.gamma) * tau)
c_thry[:, 0, 1] = (r.D / (k / r.gamma)) * np.exp(-(k / r.gamma) * np.abs(tau)) * -np.sin((alpha / r.gamma) * tau)
c_thry[:, 1, 0] = (r.D / (k / r.gamma)) * np.exp(-(k / r.gamma) * np.abs(tau)) * np.sin((alpha / r.gamma) * tau)

c_thry_fft_prefactor = 2 * r.D / (((args.k_multiple + 1j * omega)**2 + args.alpha_multiple**2) *
                                  ((args.k_multiple - 1j * omega)**2 + args.alpha_multiple**2))
c_thry_fft[:, 0, 0] = c_thry_fft_prefactor * (args.alpha_multiple**2 + args.k_multiple**2 + omega**2)
c_thry_fft[:, 1, 1] = c_thry_fft_prefactor * (args.alpha_multiple**2 + args.k_multiple**2 + omega**2)
c_thry_fft[:, 0, 1] = c_thry_fft_prefactor * 2j * args.alpha_multiple * omega
c_thry_fft[:, 1, 0] = -c_thry_fft_prefactor * 2j * args.alpha_multiple * omega


# Start plotting
fig, ax = plt.subplots(2, 2, sharex=True)
for ii in range(args.nsim):
    ax[0, 0].plot(tau, c_all[ii, :, 0, 0], 'k-', alpha=0.2)
    ax[0, 1].plot(tau, c_all[ii, :, 0, 1], 'k-', alpha=0.2)
    ax[1, 0].plot(tau, c_all[ii, :, 1, 0], 'k-', alpha=0.2)
    ax[1, 1].plot(tau, c_all[ii, :, 1, 1], 'k-', alpha=0.2)

ax[0, 0].plot(tau, np.mean(c_all[:, :, 0, 0], axis=0), 'r-', linewidth=3)
ax[0, 1].plot(tau, np.mean(c_all[:, :, 0, 1], axis=0), 'r-', linewidth=3)
ax[1, 0].plot(tau, np.mean(c_all[:, :, 1, 0], axis=0), 'r-', linewidth=3)
ax[1, 1].plot(tau, np.mean(c_all[:, :, 1, 1], axis=0), 'r-', linewidth=3,
              label='ensemble mean')

ax[0, 0].plot(tau, c_thry[:, 0, 0], 'c-')
ax[0, 1].plot(tau, c_thry[:, 0, 1], 'c-')
ax[1, 0].plot(tau, c_thry[:, 1, 0], 'c-')
ax[1, 1].plot(tau, c_thry[:, 1, 1], 'c-',
              label='theory')

ax[0, 0].set(ylabel=r'$C_{xx}$')
ax[0, 1].set(ylabel=r'$C_{xy}$')
ax[1, 0].set(xlabel=r'lag time, $\tau$ [s]', ylabel=r'$C_{yx}$')
ax[1, 1].set(xlabel=r'lag time, $\tau$ [s]', ylabel=r'$C_{yy}$')

plt.tight_layout()
plt.legend()

fig_fft, ax_fft = plt.subplots(2, 2, sharex=True)
for ii in range(args.nsim):
    ax_fft[0, 0].plot(omega, c_all_fft[ii, :, 0, 0].real, 'k-', alpha=0.2)
    ax_fft[0, 1].plot(omega, c_all_fft[ii, :, 0, 1].imag, 'k-', alpha=0.2)
    ax_fft[1, 0].plot(omega, c_all_fft[ii, :, 1, 0].imag, 'k-', alpha=0.2)
    ax_fft[1, 1].plot(omega, c_all_fft[ii, :, 1, 1].real, 'k-', alpha=0.2)

ax_fft[0, 0].plot(omega, np.mean(c_all_fft[:, :, 0, 0], axis=0).real, 'r-', linewidth=3)
ax_fft[0, 1].plot(omega, np.mean(c_all_fft[:, :, 0, 1], axis=0).imag, 'r-', linewidth=3)
ax_fft[1, 0].plot(omega, np.mean(c_all_fft[:, :, 1, 0], axis=0).imag, 'r-', linewidth=3)
ax_fft[1, 1].plot(omega, np.mean(c_all_fft[:, :, 1, 1], axis=0).real, 'r-', linewidth=3,
                  label='ensemble mean')

ax_fft[0, 0].plot(omega, c_thry_fft[:, 0, 0].real, 'c-')
ax_fft[0, 1].plot(omega, c_thry_fft[:, 0, 1].imag, 'c-')
ax_fft[1, 0].plot(omega, c_thry_fft[:, 1, 0].imag, 'c-')
ax_fft[1, 1].plot(omega, c_thry_fft[:, 1, 1].real, 'c-',
                  label='theory')

ax_fft[0, 0].set(xlim=[-15, 15], ylabel=r'$\Re \left[ \mathcal{F}\lbrace C_{xx} \rbrace \right]$')
ax_fft[0, 1].set(xlim=[-15, 15], ylabel=r'$\Im \left[ \mathcal{F}\lbrace C_{xy} \rbrace \right]$')
ax_fft[1, 0].set(xlim=[-15, 15], xlabel=r'frequency, $\omega$ [rad/s]', ylabel=r'$\Im \left[ \mathcal{F}\lbrace C_{yx} \rbrace \right]$')
ax_fft[1, 1].set(xlim=[-15, 15], xlabel=r'frequency, $\omega$ [rad/s]', ylabel=r'$\Re \left[ \mathcal{F}\lbrace C_{yy} \rbrace \right]$')

plt.tight_layout()
plt.legend()

if args.save:
    argDict = vars(args)
    argDict['datetime'] = datetime.now()
    argDict['seeds'] = seeds

    with open(os.path.join(args.savepath, args.filename + '_params.csv'), 'w') as csv_file:
        w = csv.DictWriter(csv_file, argDict.keys())
        w.writeheader()
        w.writerow(argDict)

    fig.savefig(os.path.join(args.savepath, args.filename + '.pdf'), format='pdf')
    fig_fft.savefig(os.path.join(args.savepath, args.filename + '_fft.pdf'), format='pdf')

plt.show()
