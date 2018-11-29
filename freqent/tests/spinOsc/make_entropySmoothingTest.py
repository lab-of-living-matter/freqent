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
from astropy.convolution import Gaussian1DKernel, convolve
mpl.rcParams['pdf.fonttype'] = 42
# plt.close('all')

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
parser.add_argument('--r0', type=float, nargs=2, default=np.random.rand(2) - 0.5,
                    help='starting xy position in um')
parser.add_argument('--nsim', type=int, default=50,
                    help='number of simulations to run')
parser.add_argument('-k', '--k_multiple', type=float, default=2,
                    help='Spring constant of harmonic potential in units of gamma')
parser.add_argument('-a', '--alpha_multiple', type=float, default=2,
                    help='Rotational force strength in units of gamma')
parser.add_argument('--seed_type', type=str, default='time',
                    help=('a string to decide what seed to use when generating '
                          'trajectories. use "time" to use current microsecond or '
                          '"input" to directly input the seed'))
parser.add_argument('--norm', '-norm', type=str, default='unbiased',
                    help=('Normalization of correlation function to use. Options are '
                          '"biased", "unbiased", and "none"'))
parser.add_argument('--seed_input', '-seed', type=float, default=None,
                    help='if seed_type=input, what the seed explicitly is')
parser.add_argument('--scale_array', '-scale', type=float, nargs=3, default=[1, 10, 10])

args = parser.parse_args()
# create object
r = spinOscLangevin(dt=args.dt, nsteps=args.nsteps, kT=args.kT, gamma=args.gamma,
                    r0=args.r0)

# forcing parameters
# equilibriationFrames = int(nsteps/2);
k = args.k_multiple * r.gamma
alpha = args.alpha_multiple * r.gamma

# get the different scales of the Gaussian used to smooth the data
scales = np.linspace(args.scale_array[0], args.scale_array[1], int(args.scale_array[2]))
colors = plt.cm.winter(np.linspace(0, 1, len(scales)))


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
elif str(args.seed_type) == 'input':
    seeds = np.arange(args.seed_input, args.seed_input + args.nsim)
else:
    ValueError('Expected seed_type = {''time'', ''ints''}, received {0}.'.format(int(args.seed)))

if args.nsim < 5:
    nProcesses = args.nsim
elif args.nsim >= 5:
    nProcesses = 5

with multiprocessing.Pool(processes=nProcesses) as pool:
    # result_real = pool.map(get_corr_mat, seeds)
    result_freq = pool.map(get_corr_mat_fft, seeds)

# c_all = np.mean(np.asarray(result_real), axis=0)
c_all_fft = np.mean(np.asarray(result_freq), axis=0)

c_all_fft_smoothed = np.zeros(c_all_fft.shape, dtype=complex)
sArray = np.zeros(len(scales) + 1, dtype=complex)
sArray[0] = fe.entropy(c_all_fft, sample_spacing=args.dt)

# maxTau = args.dt * args.nsteps
# tau = np.linspace(-maxTau, maxTau, 2 * args.nsteps + 1)
omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(2 * args.nsteps + 1, d=args.dt))
c_thry_fft = np.zeros(c_all_fft.shape, dtype=complex)

c_thry_fft_prefactor = 2 * r.D / (((args.k_multiple + 1j * omega)**2 + args.alpha_multiple**2) *
                                  ((args.k_multiple - 1j * omega)**2 + args.alpha_multiple**2))
c_thry_fft[:, 0, 0] = c_thry_fft_prefactor * (args.alpha_multiple**2 + args.k_multiple**2 + omega**2)
c_thry_fft[:, 1, 1] = c_thry_fft_prefactor * (args.alpha_multiple**2 + args.k_multiple**2 + omega**2)
c_thry_fft[:, 0, 1] = c_thry_fft_prefactor * 2j * args.alpha_multiple * omega
c_thry_fft[:, 1, 0] = -c_thry_fft_prefactor * 2j * args.alpha_multiple * omega

fig, ax = plt.subplots(2, 2, sharex=True)
ax[0, 0].plot(omega, c_all_fft[:, 0, 0].real, 'k', alpha=0.7)
ax[0, 1].plot(omega, c_all_fft[:, 0, 1].imag, 'k', alpha=0.7)
ax[1, 0].plot(omega, c_all_fft[:, 1, 0].imag, 'k', alpha=0.7)
ax[1, 1].plot(omega, c_all_fft[:, 1, 1].real, 'k', alpha=0.7)

for ind, scale in enumerate(scales):
    gauss = Gaussian1DKernel(scale)
    c_all_fft_smoothed[:, 0, 0].real = convolve(c_all_fft[:, 0, 0].real, gauss)
    c_all_fft_smoothed[:, 0, 1].imag = convolve(c_all_fft[:, 0, 1].imag, gauss)
    c_all_fft_smoothed[:, 1, 0].imag = convolve(c_all_fft[:, 1, 0].imag, gauss)
    c_all_fft_smoothed[:, 1, 1].real = convolve(c_all_fft[:, 1, 1].real, gauss)
    ax[0, 0].plot(omega, c_all_fft_smoothed[:, 0, 0].real, color=colors[ind, :])
    ax[0, 1].plot(omega, c_all_fft_smoothed[:, 0, 1].imag, color=colors[ind, :])
    ax[1, 0].plot(omega, c_all_fft_smoothed[:, 1, 0].imag, color=colors[ind, :])
    ax[1, 1].plot(omega, c_all_fft_smoothed[:, 1, 1].real, color=colors[ind, :])
    sArray[ind + 1] = fe.entropy(c_all_fft_smoothed, sample_spacing=args.dt)

ax[0, 0].plot(omega, c_thry_fft[:, 0, 0].real, 'r')
ax[0, 1].plot(omega, c_thry_fft[:, 0, 1].imag, 'r')
ax[1, 0].plot(omega, c_thry_fft[:, 1, 0].imag, 'r')
ax[1, 1].plot(omega, c_thry_fft[:, 1, 1].real, 'r')

ax[0, 0].set(xlim=[-15, 15], ylabel=r'$\Re \left[ \mathcal{F}\lbrace C_{xx} \rbrace \right]$')
ax[0, 1].set(xlim=[-15, 15], ylabel=r'$\Im \left[ \mathcal{F}\lbrace C_{xy} \rbrace \right]$')
ax[1, 0].set(xlim=[-15, 15], xlabel=r'frequency, $\omega$ [rad/s]', ylabel=r'$\Im \left[ \mathcal{F}\lbrace C_{yx} \rbrace \right]$')
ax[1, 1].set(xlim=[-15, 15], xlabel=r'frequency, $\omega$ [rad/s]', ylabel=r'$\Re \left[ \mathcal{F}\lbrace C_{yy} \rbrace \right]$')

plt.tight_layout()
# plt.legend()

# plot difference between entropy measured and true entropy
dw = 2 * np.pi / (args.dt * len(omega))  # the step size between frequencies
sThry = 2 * args.alpha_multiple**2 / args.k_multiple

fig_ent, ax_ent = plt.subplots(1, 2, figsize=[8, 4])
ax_ent[0].plot(np.concatenate((np.zeros(1), scales)) * dw, sArray.real, 'ko')
ax_ent[0].plot([0, scales.max() * dw], [sThry, sThry],
               '--k', label='$S^{{thry}} = {0}$'.format(sThry))
ax_ent[0].set(xlabel=r'$\sigma_{kernel}$',
              ylabel=r'$d{S}/dt$')

ax_ent[1].loglog(np.concatenate((np.zeros(1), scales)) * dw, abs(sThry - sArray.real), 'ko')
ax_ent[1].set(xlabel=r'$\sigma_{kernel}$',
              ylabel=r'$\vert \dot{S}_{thry} - \dot{S} \vert$')
plt.tight_layout()

if args.save:
    argDict = vars(args)
    argDict['datetime'] = datetime.now()
    argDict['seeds'] = seeds

    with open(os.path.join(args.savepath, args.filename + '_params.csv'), 'w') as csv_file:
        w = csv.DictWriter(csv_file, argDict.keys())
        w.writeheader()
        w.writerow(argDict)

    fig.savefig(os.path.join(args.savepath, args.filename + '_corrFuncs.pdf'), format='pdf')
    fig_ent.savefig(os.path.join(args.savepath, args.filename + '_entropy.pdf'), format='pdf')

plt.show()
