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
from itertools import product
# import scipy.signal as signal

'''
NOTE: THIS ONLY WORKS WITH FREQENT VERSION 0.0.1, BEFORE freqent.entropy TOOK IN
DATA DIRECTLY, RATHER THAN USING AN ALREADY CALCULATED CORRELATION FUNCTION
'''

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

# simulation parameters
parser.add_argument('--gamma', type=float, default=2e-8,
                    help='drag on 1um particle in water in kg/s')
parser.add_argument('--dt', type=float, default=1e-3,
                    help='time step of simulation in seconds')
parser.add_argument('--nsteps', type=int, default=int(1e4),
                    help='number of simulation steps')
parser.add_argument('--kT', type=float, default=4e-9,
                    help='thermal energy in kg um^2 / s^2')
parser.add_argument('--nsim', type=int, default=10,
                    help='number of simulations to run')
parser.add_argument('-k', '--k_multiple', type=float, default=2,
                    help='Spring constant of harmonic potential in units of gamma')
parser.add_argument('-a', '--alpha_multiple', type=float, default=2,
                    help='Rotational force strength in units of gamma')

# things to calculate with
parser.add_argument('--seed_type', type=str, default='time',
                    help=('a string to decide what seed to use when generating '
                          'trajectories. use "time" to use current microsecond or '
                          '"input" to directly input the seed'))
parser.add_argument('--seed_input', '-seed', type=float, default=None,
                    help='if seed_type=input, what the seed explicitly is')
parser.add_argument('--scale_array', '-scale', type=float, nargs=3, default=[1, 10, 10])
parser.add_argument('--ndim', type=int, default=2,
                    help='Number of dimensions of simulation')
# parser.add_argument('--window', '-window', type=str, default=)

args = parser.parse_args()
r0 = np.random.randn(args.ndim)
idx_pairs = list(product(range(args.ndim), repeat=2))
# create object
r = spinOscLangevin(dt=args.dt, nsteps=args.nsteps, kT=args.kT, gamma=args.gamma,
                    r0=r0)

# forcing parameters
# equilibriationFrames = int(nsteps/2);
k = args.k_multiple * r.gamma
alpha = args.alpha_multiple * r.gamma

# get the different scales of the Gaussian used to smooth the data
scales = np.linspace(args.scale_array[0], args.scale_array[1], int(args.scale_array[2]))
colors_real = plt.cm.winter(np.linspace(0, 1, len(scales)))
colors_imag = plt.cm.summer(np.linspace(0, 1, len(scales)))

def get_corr_mat_fft(seed):
    '''
    function to pass to multiprocessing pool
    '''
    np.random.seed(seed)
    r.reset()
    r.runSimulation(k=k, alpha=alpha)
    c_fft, omega = fe.corr_matrix(r.pos,
                                  sample_spacing=r.dt,
                                  window='boxcar',
                                  nperseg=None,
                                  noverlap=None,
                                  nfft=None,
                                  detrend='constant',
                                  padded=False,
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
    result_freq = pool.map(get_corr_mat_fft, seeds)

c_all_fft = np.mean(np.asarray(result_freq), axis=0)

c_all_fft_smoothed = np.zeros(c_all_fft.shape, dtype=complex)
sArray = np.zeros(len(scales), dtype=complex)
# sArray[0] = fe.entropy(c_all_fft, T=r.t.max())

omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(r.t), d=args.dt))
dw = 2 * np.pi / (args.dt * len(omega))  # the step size between frequencies

fig_real, ax_real = plt.subplots(args.ndim, args.ndim, sharex=True, sharey=True)
fig_imag, ax_imag = plt.subplots(args.ndim, args.ndim, sharex=True, sharey=True)

for ind, scale in enumerate(scales):
    gauss = Gaussian1DKernel(scale)
    for idx in idx_pairs:
        if ind == 0:
            ax_real[idx].plot(omega[omega != 0], c_all_fft[omega != 0, idx[0], idx[1]].real, 'k', alpha=0.5)
            ax_imag[idx].plot(omega[omega != 0], c_all_fft[omega != 0, idx[0], idx[1]].imag, 'k', alpha=0.5)
            ax_real[idx].set(ylabel=r'$\Re(C_{{{0} {1}}})$'.format(idx[0], idx[1]))
            ax_imag[idx].set(xlim=[-20, 20], ylabel=r'$\Im(C_{{{0} {1}}})$'.format(idx[0], idx[1]))

        c_all_fft_smoothed[:, idx[0], idx[1]].real = convolve(c_all_fft[:, idx[0], idx[1]].real, gauss,
                                                              normalize_kernel=True)
        c_all_fft_smoothed[:, idx[0], idx[1]].imag = convolve(c_all_fft[:, idx[0], idx[1]].imag, gauss,
                                                              normalize_kernel=True)

        ax_real[idx].semilogy(omega, c_all_fft_smoothed[:, idx[0], idx[1]].real, color=colors_real[ind, :],
                              label='real')
        ax_imag[idx].plot(omega, c_all_fft_smoothed[:, idx[0], idx[1]].imag, color=colors_imag[ind, :],
                          label='imag')

    bias = (np.pi**-0.5) * (args.ndim * (args.ndim - 1) / 2) * (omega.max() / (args.nsim * r.t.max() * scale * dw))
    sArray[ind] = fe.entropy(c_all_fft_smoothed, T=r.t.max()).real - bias

plt.tight_layout()
# plt.legend()

# plot difference between entropy measured and true entropy

sThry = 2 * args.alpha_multiple**2 / args.k_multiple

fig_ent, ax_ent = plt.subplots(1, 2, figsize=[8, 4])
ax_ent[0].plot(scales * dw, sArray.real, 'ko')
ax_ent[0].plot([0, scales.max() * dw], [sThry, sThry],
               '--k', label='$S^{{thry}} = {0}$'.format(sThry))
ax_ent[0].set(xlabel=r'$\sigma_{kernel}$',
              ylabel=r'$d{S}/dt$')

ax_ent[1].loglog(scales * dw, abs(sThry - sArray.real), 'ko')
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
