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
parser.add_argument('--nsim', type=int, default=128,
                    help='number of simulations to run. Make a power of 2')
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
parser.add_argument('--ndim', type=int, nargs=2, default=[10, 3],
                    help=('Number of dimensions of simulation, and number of sub-dimensions '
                          'to calculate entropy production'))
# parser.add_argument('--window', '-window', type=str, default=)

args = parser.parse_args()
r0 = np.random.randn(args.ndim[0])
nsim_array = (2**np.arange(np.log2(args.nsim) + 1)).astype(int)
ndim_array = np.linspace(2, args.ndim[0], int(args.ndim[1])).astype(int)
# create object
r = spinOscLangevin(dt=args.dt, nsteps=args.nsteps, kT=args.kT, gamma=args.gamma,
                    r0=r0)

# forcing parameters
# equilibriationFrames = int(nsteps/2);
k = args.k_multiple * r.gamma
alpha = args.alpha_multiple * r.gamma

# get the different scales of the Gaussian used to smooth the data
scales = np.linspace(args.scale_array[0], args.scale_array[1], int(args.scale_array[2]))
colors = plt.cm.viridis(np.linspace(0, 1, len(scales)))

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

if args.nsim < 8:
    nProcesses = args.nsim
elif args.nsim >= 8:
    nProcesses = 8

with multiprocessing.Pool(processes=nProcesses) as pool:
    result_freq = pool.map(get_corr_mat_fft, seeds)

c_all_fft = np.asarray(result_freq)

omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(r.t), d=args.dt))
dw = 2 * np.pi / (args.dt * len(r.t))

sdotArray = np.zeros((len(ndim_array), len(nsim_array), len(scales)))
biasArray = np.zeros(sdotArray.shape)
fig, ax = plt.subplots()

for dInd, d in enumerate(ndim_array):
    idx_pairs = list(product(range(d), repeat=2))
    for nsimInd, nsim in enumerate(nsim_array):
        c_all_fft_submean = np.mean(c_all_fft[:nsim, :, :d, :d], axis=0)
        c_all_fft_smoothed = np.zeros(c_all_fft_submean.shape, dtype=complex)
        for scaleInd, scale in enumerate(scales):
            bias = (np.pi**-0.5) * (d * (d - 1) / 2) * (omega.max() / (nsim * r.t.max() * scale * dw))
            gauss = Gaussian1DKernel(scale)
            for idx in idx_pairs:
                c_all_fft_smoothed[:, idx[0], idx[1]].real = convolve(c_all_fft_submean[:, idx[0], idx[1]].real, gauss,
                                                                      normalize_kernel=True)
                c_all_fft_smoothed[:, idx[0], idx[1]].imag = convolve(c_all_fft_submean[:, idx[0], idx[1]].imag, gauss,
                                                                      normalize_kernel=True)

            sdot = fe.entropy(c_all_fft_smoothed, T=r.t.max()).real - bias
            sdotArray[dInd, nsimInd, scaleInd] = sdot
            biasArray[dInd, nsimInd, scaleInd] = bias
            ax.semilogx(nsim * r.t.max() * (1 + dInd / 10), sdot, marker=(d, 0, 45), markersize=10, linestyle='None', color=colors[scaleInd, :])

ax.set_title(r'$\alpha = {0}$'.format(args.alpha_multiple))

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
