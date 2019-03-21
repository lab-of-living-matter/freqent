∞import numpy as np
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
parser.add_argument('--ndim', type=int, default=2,
                    help='Number of dimensions of simulation, greater or equal to 2')
# parser.add_argument('--ndim', type=int, nargs=2, default=[10, 3],
#                     help=('Number of dimensions of simulation, and number of sub-dimensions '
#                          'to calculate entropy production'))
# parser.add_argument('--window', '-window', type=str, default=)

args = parser.parse_args()
# r0 = np.random.randn(args.ndim[0])
nsim_array = (2**np.arange(np.log2(args.nsim) + 1)).astype(int)
# ndim_array = np.linspace(2, args.ndim[0], int(args.ndim[1])).astype(int)

# forcing parameters
# equilibriationFrames = int(nsteps/2);
k = args.k_multiple * args.gamma
alpha = args.alpha_multiple * args.gamma

# get the different scales of the Gaussian used to smooth the data
scales = np.linspace(args.scale_array[0], args.scale_array[1], int(args.scale_array[2]))
colors = plt.cm.viridis(np.linspace(0, 1, len(scales)))


def runSim(seed):
    '''
    function to pass to multiprocessing pool
    '''
    np.random.seed(seed)
    # create object
    r = spinOscLangevin(dt=args.dt, nsteps=args.nsteps, kT=args.kT, gamma=args.gamma,
                        r0=np.random.randn(args.ndim))
    r.runSimulation(k=k, alpha=alpha)
    return r.pos


if str(args.seed_type) == 'time':
    seeds = np.arange(args.nsim) + datetime.now().microsecond
elif str(args.seed_type) == 'input':
    seeds = np.arange(args.seed_input, args.seed_input + args.nsim)
else:
    raise ValueError('Expected seed_type = {''time'', ''ints''}, received {0}.'.format(int(args.seed)))

if args.nsim < 8:
    nProcesses = args.nsim
elif args.nsim >= 8:
    nProcesses = 8

print('Running simulations...')
with multiprocessing.Pool(processes=nProcesses) as pool:
    result = pool.map(runSim, seeds)
print('Done.')

pos_all = np.asarray(result)

T = args.dt * args.nsteps
omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(args.nsteps + 1, d=args.dt))
dw = 2 * np.pi / T

sdotArray = np.zeros((len(nsim_array), len(scales)))
biasArray = np.zeros(sdotArray.shape)
fig, ax = plt.subplots()


idx_pairs = list(product(range(args.ndim), repeat=2))
# for nsimInd, nsim in enumerate(nsim_array):
#     for scaleInd, scale in enumerate(scales):
#         bias = (np.pi**-0.5) * (args.ndim * (args.ndim - 1) / 2) * (omega.max() / (nsim * T * scale * dw))
#         sdot = fe.entropy(pos_all[:nsim, :, args.nsteps // 2:],
#                           sample_spacing=args.dt,
#                           smooth_corr=True,
#                           sigma=scale)
#         sdotArray[nsimInd, scaleInd] = sdot.real
#         biasArray[nsimInd, scaleInd] = bias
#         ax.semilogx(nsim * T / 2, sdot.real, marker=(args.ndim, 0, 45), markersize=10, linestyle='None', color=colors[scaleInd, :])
# ax.set_title(r'$\alpha = {0}$'.format(args.alpha_multiple))
# ax.plot([nsim_array[0] * T / 2, nsim_array[-1] * T / 2],
#         [2 * args.alpha_multiple**2 / args.k_multiple] * 2,
#         '--k')

tDivisors = np.linspace(1, 2, 5)
t = (tDivisors**-1 - 1 / 3) * T
sdotArray = np.zeros((len(tDivisors), len(scales)))
biasArray = np.zeros(sdotArray.shape)

for tInd, tFrac in enumerate(tDivisors):
    for scaleInd, scale in enumerate(scales):
        bias = (np.pi**-0.5) * (args.ndim * (args.ndim - 1) / 2) * (omega.max() / (args.nsim * t[tInd] * scale * dw))
        sdot = fe.entropy(pos_all[:, :, args.nsteps // 3:int(args.nsteps // tDivisors[tInd])],
                          sample_spacing=args.dt,
                          smooth_corr=True,
                          sigma=scale)
        sdotArray[tInd, scaleInd] = sdot.real
        biasArray[tInd, scaleInd] = bias
        ax.semilogx(args.nsim * t[tInd], sdot.real, marker=(args.ndim, 0, 45), markersize=10, linestyle='None', color=colors[scaleInd, :])

ax.set_title(r'$\alpha = {0}$'.format(args.alpha_multiple))
ax.plot([args.nsim * t[-1], args.nsim * t[0]],
        [2 * args.alpha_multiple**2 / args.k_multiple] * 2,
        '--k')



# if args.save:
#     argDict = vars(args)
#     argDict['datetime'] = datetime.now()
#     argDict['seeds'] = seeds

#     with open(os.path.join(args.savepath, args.filename + '_params.csv'), 'w') as csv_file:
#         w = csv.DictWriter(csv_file, argDict.keys())
#         w.writeheader()
#         w.writerow(argDict)

#     fig.savefig(os.path.join(args.savepath, args.filename + '_corrFuncs.pdf'), format='pdf')
#     fig_ent.savefig(os.path.join(args.savepath, args.filename + '_entropy.pdf'), format='pdf')

plt.show()
