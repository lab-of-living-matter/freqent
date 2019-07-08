import numpy as np
import matplotlib.pyplot as plt
from spinOscSimulation import spinOscLangevin
import freqent.freqent as fe
import matplotlib as mpl
import os
import argparse
import multiprocessing
from datetime import datetime
from itertools import product
import h5py
# import scipy.signal as signal

mpl.rcParams['pdf.fonttype'] = 42
# plt.close('all')

parser = argparse.ArgumentParser(description=('Perform simulations of Brownian particles'
                                              ' in a harmonic potential plus a rotating'
                                              ' force.'))
parser.add_argument('--save', type=bool, default=True,
                    help='Boolean of whether to save outputs')
parser.add_argument('--savepath', type=str, default='',
                    help='Path to save outputs if save')

# simulation parameters
parser.add_argument('--dt', type=float, default=1e-3,
                    help='time step of simulation in seconds')
parser.add_argument('--nsteps', type=int, default=int(1e6),
                    help='number of simulation steps')
parser.add_argument('--nsim', type=int, default=128,
                    help='number of simulations to run. Make a power of 2')
parser.add_argument('-a', '--alpha', type=float, default=2,
                    help='Rotational force strength')

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


args = parser.parse_args()

# scales of the Gaussian used for smoothing
scales = np.linspace(args.scale_array[0], args.scale_array[1], int(args.scale_array[2]))


def runSim(seed):
    '''
    function to pass to multiprocessing pool
    '''
    np.random.seed(seed)
    # create object
    r = spinOscLangevin(dt=args.dt, nsteps=args.nsteps,
                        r0=np.random.randn(args.ndim))
    r.runSimulation(alpha=args.alpha)
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

trajs = np.asarray(result)

t_divisors = np.linspace(1, 2, 5)
T = (args.nsteps + 1) * args.dt
t_epr = (t_divisors**-1 - 1 / 3) * T  # the total amount of time used to calculate the epr
sdot_array = np.zeros((len(t_divisors), len(scales)))

for tInd, tFrac in enumerate(t_divisors):
    for scaleInd, scale in enumerate(scales):
        sdot = fe.entropy(trajs[:, :, args.nsteps // 3:int(args.nsteps // tFrac)],
                          sample_spacing=args.dt,
                          window='boxcar',
                          npersep=None,
                          noverlap=None,
                          nfft=None,
                          detrend='constant',
                          padded=False,
                          smooth_corr=True,
                          sigma=scale,
                          subtract_bias=True,
                          return_density=False)
        sdot_array[tInd, scaleInd] = sdot.real


if args.save:

    fullpath = os.path.join(args.savepath, datetime.now().strftime('%y%m%d'))
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)

    params = {'dt': args.dt,
              'nsteps': args.nsteps,
              'alpha': args.alpha,
              'nsim': args.nsim,
              't_divisors': t_divisors,
              't_epr': t_epr,
              'scales': scales,
              'seeds': seeds}

    paramsattrs = {'dt': 'simulation step size',
                   'nsteps': 'number of simulation steps',
                   'alpha': 'strength of non-equilibrium force',
                   'nsim': 'number of simulations',
                   't_divisors': 'used to calculate epr for times between T/3 and T/t_divisor',
                   't_epr': 'total time used for each calculation of epr',
                   'scales': 'widths of Gaussians used to smooth correlation functions',
                   'seeds': 'seeds for random number generator'}

    data = {'trajs': trajs,
            't': np.linspace(0, T, args.nsteps + 1),
            'sdot_array': sdot_array}

    dataattrs = {'trajs': 'all trajectories',
                 't': 'time array for trajectories',
                 'sdot_array': 'entropy production rate array in shape [len(t_divisors), len(scales)]'}

    filename = 'alpha{a}_nSim{n}_T{t}.h5py'.format(a=args.alpha, n=args.nsim, t=T)
    with h5py.File(os.path.join(fullpath, filename), 'w') as f:
        datagrp = f.create_group('data')
        paramsgrp = f.create_group('params')

        for dataname in data.keys():
            d = datagrp.create_dataset(dataname, data=data[dataname])
            d.attrs['description'] = dataattrs[dataname]

        for paramsname in params.keys():
            p = paramsgrp.create_dataset(paramsname, data=params[paramsname])
            p.attrs['description'] = paramsattrs[paramsname]
