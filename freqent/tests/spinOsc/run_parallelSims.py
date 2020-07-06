import numpy as np
from spinOscSimulation import spinOscLangevin
import freqent.freqent as fe
import matplotlib as mpl
import os
import argparse
import multiprocessing
from datetime import datetime
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
parser.add_argument('--nsim', type=int, default=64,
                    help='number of simulations to run. Make a power of 2')
parser.add_argument('-a', '--alpha', type=float, default=2,
                    help='Rotational force strength')
parser.add_argument('--ndim', type=int, default=2,
                    help='Number of dimensions of simulation, greater or equal to 2')

# things to calculate with
parser.add_argument('--seed_type', type=str, default='time',
                    help=('a string to decide what seed to use when generating '
                          'trajectories. use "time" to use current microsecond or '
                          '"input" to directly input the seed'))
parser.add_argument('--seed_input', '-seed', type=float, default=None,
                    help='if seed_type=input, what the seed explicitly is')
parser.add_argument('--scale_array', '-scale', type=float, nargs=3, default=[1, 10, 10],
                    help='standard deviations of Gaussians used to smooth correlation function')
parser.add_argument('--n_epr', '-ns', type=int, nargs=5, default=[4, 8, 16, 32, 48],
                    help='number of simulations to average over when calculating epr')


args = parser.parse_args()

# scales of the Gaussian used for smoothing
scales = np.linspace(args.scale_array[0], args.scale_array[1], int(args.scale_array[2]))

T = args.nsteps * args.dt
if T < 10:
    raise ValueError('Total time of simulation is less than 10. Use longer to assure steady state')


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
sdot_array = np.zeros((len(args.n_epr), len(scales)))

for nInd, n in enumerate(args.n_epr):
    for scaleInd, scale in enumerate(scales):
        # calculate epr for data after 10 simulation times
        sdot = fe.entropy(trajs[:n, :, int(10 / args.dt):],
                          sample_spacing=args.dt,
                          window='boxcar',
                          nperseg=None,
                          noverlap=None,
                          nfft=None,
                          detrend='constant',
                          padded=False,
                          smooth_corr=True,
                          sigma=scale,
                          subtract_bias=True,
                          return_epf=False)
        sdot_array[nInd, scaleInd] = sdot.real


t_epr = T - 10
dw = 2 * np.pi / t_epr

if args.save:

    fullpath = os.path.join(args.savepath, datetime.now().strftime('%y%m%d'))
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)

    params = {'dt': args.dt,
              'nsteps': args.nsteps,
              'alpha': args.alpha,
              'nsim': args.nsim,
              'ndim': args.ndim,
              'n_epr': args.n_epr,
              't_epr': t_epr,
              'dw': dw,
              'scales': scales,
              'sigma': scales * dw,
              'seeds': seeds}

    paramsattrs = {'dt': 'simulation step size',
                   'nsteps': 'number of simulation steps',
                   'alpha': 'strength of non-equilibrium force',
                   'nsim': 'number of simulations',
                   'ndim': 'number of dimensions in simulations',
                   'n_epr': 'number of simulations used when calculating each epr',
                   't_epr': 'total time used for each calculation of epr',
                   'dw': 'spacing of frequencies in correlation function',
                   'scales': 'widths of Gaussians used to smooth correlation functions in units of dw',
                   'sigma': 'widths of Gaussians used to smooth correlation functions in units of inverse simulation time (i.e. sigma = scales * dw)',
                   'seeds': 'seeds for random number generator'}

    data = {'trajs': trajs,
            't': np.linspace(0, T, args.nsteps + 1),
            'sdot_array': sdot_array}

    dataattrs = {'trajs': 'all trajectories',
                 't': 'time array for trajectories',
                 'sdot_array': 'entropy production rate array in shape [len(n_epr), len(scales)]'}

    filename = 'alpha{a}_nSim{n}_dim{d}_nsteps{nt}.hdf5'.format(a=args.alpha, n=args.nsim, d=args.ndim, nt=args.nsteps)
    with h5py.File(os.path.join(fullpath, filename), 'w') as f:
        datagrp = f.create_group('data')
        paramsgrp = f.create_group('params')

        for dataname in data.keys():
            d = datagrp.create_dataset(dataname, data=data[dataname])
            d.attrs['description'] = dataattrs[dataname]

        for paramsname in params.keys():
            p = paramsgrp.create_dataset(paramsname, data=params[paramsname])
            p.attrs['description'] = paramsattrs[paramsname]
