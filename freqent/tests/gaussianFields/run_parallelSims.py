import numpy as np
from datetime import datetime
import time
import matplotlib as mpl
# mpl.use('Agg')  # use backend that doesn't immediately create figures
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import multiprocessing
from gaussianFieldSimulation import *
import argparse
import os
# from scipy import stats
import freqent.freqentn as fen
import h5py


def get_traj(seed):
    '''
    function to pass to multiprocessing pool to run parallel simulations
    '''
    np.random.seed(seed)
    XY_init = np.random.randn(2, args.nsites)
    f = gaussianFields1D(dt=args.dt, dx=args.dx,
                         ic=XY_init, nsteps=args.nsteps)
    f.runSimulation(alpha=args.alpha)

    return f


parser = argparse.ArgumentParser()

parser.add_argument('--dt', type=float, default=1e-4,
                    help='time step')
parser.add_argument('--dx', type=float, default=1e-1,
                    help='lattice spacing')
parser.add_argument('--nsteps', type=int, default=1000000,
                    help='number of time steps')
parser.add_argument('--nsites', type=int, default=200,
                    help='number of lattice sites')
parser.add_argument('--alpha', '-a', type=float, default=0,
                    help='strength of nonequilibrium forcing')
parser.add_argument('--nSim', type=int, default=10,
                    help='Number of simulations to run in parallel')
parser.add_argument('--seed_type', type=str, default='time',
                    help='Type of seed to use. Either "time" to use current microsecond,'
                         ' or "input" for inputting specific seeds')
parser.add_argument('--seed_input', type=int, nargs='*',
                    help='If seed_type="input", the seeds to use for the simulations')
parser.add_argument('--sigma', '-std', type=int, default=2,
                    help='Size of Gaussian to smooth with, in units of sample sample_spacing')
parser.add_argument('--savepath', default='.',
                    help='path to save outputs of simulations ')

args = parser.parse_args()

# handle random seeds
if str(args.seed_type) == 'time':
    seeds = np.zeros(args.nSim, dtype=int)
    for ii in range(args.nSim):
        seeds[ii] = datetime.now().microsecond
        time.sleep(0.0001)

elif str(args.seed_type) == 'input':
    seeds = list(args.seed_input)
elif str(args.seed_type) not in ['time', 'input']:
    raise ValueError('Seed_type must be either "time" or "input"\n'
                     'Currently {0}'.format(str(args.seed_type)))

if int(args.nSim) < 10:
    nProcesses = int(args.nSim)
else:
    nProcesses = 10

print('Running simulations...')
tic = time.time()
with multiprocessing.Pool(processes=nProcesses) as pool:
    result = pool.map(get_traj, seeds)
toc = time.time()
print('Done. Total time = {t:.2f} s'.format(t=toc - tic))

# save trajectories
trajs = np.zeros((args.nSim, 2, args.nsteps + 1, args.nsites))

# get time and space arrays
L = result[0].L
t = result[0].t

for ii in range(args.nSim):
    traj = result[ii].pos
    trajs[ii] = traj

# Calculate mean entropy production rate from spectral method
epr_spectral = (fen.entropy(trajs[: args.nsteps // 2:, :],
                            sample_spacing=[args.dt, args.dx],
                            window='boxcar',
                            nperseg=None,
                            noverlap=None,
                            nfft=None,
                            detrend='constant',
                            smooth_corr=True,
                            sigma=args.sigma,
                            subtract_bias=True,
                            many_traj=True)).real


# create filename and create folder with that name under savepath
filename = 'alpha{a}_nSim{n}_sigma{s}'.format(a=args.alpha, n=args.nSim, s=args.sigma)
if not os.path.exists(os.path.join(args.savepath, filename)):
    os.makedirs(os.path.join(args.savepath, filename))

# save parameters
params = vars(args)
params['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
params['seeds'] = seeds

# all you need are the exact seeds. seed_input is often None, which
# creates a problem in saving below
params.pop('seed_input')
params.pop('seed_type')

# with open(os.path.join(args.savepath, filename, 'params.csv'), 'w') as csv_file:
#     w = csv.DictWriter(csv_file, params.keys())
#     w.writeheader()
#     w.writerow(params)

# save representative figure
result[0].plotTrajectory(savepath=os.path.join(args.savepath, filename),
                         delta=10)

dat = {'trajs': trajs,
       't': t,
       'L': L,
       'epr_spectral': epr_spectral}

with h5py.File(os.path.join(args.savepath, filename, 'data.hdf5'), 'w') as f:
    # save data and params to hdf5 file
    datagrp = f.create_group('data')
    paramsgrp = f.create_group('params')

    for name in dat.keys():
        datagrp.create_dataset(name, data=dat[name])

    for name in params.keys():
        paramsgrp.create_dataset(name, data=params[name])
