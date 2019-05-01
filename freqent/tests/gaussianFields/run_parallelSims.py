import numpy as np
from datetime import datetime
import time
import matplotlib as mpl
mpl.use('Agg')  # use backend that doesn't immediately create figures
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import multiprocessing
import csv
from gaussianFieldSimulation import *
import argparse
import os
from scipy import stats
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
parser.add_argument('--nsteps', type=int, default=1e6,
                    help='number of time steps')
parser.add_argument('--nsites', type=int, default=2e2,
                    help='number of lattice sites')
parser.add_argument('--alpha', '-a', type=float, default=0,
                    help='strength of nonequilibrium forcing')


args = parser.parse_args()

# get time points of outputs
t_points = np.linspace(0, args.t_final, args.n_t_points)

# handle random seeds
if str(args.seed_type) == 'time':
    seeds = np.zeros(args.nSim)
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
    result = pool.map(get_traj, seeds.astype(int))
toc = time.time()
print('Done. Total time = {t:.2f} s'.format(t=toc - tic))

# plotting and preparing data for saving
fig_traj, ax_traj = plt.subplots(1, 2, sharey=True)
fig_ep, ax_ep = plt.subplots()
fig_ep_blind, ax_ep_blind = plt.subplots()

# save trajectories
trajs = np.zeros((args.nSim, 2, args.n_t_points, args.nCompartments))

# save entropy productions
eps = np.zeros((args.nSim, args.n_t_points), dtype=float)

# save blinded entropy production
ep_blinds = np.zeros((args.nSim, args.n_t_points), dtype=float)

# save total number of simulatinon steps, in order to see what percentage of data we're getting
ns = np.zeros(args.nSim)

for ii in range(args.nSim):
    traj = np.moveaxis(result[ii].population, 1, 0)
    trajs[ii] = traj

    ep = result[ii].ep
    eps[ii] = ep

    ep_blind = result[ii].ep_blind
    ep_blinds[ii] = ep_blind

    n = result[ii].n
    ns[ii] = n

    ax_traj[0].pcolorfast(list(range(args.nCompartments)),
                          t_points, traj[0], cmap='Reds')
    ax_traj[1].pcolorfast(list(range(args.nCompartments)),
                          t_points, traj[1], cmap='Blues')
    ax_ep.plot(t_points, ep, 'k', alpha=0.3)
    ax_ep_blind.plot(t_points, ep_blind, 'k', alpha=0.3)

ax_ep.plot(t_points, eps.mean(axis=0), 'r', linewidth=2)
ax_ep_blind.plot(t_points, ep_blinds.mean(axis=0), 'r', linewidth=2)

ax_traj[0].set(xlabel='x', ylabel='t', title=r'$X(x,t)$')
ax_traj[1].set(xlabel='x', title=r'$Y(x,t)$')
plt.tight_layout()

ax_ep.set(xlabel='t', ylabel=r'$\Delta S$', title='')
ax_ep.set_aspect(np.diff(ax_ep.set_xlim())[0] / np.diff(ax_ep.set_ylim())[0])
plt.tight_layout()

ax_ep_blind.set(xlabel='t', ylabel=r'$\Delta S_{blind}$')
ax_ep_blind.set_aspect(np.diff(ax_ep_blind.set_xlim())[0] / np.diff(ax_ep_blind.set_ylim())[0])
plt.tight_layout()

# Calculate mean entropy production rate from halway through the simulation to ensure steady state reached
epr, _, _, _, _ = stats.linregress(t_points[args.n_t_points // 2:],
                                   eps.mean(axis=0)[args.n_t_points // 2:])

epr_blind, _, _, _, _ = stats.linregress(t_points[args.n_t_points // 2:],
                                         ep_blinds.mean(axis=0)[args.n_t_points // 2:])

dt = np.diff(t_points)[0]
dx = args.lCompartment
# Calculate mean entropy production rate from spectral method
epr_spectral = (fen.entropy(trajs[: args.n_t_points//2:, :],
                            sample_spacing=[dt, dx],
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
filename = 'alpha{a}_nSim{n}_sigma{s}'.format(a=alpha, n=args.nSim, s=args.sigma)
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

# save figures
fig_traj.savefig(os.path.join(args.savepath, filename, 'traj.pdf'), format='pdf')
fig_ep.savefig(os.path.join(args.savepath, filename, 'ep.pdf'), format='pdf')
fig_ep_blind.savefig(os.path.join(args.savepath, filename, 'ep_blind.pdf'), format='pdf')

dat = {'trajs': trajs,
       'eps': eps,
       'ep_blinds': ep_blinds,
       't_points': t_points,
       'n': n,
       'epr': epr,
       'epr_blind': epr_blind,
       'epr_spectral': epr_spectral}

with h5py.File(os.path.join(args.savepath, filename, 'data.hdf5'), 'w') as f:
    # save data and params to hdf5 file
    datagrp = f.create_group('data')
    paramsgrp = f.create_group('params')

    for name in dat.keys():
        datagrp.create_dataset(name, data=dat[name])

    for name in params.keys():
        paramsgrp.create_dataset(name, data=params[name])


plt.show()
# with open(os.path.join(args.savepath, filename, 'data.pickle'), 'wb') as f:
#     # Pickle the 'data' dictionary using the highest protocol available.
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
