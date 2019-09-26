import numpy as np
from datetime import datetime
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing
from brusselator_gillespie import brusselatorStochSim
import argparse
import os
from scipy import stats
import freqent.freqent as fe
import h5py

mpl.use('Agg')  # use backend that doesn't immediately create figures
mpl.rcParams['pdf.fonttype'] = 42


def get_traj(seed):
    '''
    function to pass to multiprocessing pool to run parallel simulations
    '''
    np.random.seed(seed)
    [X0, Y0] = (np.random.rand(2) * 7 * args.V).astype(int)
    bz = brusselatorStochSim([X0, Y0, args.A, args.B, args.C], args.rates, args.V, t_points, seed)
    bz.runSimulation()

    return bz


parser = argparse.ArgumentParser()
parser.add_argument('--rates', type=float, nargs=6,
                    default=[0.5, 0.25, 1, 0.25, 1, 0.25])
parser.add_argument('--V', type=float, default=100,
                    help='Volume of solution')
parser.add_argument('--A', type=int, default=100,
                    help='Number of A molecules in solution')
parser.add_argument('--B', type=int, default=100 * 7,
                    help='Number of B molecules in solution')
parser.add_argument('--C', type=int, default=100,
                    help='Number of C molecules in solution')
parser.add_argument('--t_final', type=float, default=100,
                    help='Final time of simulations in seconds')
parser.add_argument('--n_t_points', type=int, default=1001,
                    help='Number of time points between 0 and t_final')
parser.add_argument('--nSim', type=int, default=10,
                    help='Number of simulations to run in parallel')
parser.add_argument('--seed_type', type=str, default='time',
                    help='Type of seed to use. Either "time" to use current microsecond,'
                         ' or "input" for inputting specific seeds')
parser.add_argument('--seed_input', type=int, nargs='*',
                    help='If seed_type="input", the seeds to use for the simulations')
parser.add_argument('--sigma', '-std', type=int, default=2,
                    help='Size of Gaussian to smooth with in units of sample spacing')
parser.add_argument('--savepath', default='.',
                    help='path to save outputs of simulations ')

args = parser.parse_args()

# get time points of outputs
t_points = np.linspace(0, args.t_final, args.n_t_points)
dt = np.diff(t_points)[0]

# get non-equilibrium parameter
# if not equal to 1, then system is away from equilibrium
alpha = args.B * args.rates[2] * args.rates[4] / (args.C * args.rates[3] * args.rates[5])

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
fig_traj, ax_traj = plt.subplots()
fig_ep, ax_ep = plt.subplots()
fig_ep_blind, ax_ep_blind = plt.subplots()

# save entropy produced
eps = np.zeros((args.nSim, args.n_t_points), dtype=float)

# save blind entropy produced
ep_blinds = np.zeros((args.nSim, args.n_t_points), dtype=float)

# save trajectories
trajs = np.zeros((args.nSim, 2, args.n_t_points), dtype=int)

# save total number of simulation steps, in order to see what percentage of data we're getting
ns = np.zeros(args.nSim, dtype=int)

for ii in range(args.nSim):
    traj = result[ii].population.T
    trajs[ii] = traj

    ep = result[ii].ep
    eps[ii] = ep

    ep_blind = result[ii].ep_blind
    ep_blinds[ii] = ep_blind

    n = result[ii].n
    ns[ii] = n

    ax_traj.plot(traj[0], traj[1], 'k', alpha=0.2)
    ax_ep.plot(t_points, ep, 'k', alpha=0.3)
    ax_ep_blind.plot(t_points, ep_blind, 'k', alpha=0.3)


ax_ep.plot(t_points, eps.mean(axis=0), 'r', linewidth=2)
ax_ep_blind.plot(t_points, ep_blinds.mean(axis=0), 'r', linewidth=2)

ax_traj.set(xlabel='X', ylabel='Y')
ax_traj.set_aspect(np.diff(ax_traj.set_xlim())[0] / np.diff(ax_traj.set_ylim())[0])
plt.tight_layout()

ax_ep.set(xlabel='t [s]', ylabel=r'$\Delta S$')
ax_ep.set_aspect(np.diff(ax_ep.set_xlim())[0] / np.diff(ax_ep.set_ylim())[0])
plt.tight_layout()

ax_ep_blind.set(xlabel='t [s]', ylabel=r'$\Delta S$ (blind)')
ax_ep_blind.set_aspect(np.diff(ax_ep_blind.set_xlim())[0] / np.diff(ax_ep_blind.set_ylim())[0])
plt.tight_layout()

# Calculate mean entropy production rate from halway through the simulation to ensure steady state reached
epr, intercept, r_value, p_val, std_err = stats.linregress(t_points[args.n_t_points // 2:],
                                                           eps.mean(axis=0)[args.n_t_points // 2:])

epr_blind, intercept, r_value, p_val, std_err = stats.linregress(t_points[args.n_t_points // 2:],
                                                                 ep_blinds.mean(axis=0)[args.n_t_points // 2:])

# Calculate mean entropy production rate from spectral method
# epr_spectral, epr_spectral_density, w = (fe.entropy(trajs[..., args.n_t_points // 2:],
#                                          sample_spacing=dt,
#                                          window='boxcar',
#                                          nperseg=None,
#                                          noverlap=None,
#                                          nfft=None,
#                                          detrend='constant',
#                                          padded=False,
#                                          smooth_corr=True,
#                                          sigma=args.sigma,
#                                          subtract_bias=True,
#                                          return_density=True)).real


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

# save figures
fig_traj.savefig(os.path.join(args.savepath, filename, 'traj.pdf'), format='pdf')
fig_ep.savefig(os.path.join(args.savepath, filename, 'ep.pdf'), format='pdf')
fig_ep_blind.savefig(os.path.join(args.savepath, filename, 'ep_blind.pdf'), format='pdf')

dat = {'trajs': trajs,
       'eps': eps,
       'ep_blinds': ep_blinds,
       't_points': t_points,
       'ns': ns,
       'epr': epr,
       'epr_blind': epr_blind}

datattrs = {'trajs': 'trajectory data',
            'eps': 'entropy produced by each trajectory, used to get epr',
            'ep_blinds': 'blind entropy produced by each trajectory, used to get epr_blind',
            't_points': 'time points of outputs',
            'ns': 'number of total reactions taken in simulation',
            'epr': 'true entropy production rate',
            'epr_blind': 'blinded entropy production rate'}

with h5py.File(os.path.join(args.savepath, filename, 'data.hdf5'), 'w') as f:
    # save data and params to hdf5 file
    datagrp = f.create_group('data')
    paramsgrp = f.create_group('params')

    for name in dat.keys():
        d = datagrp.create_dataset(name, data=dat[name])
        d.attrs['description'] = datattrs[name]

    for name in params.keys():
        paramsgrp.create_dataset(name, data=params[name])
