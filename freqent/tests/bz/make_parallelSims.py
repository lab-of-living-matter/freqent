import numpy as np
from datetime import datetime
import time
import matplotlib as mpl
mpl.use('Agg')  # use backend that doesn't immediately create figures

import matplotlib.pyplot as plt
import multiprocessing
import csv
from brusselator_gillespie import brusselatorStochSim
import argparse
import os
import pickle
from scipy import stats
import freqent.freqent as fe
mpl.rcParams['pdf.fonttype'] = 42


def get_traj(seed):
    '''
    function to pass to multiprocessing pool to run parallel simulations
    '''
    np.random.seed(seed)
    [X0, Y0] = (np.random.rand(2) * 7 * args.V).astype(int)
    bz = brusselatorStochSim([X0, Y0, args.A, args.B, args.C], args.rates, args.V, t_points, seed)
    bz.runSimulation()

    return [bz.population, bz.ep, bz.n]


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
parser.add_argument('--savepath', default='.',
                    help='path to save outputs of simulations ')

args = parser.parse_args()

# get time points of outputs
t_points = np.linspace(0, args.t_final, args.n_t_points)

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

with multiprocessing.Pool(processes=nProcesses) as pool:
    result = pool.map(get_traj, seeds.astype(int))

# plotting and preparing data for saving
fig_traj, ax_traj = plt.subplots()
fig_ep, ax_ep = plt.subplots()

trajs = np.zeros((args.nSim, 2, args.n_t_points))
eps = np.zeros((args.nSim, args.n_t_points))
n = np.zeros(args.nSim)

for ii in range(args.nSim):
    traj = result[ii][0].T
    ep = result[ii][1]
    ax_traj.plot(traj[0], traj[1], 'k', alpha=0.2)
    ax_ep.plot(t_points, ep, 'k', alpha=0.3)
    trajs[ii] = traj
    eps[ii] = ep
    n[ii] = result[ii][2]

ax_ep.plot(t_points, eps.mean(axis=0), 'r', linewidth=2)

ax_traj.set(xlabel='X', ylabel='Y')
ax_traj.set_aspect(np.diff(ax_traj.set_xlim())[0] / np.diff(ax_traj.set_ylim())[0])
plt.tight_layout()

ax_ep.set(xlabel='t [s]', ylabel=r'$\Delta S$')
ax_ep.set_aspect(np.diff(ax_ep.set_xlim())[0] / np.diff(ax_ep.set_ylim())[0])
plt.tight_layout()

# Calculate mean entropy production rate from halway through the simulation to ensure steady state reached
epr, intercept, r_value, p_val, std_err = stats.linregress(t_points[args.n_t_points // 2:],
                                                           eps.mean(axis=0)[args.n_t_points // 2:])

# Calculate mean entropy production rate from spectral method
epr_spectral = (fe.entropy(trajs,
                           sample_spacing=np.diff(t_points)[0],
                           window='boxcar',
                           nperseg=None,
                           noverlap=None,
                           nfft=None,
                           detrend='constant',
                           padded=False,
                           smooth_corr=True,
                           sigma=2,
                           subtract_bias=True)).real


# create filename and create folder with that name under savepath
filename = 'alpha{a}_nSim{n}'.format(a=alpha, n=args.nSim)
if not os.path.exists(os.path.join(args.savepath, filename)):
    os.makedirs(os.path.join(args.savepath, filename))

# save parameters
params = vars(args)
params['datetime'] = datetime.now()
params['seeds'] = seeds

with open(os.path.join(args.savepath, filename, 'params.csv'), 'w') as csv_file:
    w = csv.DictWriter(csv_file, params.keys())
    w.writeheader()
    w.writerow(params)

# save figures
fig_traj.savefig(os.path.join(args.savepath, filename, 'traj.pdf'), format='pdf')
fig_ep.savefig(os.path.join(args.savepath, filename, 'ep.pdf'), format='pdf')

data = {'trajs': trajs,
        'eps': eps,
        't_points': t_points,
        'n': n,
        'epr': epr,
        'epr_spectral': epr_spectral}
with open(os.path.join(args.savepath, filename, 'data.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
