import numpy as np
from datetime import datetime
# import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
from brusselator_gillespie import brusselator1DFieldStochSim
import argparse
import os
import pickle
# from scipy import stats
# import freqent.freqent as fe
mpl.rcParams['pdf.fonttype'] = 42


parser = argparse.ArgumentParser()
parser.add_argument('--rates', type=float, nargs=6,
                    default=[0.5, 0.25, 1, 0.25, 1, 0.25])
parser.add_argument('--V', '-V', type=float, default=100,
                    help='Volume of solution')
parser.add_argument('--A', '-A', type=int, default=100,
                    help='Number of A molecules in solution')
parser.add_argument('--B', '-B', type=int, default=100 * 7,
                    help='Number of B molecules in solution')
parser.add_argument('--C', '-C', type=int, default=100,
                    help='Number of C molecules in solution')
parser.add_argument('--t_final', type=float, default=100,
                    help='Final time of simulations in seconds')
parser.add_argument('--n_t_points', type=int, default=1001,
                    help='Number of time points between 0 and t_final')
parser.add_argument('--nCompartments', '-K', type=int, default=64,
                    help='Number of compartments to divide space into')
parser.add_argument('--lCompartment', '-l', type=float, default=100,
                    help='Length of each compartment, to be used in getting diffusive rates')
parser.add_argument('--diffusion', '-D', type=float, nargs=2, default=[1000, 1000],
                    help='Diffusion constant of molecule X and Y')
parser.add_argument('--initial_condition', '-ic', type=str, default='random',
                    help='Initial distribution of X and Y, either random, or centered')
parser.add_argument('--seed_type', type=str, default='time',
                    help='Type of seed to use. Either "time" to use current microsecond,'
                         ' or "input" for inputting specific seeds')
parser.add_argument('--seed_input', type=int, nargs='*',
                    help='If seed_type="input", the seeds to use for the simulations')
parser.add_argument('--savepath', default='.',
                    help='path to save outputs of simulations ')

args = parser.parse_args()

if args.initial_condition == 'random':
    [X0, Y0] = (np.random.rand(2, args.nCompartments) * 7 * args.V).astype(int)
elif args.initial_condition == 'centered':
    X0 = np.zeros(args.nCompartments).astype(int)
    X0[args.nCompartments // 2 - 1:args.nCompartments // 2 + 1] = np.random.rand() * 7 * args.V
    Y0 = X0
elif args.initial_condition not in ['random', 'centered']:
    raise ValueError('Initial condition is either random or centered.\n'
                     'Currently given as {0}'.format(args.initial_condition))

t_points = np.linspace(0, args.t_final, args.n_t_points)

# get non-equilibrium parameter
# if not equal to 1, then system is away from equilibrium
alpha = args.B * args.rates[2] * args.rates[4] / (args.C * args.rates[3] * args.rates[5])

# get diffusion ratio, Dx / Dy
dRatio = args.diffusion[0] / args.diffusion[1]

# handle random seeds
if str(args.seed_type) == 'time':
    seed = datetime.now().microsecond
elif str(args.seed_type) == 'input':
    seed = int(args.seed_input)
elif str(args.seed_type) not in ['time', 'input']:
    raise ValueError('Seed_type must be either "time" or "input"\n'
                     'Currently {0}'.format(str(args.seed_type)))

brussfield = brusselator1DFieldStochSim([X0, Y0],
                                        [args.A, args.B, args.C],
                                        args.rates,
                                        args.V,
                                        t_points,
                                        args.diffusion,
                                        args.nCompartments,
                                        args.lCompartment,
                                        seed)

print('Running simulation...')
brussfield.runSimulation()
print('Done.')

fig, ax = plt.subplots(1, 2, sharey=True)

ax[0].pcolorfast(np.arange(-25, 25), t_points, brussfield.population[:, 0, :], cmap='Reds')
ax[1].pcolorfast(np.arange(-25, 25), t_points, brussfield.population[:, 1, :], cmap='Blues')

ax[0].set(xlabel='r', ylabel='t (s)', title='X(r,t)')
ax[1].set(xlabel='r', title='Y(r,t)')

# create filename and create folder with that name under savepath
filename = 'alpha{a}_dRatio{dR}'.format(a=alpha, dR=dRatio)
if not os.path.exists(os.path.join(args.savepath, filename)):
    os.makedirs(os.path.join(args.savepath, filename))

# save parameters
params = vars(args)
params['datetime'] = datetime.now()
params['seed'] = seed

with open(os.path.join(args.savepath, filename, 'params.csv'), 'w') as csv_file:
    w = csv.DictWriter(csv_file, params.keys())
    w.writeheader()
    w.writerow(params)

# save figures
fig.savefig(os.path.join(args.savepath, filename, 'traj.pdf'), format='pdf')

with open(os.path.join(args.savepath, filename, 'data.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(brussfield, f, pickle.HIGHEST_PROTOCOL)


plt.show()
