
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
mpl.rcParams['pdf.fonttype'] = 42


parser = argparse.ArgumentParser(description=('Perform simulations of Brownian particles'
                                              ' in a harmonic potential plus a rotating'
                                              ' force.'))
parser.add_argument('--save', type=bool, default=False,
                    help='Boolean of whether to save outputs')
parser.add_argument('--savepath', type=str, default='',
                    help='Path to save outputs if save')
parser.add_argument('--filename', type=str, default='entropy_dtError',
                    help='Name of image file to save at savepath')
parser.add_argument('--gamma', type=float, default=2e-8,
                    help='drag on 1um particle in water in kg/s')
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
parser.add_argument('--seed_type', type=str, default='time',
                    help=('a string to decide what seed to use when generating trajectories. '
                          'Use ''time'' to use current microsecond or ''nsim'' to use nsim as seeds.'))
parser.add_argument('-dt', '--dtArray', type=float, nargs=3, default=[-5, -1, 5],
                    help='build array of different time steps to be put in as arguments to np.logspace. In format [minPower, maxPower, N]')

args = parser.parse_args()

# forcing parameters
# equilibriationFrames = int(nsteps/2);
k = args.k_multiple * args.gamma
alpha = args.alpha_multiple * args.gamma
dtArray = np.logspace(*args.dtArray)
sArray = np.zeros(len(dtArray) * args.nsim, dtype=complex)

# get seeds
if str(args.seed_type) == 'time':
    seed = datetime.now().microsecond
    np.random.seed(seed)
elif str(args.seed_type) == 'nsim':
    seed = args.nsim
    np.random.seed(seed)
else:
    ValueError('Expected seed_type = {''time'', ''nsim''}, received {0}.'.format(str(args.seed)))

# create object
for tind, dt in enumerate(dtArray):
    for nind, n in enumerate(range(args.nsim)):
        txt = 'dt = {dt}, simulation {n}/{N}'.format(dt=dt, n=nind + 1, N=args.nsim)
        print(txt)
        r = spinOscLangevin(dt=dt, nsteps=args.nsteps, kT=args.kT, gamma=args.gamma, r0=np.random.randn(2))
        r.runSimulation(k=k, alpha=alpha)
        c_fft, omega = fe.corr_matrix(r.pos,
                                      sample_spacing=dt,
                                      return_fft=True)
        sArray[tind * args.nsim + nind] = fe.entropy(c_fft, sample_spacing=dt)

fig, ax = plt.subplots(1, 2, figsize=[8, 4])

ax[0].loglog(dtArray, dtArray**-1, 'r--')
ax[0].loglog(np.repeat(dtArray, args.nsim), sArray.real, 'ko', alpha=0.75)
ax[0].set(xlabel=r'$\Delta t\ [s]$', ylabel=r'$dS/dt\ [k_B \ s^{-1}]$')
ax[0].grid(1)
# ax[0].set_aspect(np.diff(ax[0].set_xlim())[0] / np.diff(ax[0].set_ylim())[0])

ax[1].semilogx(np.repeat(dtArray, args.nsim), np.repeat(dtArray, args.nsim) * sArray.real, 'ko', alpha=0.75)
ax[1].set(xlabel=r'$\Delta t\ [s]$', ylabel=r'$\Delta t \times dS/dt$')
# ax[1].set_aspect(np.diff(ax[1].set_xlim())[0] / np.diff(ax[1].set_ylim())[0])

plt.tight_layout()

if args.save:
    argDict = vars(args)
    argDict['datetime'] = datetime.now()
    argDict['seed'] = seed

    with open(os.path.join(args.savepath, args.filename + '_params.csv'), 'w') as csv_file:
        w = csv.DictWriter(csv_file, argDict.keys())
        w.writeheader()
        w.writerow(argDict)

    fig.savefig(os.path.join(args.savepath, args.filename + '.pdf'), format='pdf')

plt.show()
