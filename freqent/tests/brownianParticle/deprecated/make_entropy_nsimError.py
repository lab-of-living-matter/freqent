
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
parser.add_argument('--dt', type=float, default=1e-3,
                    help='time step of simulation in seconds')
parser.add_argument('--kT', type=float, default=4e-9,
                    help='thermal energy in kg um^2 / s^2')
parser.add_argument('-k', '--k_multiple', type=float, default=2,
                    help='Spring constant of harmonic potential in units of gamma')
parser.add_argument('-a', '--alpha_multiple', type=float, default=2,
                    help='Rotational force strength in units of gamma')
parser.add_argument('--seed', type=str, default='time',
                    help=('a string to decide what seed to use when generating trajectories. '
                          'Use ''time'' to use current microsecond or ''nsim'' to use nsim as seeds.'))
parser.add_argument('-nsim', '--nsimArray', type=float, nargs=3, default=[1, 100, 10],
                    help='build array of different time steps to be put in as arguments to np.arange.')

args = parser.parse_args()

# forcing parameters
# equilibriationFrames = int(nsteps/2);
k = args.k_multiple * args.gamma
alpha = args.alpha_multiple * args.gamma
nsimArray = np.arange(*args.nsimArray)
sArray = np.zeros(len(nsimArray), dtype=complex)

# get seeds
if str(args.seed) == 'time':
    seed = datetime.now().microsecond
    np.random.seed(seed)
elif str.isnumeric(args.seed):
    seed = args.seed
    np.random.seed(seed)
else:
    ValueError('Expected seed = "time" or number, received {0}.'.format(str(args.seed)))

# create object
r = spinOscLangevin(dt=args.dt, nsteps=args.nsteps, kT=args.kT, gamma=args.gamma, r0=np.random.randn(2))


def get_corr_mat(seed):
    '''
    helper function to pass to multiprocessing pool
    '''
    # np.random.seed(seed)
    r.reset()
    r.runSimulation(k=k, alpha=alpha)
    c, t = fe.corr_matrix(r.pos,
                          sample_spacing=r.dt,
                          mode='full',
                          method='auto',
                          return_fft=False)
    return c


# create object
for nind, n in enumerate(nsimArray):
    txt = 'nsim = {nsim}, simulation set {n}/{N}'.format(nsim=n, n=nind + 1, N=len(nsimArray))
    print(txt)

    # run simulations in parallel
    if n < 5:
        n_processes = int(n)
    else:
        n_processes = 5

    with multiprocessing.Pool(processes=n_processes) as pool:
        result = pool.map(get_corr_mat, np.arange(n))

    c = np.asarray(result)

    c_mean = c.mean(axis=0)
    c_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(c_mean, axes=0), axis=0), axes=0) * args.dt
    sArray[nind] = fe.entropy(c_fft, sample_spacing=args.dt)


fig, ax = plt.subplots()
ax.plot(nsimArray, sArray.real * args.dt, 'o')

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
