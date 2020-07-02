import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


parser = argparse.ArgumentParser()
parser.add_argument('--muRange', '-mus', type=float, nargs=2, default=[6, 7],
                    help='[min, max] of Delta mu to use')
parser.add_argument('--nSteps', '-nmu', type=int, default=100,
                    help='number of steps in Delta mu between min and max')
parser.add_argument('--tFinal', '-t', type=float, default=300,
                    help='final time to integrate ode')
parser.add_argument('--tSteps', '-nt', type=int, default=10001,
                    help='number of time steps')
parser.add_argument('--nRuns', '-nr', type=int, default=1,
                    help='number of runs at each Delta mu to run')
args = parser.parse_args()


mus = np.linspace(args.muRange[0], args.muRange[1], args.nSteps)
t = np.linspace(0, args.tFinal, args.tSteps)
k1plus, k1minus, k2plus, k2minus, k3plus, k3minus = [1, 0.5, 2, 0.5, 2, 0.5]
a = 1
b = np.exp(mus / 2) / (k2plus * k3plus)
c = np.exp(-mus / 2) / (k2minus * k3minus)
xss = a * k1plus / k1minus
yss = (k2plus * b * xss + k3minus * xss**3) / (k2minus * c + k3plus * xss**2)
x_means = np.zeros((len(mus), args.nRuns))
y_means = np.zeros((len(mus), args.nRuns))


def brusselator(r, t, a, b, c, rates):
    '''
    r = [x, y]
    '''
    x, y = r
    k1plus, k1minus, k2plus, k2minus, k3plus, k3minus = rates
    drdt = [k1plus * a - k1minus * x + k2minus * y * c - k2plus * b * x + k3plus * x**2 * y - k3minus * x**3,
            -k2minus * y * c + k2plus * b * x - k3plus * x**2 * y + k3minus * x**3]
    return drdt


for mu_ind, mu in enumerate(mus):
    for run_ind, r0 in enumerate(np.array([xss, yss[mu_ind]]) + np.random.randn(10, 2) * 0.05):
        # numerically solve the differntial equations
        sol = odeint(brusselator, r0, t, args=(a, b, c, [k1plus, k1minus, k2plus, k2minus, k3plus, k3minus]))
        x_means[mu_ind, run_ind], y_means[mu_ind, run_ind] = sol[len(t) // 2:, :].mean(axis=0)
