import numpy as np
import os
from datetime import datetime
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
parser.add_argument('--savepath', '-save', default=None,
                    help='absolute path to save plot')
args = parser.parse_args()

args = parser.parse_args()


mu_array = np.linspace(args.muRange[0], args.muRange[1], args.nSteps)
t = np.linspace(0, args.tFinal, args.tSteps)
k1plus, k1minus, k2plus, k2minus, k3plus, k3minus = [1, 0.5, 2, 0.5, 2, 0.5]
a = 1
b_array = np.exp(mu_array / 2) / (k2plus * k3plus)
c_array = np.exp(-mu_array / 2) / (k2minus * k3minus)
xss = a * k1plus / k1minus
yss_array = (k2plus * b_array * xss + k3minus * xss**3) / (k2minus * c_array + k3plus * xss**2)
x_means = np.zeros((len(mu_array), args.nRuns))
y_means = np.zeros((len(mu_array), args.nRuns))


def brusselator(r, t, a, b, c, rates):
    '''
    r = [x, y]
    '''
    x, y = r
    k1plus, k1minus, k2plus, k2minus, k3plus, k3minus = rates
    drdt = [k1plus * a - k1minus * x + k2minus * y * c - k2plus * b * x + k3plus * x**2 * y - k3minus * x**3,
            -k2minus * y * c + k2plus * b * x - k3plus * x**2 * y + k3minus * x**3]
    return drdt


for mu_ind, (mu, b, c, yss) in enumerate(zip(mu_array, b_array, c_array, yss_array)):
    for run_ind, r0 in enumerate(np.array([xss, yss]) + np.random.randn(args.nRuns, 2) * 0.05):
        # numerically solve the differntial equations
        sol = odeint(brusselator, r0, t, args=(a, b, c, [k1plus, k1minus, k2plus, k2minus, k3plus, k3minus]))
        x_means[mu_ind, run_ind], y_means[mu_ind, run_ind] = sol[len(t) // 2:, :].mean(axis=0)


j_forward = b_array * x_means.mean(axis=1) * k2plus
j_reverse = c_array * y_means.mean(axis=1) * k2minus

j_forward_blind = b_array * x_means.mean(axis=1) * k2plus + x_means.mean(axis=1)**3 * k3minus
j_reverse_blind = c_array * y_means.mean(axis=1) * k2minus + x_means.mean(axis=1)**2 * y_means.mean(axis=1) * k3plus

fig, ax = plt.subplots(figsize=(4, 4))
ax.semilogy(mu_array, j_forward, '.', color='C2', label=r'$J^\mathrm{F}$')
ax.semilogy(mu_array, j_reverse, '.', color='C3', label=r'$J^\mathrm{R}$')
ax.plot([6.16, 6.16], [j_reverse.min(), j_forward.max()], 'k--')
ax.legend(loc='center right')
ax.set(xlabel=r'$\Delta \mu$', ylabel='flux')
# ax[0].set_aspect((ax[0].set_xlim()[1] / ax[0].set_xlim()[0]) / np.diff(ax[0].set_ylim())[0])

ax1 = inset_axes(ax, width="75%", height="75%",
                 bbox_to_anchor=(0.1, 0.25, 0.7, 0.7), bbox_transform=ax.transAxes,
                 loc='center left', borderpad=2)

ax1.plot(mu_array, j_forward_blind, '.', color='C2', label=r'$J^\mathrm{F}_\mathrm{blind}$')
ax1.plot(mu_array, j_reverse_blind, '.', color='C3', label=r'$J^\mathrm{R}_\mathrm{blind}$')
ax1.plot([6.16, 6.16], [j_reverse_blind.min(), j_reverse_blind.max()], 'k--')
ax1.legend(loc='center right')
ax1.set(xlabel=r'$\Delta \mu$', ylabel='flux')
ax1.set_aspect(np.diff(ax1.set_xlim())[0] / np.diff(ax1.set_ylim())[0])
plt.tight_layout()

if args.savepath:
    today = datetime.now().strftime('%y%m%d')
    fig.savefig(os.path.join(args.savepath, today + '_forwardReverseFlux.pdf'), format='pdf')

plt.show()
