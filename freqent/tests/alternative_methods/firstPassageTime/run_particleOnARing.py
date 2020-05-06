import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from particleOnARing import *

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

dt = 1e-5
x0 = 0
v = 1
D = 0.008
nsteps = 5e4
nsims = 50
alphas = np.logspace(-5, -1, 10)
epr = v**2 / D


def L(a):
    '''
    Threshold for making decision on arrow of time with error a
    See Roldan, et al, PRL 115, 250602 (2015)
    '''
    return np.log((1 - a) / a)


p = particleOnARing(x0=x0, dt=dt, nsteps=nsteps)

fig, ax = plt.subplots(1, 2)

for ind, a in enumerate(alphas):
    first_crossing_times = 0
    for ii in range(nsims):
        p.runSimulation(v=v, D=D)
        log_prob_ratio = (v / D) * (p.pos - p.pos[0])
        crossing_inds = np.logical_or(log_prob_ratio > L(a), log_prob_ratio < -L(a))
        first_crossing_times += p.t[crossing_inds][0]

        if ind == len(alphas) - 1:
            ax[0].plot(p.t, log_prob_ratio, 'k-', alpha=0.5, lw=0.5)

    mean_fpt = first_crossing_times / nsims
    epr_estimate = L(a) * (1 - 2 * a) / mean_fpt
    ax[1].semilogx(a, epr_estimate / epr, 'ko')


ax[0].plot([0, p.t[-1]], [L(alphas[0]), L(alphas[0])], 'r--')
ax[0].plot([0, p.t[-1]], [-L(alphas[0]), -L(alphas[0])], 'r--')

ax[0].set(xlabel='t (s)', ylabel=r'$\mathcal{L}$')
ax[0].text(0, -2 * L(alphas[0]), r'$L(\alpha)$', color='red')
ax[1].set(xlabel=r'$\alpha$', ylabel=r'$\langle \tilde{\dot{S}} \rangle / \dot{S}$', ylim=[0, 1.5])

plt.tight_layout()

plt.show()
