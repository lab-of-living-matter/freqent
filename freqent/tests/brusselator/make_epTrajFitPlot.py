import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from datetime import datetime
from scipy import stats

# plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2

if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/reactionsOnly/191026/'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations/'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/reactionsOnly/191026/'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations/'

file = 'alpha90.01713130052228_nSim50'

with h5py.File(os.path.join(datapath, file, 'data.hdf5')) as d:
    ep_blind_traj = d['data']['ep_blinds'][:, ::10]
    epr_blind = d['data']['epr_blind'][()]
    t_points = d['data']['t_points'][::10]
    mu = np.log((d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
                (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5])))

r = stats.linregress(t_points[len(t_points) // 2:],
                     ep_blind_traj[:, len(t_points) // 2:].mean(axis=0))

fig, ax = plt.subplots()
ax.plot(t_points, ep_blind_traj.T, 'k', alpha=0.1, lw=0.5)
ax.plot(t_points, ep_blind_traj.mean(axis=0), 'k')
ax.plot(t_points[len(t_points) // 2:],
        t_points[len(t_points) // 2:] * r.slope + r.intercept,
        'r--', lw=2)

ax.tick_params(which='both', direction='in')
ax.set(xlabel='t', ylabel=r'$\Delta S_{blind}$', title=r'$\Delta \mu = {m:0.2f}$'.format(m=mu))
ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])

legend_handles = [mpl.lines.Line2D([0], [0], color='k', alpha=0.1, lw=0.5, label='trajectories'),
                  mpl.lines.Line2D([0], [0], color='k', label='mean'),
                  mpl.lines.Line2D([0], [0], linestyle='--', color='r', lw=2, label='linear fit')]
ax.legend(handles=legend_handles, loc='best')
fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_epBlind_fit_mu{m:0.2f}.pdf'.format(m=mu)), format='pdf')

plt.show()
