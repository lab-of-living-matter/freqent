import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import os
import sys
import freqent.freqent as fe
from datetime import datetime

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2

if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/reactionsOnly/191026'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/reactionsOnly/191026'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations'

fig, ax = plt.subplots(figsize=(5.5, 5))
files = ['alpha33.11545195869248_nSim50',
         'alpha200.33680997479286_nSim50',
         'alpha492.7490410932593_nSim50']

cmap = mpl.cm.get_cmap('Dark2')
normalize = mpl.colors.Normalize(vmin=0, vmax=10)
colors = [cmap(normalize(value)) for value in np.arange(1, 10)]

fullfilepath = [os.path.join(datapath, file, 'data.hdf5') for file in files]
sigma = 20
t_factor = 10

for fInd, file in enumerate(fullfilepath):
    with h5py.File(file, 'r') as d:
        t_points = d['data']['t_points'][:]
        t_epr = np.where(t_points > 10)[0]  # only calculate epr after t = 1000
        dt = np.diff(t_points)[0]
        nSim = d['params']['nSim'][()]
        mu = np.log((d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
                    (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5])))
        s = np.zeros(nSim)
        rhos = np.zeros((nSim, len(t_epr[::t_factor])))

        for ind, traj in enumerate(d['data']['trajs'][..., t_epr[::t_factor]]):
            s[ind], rhos[ind], w = fe.entropy(traj, sample_spacing=dt * t_factor,
                                              sigma=sigma, return_density=True)

        ax.plot(w[w > 0], rhos.mean(axis=0)[w > 0], '.', label=r'$\Delta \mu = {m:0.1f}$'.format(m=mu), color=colors[fInd])
        ax.fill_between(w[w > 0],
                        rhos.mean(axis=0)[w > 0] - np.std(rhos, axis=0)[w > 0],
                        rhos.mean(axis=0)[w > 0] + np.std(rhos, axis=0)[w > 0],
                        color=colors[fInd], alpha=0.5)


ax.set(xlabel=r'$\omega$', ylabel=r'$\mathcal{E}$',
       yscale='linear', xscale='log', ylim=[-0.01, 0.21], xlim=[5e-3, 20])
ax.tick_params(which='both', direction='in')
ax.legend()

# fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_epfPlot.pdf'),
#             format='pdf')

plt.show()
