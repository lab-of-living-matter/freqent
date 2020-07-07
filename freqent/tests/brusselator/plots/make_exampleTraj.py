import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
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
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/reactionsOnly/191026/'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/reactionsOnly/191026/'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations'

datafile = os.path.join(datapath, 'alpha200.33680997479286_nSim50', 'data.hdf5')


with h5py.File(datafile) as d:
    edges_x = d['data']['edges_x'][:]
    edges_y = d['data']['edges_y'][:]
    prob_map = d['data']['prob_map'][:]
    t = d['data']['t_points'][:]
    traj = d['data']['trajs'][0, :]
    alpha = (d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
             (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))

prob = prob_map / prob_map.sum()

cmap_prob = mpl.cm.get_cmap('Blues')
normalize_prob = mpl.colors.Normalize(vmin=np.log(prob).min(), vmax=np.log(prob).max())

xx, yy = np.meshgrid(edges_x[:-1], edges_y[:-1])
dx, dy = [np.diff(edges_x)[0], np.diff(edges_y)[0]]
yrange = edges_y[-1] - edges_y[0]

inds = np.logical_and(t > 800, t < 806.5)

fig, ax = plt.subplots(figsize=(5, 5))
ax.pcolormesh(edges_x, edges_y, np.log(prob.T), rasterized=True, cmap='Blues')
ax.plot(traj[0, inds], traj[1, inds], 'k', alpha=0.7)
ax.plot(traj[0, np.where(inds)[0][-1]], traj[1, np.where(inds)[0][-1]],
        'o', markersize=15, markeredgecolor='k', color=(0.9, 0.9, 0.9))

cax2, _ = mpl.colorbar.make_axes(ax)
cbar2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap_prob, norm=normalize_prob)
cbar2.ax.set_title(r'$p(X, Y)$')

ax.set_aspect('equal')
ax.set(xlabel='X', ylabel='Y', title=r'$\mu = {m:0.2f}$'.format(m=np.log(alpha)))
ax.tick_params(which='both', direction='in')


# fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_traj_alpha{a}.pdf'.format(a=alpha)), format='pdf')

plt.show()
