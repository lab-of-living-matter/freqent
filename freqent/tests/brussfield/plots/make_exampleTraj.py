import os
import sys
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import numpy as np
from datetime import datetime

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1

data = 'alpha330.29955990965095_nSim10'

if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/fieldSims/191028'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brussfield'
if sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/fieldSims/191028'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/brussfield'

with h5py.File(os.path.join(datapath, data, 'data.hdf5')) as d:
    traj = d['data']['trajs'][8]
    n_sites = d['params']['nCompartments'][()]
    t = d['data']['t_points'][:]
    mu = np.log(d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
                (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))

L = np.arange(n_sites)
inds = np.logical_and(t > 50, t < 100)

cmap_name = 'cividis'
cmap = mpl.cm.get_cmap(cmap_name)
vmax = abs(traj[:, inds]).max()
vmin = 0
normalize = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# colors = [cmap(normalize(value)) for value in np.ravel(traj)]


fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].pcolormesh(L, t[inds], traj[0, inds],
                 cmap=cmap_name, vmin=vmin, vmax=vmax, rasterized=True)
ax[1].pcolormesh(L, t[inds], traj[1, inds],
                 cmap=cmap_name, vmin=vmin, vmax=vmax, rasterized=True)

ax[0].set(xlabel=r'$r$', title=r'$X$', ylabel=r'$t \ [1/k^+_1]$',
          yticks=[50, 60, 70, 80, 90, 100])
ax[0].tick_params(which='both', direction='in')
ax[1].set(xlabel=r'$r $', title=r'$Y$', yticks=[50, 60, 70, 80, 90, 100])
ax[1].tick_params(which='both', direction='in')

cax, _ = mpl.colorbar.make_axes(ax)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
cbar.ax.tick_params(which='both', direction='in')
# cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_'  'mu{m:0.1f}_traj.pdf'.format(m=mu)), format='pdf')


# fig_trace, ax_trace = plt.subplots()
# ax_trace.plot(L, np.squeeze(traj[0, t == 80]), lw=2, label=r'$\psi$')
# ax_trace.plot(L, np.squeeze(traj[1, t == 80]), lw=2, label=r'$\phi$')

plt.show()
