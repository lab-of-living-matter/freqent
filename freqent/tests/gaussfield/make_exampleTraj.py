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
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2

data = 'alpha7.5_nSim10'

if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/freqent/gaussfield'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/gaussfield'
if sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/freqent/gaussfield'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/gaussfield'

with h5py.File(os.path.join(datapath, data, 'data.hdf5')) as d:
    traj = d['data']['trajs'][2, :, ::100, :]
    L = d['data']['L'][:]
    t = d['data']['t'][::100]

cmap = mpl.cm.get_cmap('RdBu_r')
vmax = abs(traj[:, t > 10]).max()
vmin = -vmax
normalize = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# colors = [cmap(normalize(value)) for value in np.ravel(traj)]

fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].pcolormesh(L, t[np.logical_and(t > 10, t < 30)], traj[0, np.logical_and(t > 10, t < 30)],
                 cmap='RdBu_r', vmin=vmin, vmax=vmax, rasterized=True)
ax[1].pcolormesh(L, t[np.logical_and(t > 10, t < 30)], traj[1, np.logical_and(t > 10, t < 30)],
                 cmap='RdBu_r', vmin=vmin, vmax=vmax, rasterized=True)

ax[0].set(xlabel=r'$x \ [1/\sqrt{r}]$', title=r'$\phi(x, t)$', ylabel=r'$t \ [1/Dr]$', yticks=[10, 15, 20, 25, 30])
ax[0].tick_params(which='both', direction='in')
ax[1].set(xlabel=r'$x \ [1/\sqrt{r}]$', title=r'$\psi(x, t)$', yticks=[10, 15, 20, 25, 30])
ax[1].tick_params(which='both', direction='in')

cax, _ = mpl.colorbar.make_axes(ax)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
cbar.ax.tick_params(which='both', direction='in')
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_' + data + '_traj.pdf'), format='pdf')

plt.show()
