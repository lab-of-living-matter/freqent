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

cmap = mpl.cm.get_cmap('coolwarm')
tinds = np.logical_and(t > 20, t < 30)
vmax = abs(traj[:, tinds]).max()
vmin = -vmax
normalize = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
# colors = [cmap(normalize(value)) for value in np.ravel(traj)]

fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].pcolormesh(L, t[tinds], traj[0, tinds],
                 cmap='coolwarm', vmin=vmin, vmax=vmax, rasterized=True)
ax[1].pcolormesh(L, t[tinds], traj[1, tinds],
                 cmap='coolwarm', vmin=vmin, vmax=vmax, rasterized=True)

ax[0].set(xlabel=r'$x \ [1/\sqrt{r}]$', title=r'$\phi(x, t)$', ylabel=r'$t \ [1/Dr]$', yticks=[20, 25, 30])
ax[0].tick_params(which='both', direction='in')
ax[1].set(xlabel=r'$x \ [1/\sqrt{r}]$', title=r'$\psi(x, t)$', yticks=[20, 25, 30])
ax[1].tick_params(which='both', direction='in')

cax, _ = mpl.colorbar.make_axes(ax)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
cbar.ax.tick_params(which='both', direction='in')
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_' + data + '_traj.pdf'), format='pdf')

fig_trace, ax_trace = plt.subplots()
ax_trace.plot(L, traj[0, 2550], label=r'$\psi$', lw=2, drawstyle='steps-mid')
ax_trace.plot(L, traj[1, 2550], label=r'$\phi$', lw=2, drawstyle='steps-mid')
ax_trace.tick_params(which='both', direction='in')
ax_trace.set(xlabel=r'$x \ [1/\sqrt{r}]$', ylim=[-1, 1])
ax_trace.set_aspect(np.diff(ax_trace.set_xlim())[0] / np.diff(ax_trace.set_ylim())[0])
ax_trace.legend()


plt.show()
