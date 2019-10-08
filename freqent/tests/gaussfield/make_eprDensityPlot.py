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
    epr_density = np.mean(d['data']['epr_density'][:], axis=0)
    k = d['data']['k'][:]
    w = d['data']['omega'][:]
    alpha = d['params']['alpha'][()]

fig, ax = plt.subplots(figsize=(6.5, 5))
a = ax.pcolormesh(k, w, epr_density,
                  cmap='Reds', rasterized=True)
ax.contour(k, w, epr_density, levels=[0.005, 0.01, 0.02, 0.03, 0.04], cmap='Greys_r')


ax.set(xlabel=r'$k$', title=r'$\alpha = {a}$'.format(a=alpha), ylabel=r'$\omega$',
       ylim=[-10 * np.pi, 10 * np.pi])
ax.tick_params(which='both', direction='in')
ax.set_aspect('equal')
plt.tight_layout()

cbar = fig.colorbar(a)
cbar.ax.tick_params(which='both', direction='in')
cbar.ax.set(title=r'$\rho_{\dot{s}}$')

fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_' + data + '_traj.pdf'), format='pdf')

plt.show()
