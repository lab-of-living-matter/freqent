import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import os
import sys
from datetime import datetime

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1

fig, ax = plt.subplots(figsize=(6, 5))

if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/freqent/spinOsc/190709/'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/spinOsc/'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/freqent/spinOsc/190709/'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/spinOsc/'

alpha = 2  # pick which value of alpha to plot with
sdot_array = []
ndim_array = []
sdot_thry = 2 * alpha**2
for file in os.listdir(datapath):
    if file.endswith('.hdf5'):
        with h5py.File(os.path.join(datapath, file), 'r') as f:
            if f['params']['alpha'][()] == alpha:
                ndim_array.append(f['params']['ndim'][()])
                sdot_array.append(f['data']['sdot_array'][:])

                t_epr = f['params']['t_epr'][()]
                sigma = f['params']['sigma_array'][:]
                n_epr = f['params']['n_epr'][:]

cmap = mpl.cm.get_cmap('viridis')
normalize = mpl.colors.Normalize(vmin=min(sigma), vmax=max(sigma))
colors = [cmap(normalize(s)) for s in sigma]

ndim_inds = np.argsort(ndim_array)  # get indices of number of dimensions in order
xscale_array = [0.9, 1, 1.1]  # scale x-axis for plotting distinguishability

for dimInd, ind in enumerate(ndim_inds):
    sdot = sdot_array[ind]
    ndim = ndim_array[ind]
    for nInd, n in enumerate(n_epr):
        for sInd, s in enumerate(sigma):
            ax.semilogx(n * t_epr * xscale_array[dimInd], sdot[nInd, sInd],
                        marker=(ndim, 0, 45),
                        color=colors[sInd],
                        markersize=10)

ax.plot([n_epr[0] * t_epr, n_epr[-1] * t_epr], [2 * alpha**2, 2 * alpha**2], 'k--', lw=2)

handles = [mpl.lines.Line2D([0], [0], color='k', linestyle='', marker=(2, 0, 45), markersize=10, label='2D'),
           mpl.lines.Line2D([0], [0], color='k', linestyle='', marker=(3, 0, 45), markersize=10, label='3D'),
           mpl.lines.Line2D([0], [0], color='k', linestyle='', marker=(4, 0, 45), markersize=10, label='4D'),
           mpl.lines.Line2D([0], [0], color='k', linestyle='--', label=r'$\dot{S}_{thry} = 8$')]

cax, _ = mpl.colorbar.make_axes(ax)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
cbar.ax.set_title(r'$\sigma$')


ax.legend(handles=handles, loc='best')
ax.set(xlabel=r'$N_{traj} T$', ylabel=r'$\dot{\hat{S}}$',
       xticks=[500, 1000, 5000],
       xticklabels=[r'$5 \times 10^2$', r'$10^3$', r'$5 \times 10^3$'])
ax.tick_params(axis='both', which='both', direction='in')

fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_alpha{a}_epr_vs_dataSize.pdf'.format(a=alpha)), format='pdf')

plt.show()
