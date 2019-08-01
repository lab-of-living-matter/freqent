import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import os
from datetime import datetime
from itertools import product

# plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(figsize=(8, 6))
parentDir = '/mnt/llmStorage203/Danny/freqent/spinOsc/190709/'

alpha = 2  # pick which value of alpha to plot with
sdot_array = []
ndim_array = []
sdot_thry = 2 * alpha**2
for file in os.listdir(parentDir):
    if file.endswith('.hdf5'):
        with h5py.File(os.path.join(parentDir, file), 'r') as f:
            if f['params']['alpha'][()] == alpha:
                ndim_array.append(f['params']['ndim'][()])
                sdot_array.append(f['data']['sdot_array'][:])

                t_epr = f['params']['t_epr'][()]
                sigma = f['params']['sigma'][:]
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

ax.plot([n_epr[0] * t_epr, n_epr[-1] * t_epr], [2 * alpha**2, 2 * alpha**2], 'k--')

handles = [mpl.lines.Line2D([0], [0], color='k', linestyle='', marker=(2, 0, 45), markersize=10, label='2D'),
           mpl.lines.Line2D([0], [0], color='k', linestyle='', marker=(3, 0, 45), markersize=10, label='3D'),
           mpl.lines.Line2D([0], [0], color='k', linestyle='', marker=(4, 0, 45), markersize=10, label='4D'),
           mpl.lines.Line2D([0], [0], color='k', linestyle='--', label=r'$2 \alpha^2/k$')]

cax, _ = mpl.colorbar.make_axes(ax)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
cbar.ax.set_title(r'$\sigma$')


ax.legend(handles=handles, loc='best')
ax.set(xlabel=r'$N_{traj} T$', ylabel=r'$\dot{\hat{S}}$',
       xticks=[500, 1000, 5000],
       xticklabels=[r'$5 \times 10^2$', r'$10^3$', r'$5 \times 10^3$'])
ax.tick_params(axis='both', which='both', direction='in')

fig2, ax2 = plt.subplots(figsize=(7, 6))
alphas = range(0, 11)

for a in alphas:
    sdot_array = []
    ndim_array = []
    sdot_thry = 2 * a**2
    for file in os.listdir(parentDir):
        if file.endswith('.hdf5'):
            with h5py.File(os.path.join(parentDir, file), 'r') as f:
                if f['params']['alpha'][()] == a:
                    ndim_array.append(f['params']['ndim'][()])
                    sdot_array.append(f['data']['sdot_array'][:])

                    t_epr = f['params']['t_epr'][()]
                    sigma = f['params']['sigma'][:]
                    n_epr = f['params']['n_epr'][:]

    cmap = mpl.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min(sigma), vmax=max(sigma))
    colors = [cmap(normalize(s)) for s in sigma]

    ndim_inds = np.argsort(ndim_array)  # get indices of number of dimensions in order
    xscale_array = [0.9, 1, 1.1]  # scale x-axis for plotting distinguishability

    for dimInd, ind in enumerate(ndim_inds):
        sdot = sdot_array[ind]
        ndim = ndim_array[ind]
        ax2.plot(a, sdot[-1, 0],
                 marker=(ndim, 0, 45),
                 color='k', alpha=0.5,
                 markersize=10)


ax2.plot(np.linspace(-0.1, 10.1, 100), 2 * np.linspace(-0.1, 10.1, 100)**2, 'r')
handles = [mpl.lines.Line2D([0], [0], color='k', alpha=0.5, linestyle='', marker=(2, 0, 45), markersize=10, label='2D'),
           mpl.lines.Line2D([0], [0], color='k', alpha=0.5, linestyle='', marker=(3, 0, 45), markersize=10, label='3D'),
           mpl.lines.Line2D([0], [0], color='k', alpha=0.5, linestyle='', marker=(4, 0, 45), markersize=10, label='4D'),
           mpl.lines.Line2D([0], [0], color='r', linestyle='-', label=r'$2 \alpha^2/k$')]

# cax, _ = mpl.colorbar.make_axes(ax2)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
# cbar.ax.set_title(r'$\sigma$')


ax2.legend(handles=handles, loc='lower right')
ax2.set(xlabel=r'$\alpha$', ylabel=r'$\dot{\hat{S}}$')
ax2.tick_params(axis='both', direction='in')

# fig.savefig(os.path.join(parentDir, 'alpha{a}_epr_vs_dataSize.pdf'.format(a=alpha)), format='pdf')
# fig2.savefig(os.path.join(parentDir, 'epr_vs_alpha_mostData_leastSmoothing.pdf'), format='pdf')

plt.show()
