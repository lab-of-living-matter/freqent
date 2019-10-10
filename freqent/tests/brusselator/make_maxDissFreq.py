import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import sys
import os
from glob import glob
from datetime import datetime

plt.close('all')
# mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1


if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/reactionsOnly/190904/'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations/'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/reactionsOnly/190904/'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations/'

files = glob(os.path.join(datapath, 'alpha*', 'data.hdf5'))

alphas = np.zeros(len(files))
wmax = np.zeros((len(files), 50))
epr_spectral = np.zeros((len(alphas), 50))

for ind, file in enumerate(files):
    with h5py.File(file) as d:
        alphas[ind] = (d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
                       (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))
        wmax[ind] = [d['data']['omega'][ind] for ind in np.argmax(d['data']['s_density'][:], axis=1)]
        epr_spectral[ind] = d['data']['s'][:]
        if ind == 0:
            wmax_abs = d['data']['omega'][:].max()

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.loglog(alphas, np.mean(abs(wmax), axis=1), 'ko')
ax.fill_between(np.sort(alphas),
                np.mean(abs(wmax)[np.argsort(alphas)], axis=1) + np.std(abs(wmax)[np.argsort(alphas)], axis=1),
                np.mean(abs(wmax)[np.argsort(alphas)], axis=1) - np.std(abs(wmax)[np.argsort(alphas)], axis=1),
                color='k', alpha=0.5)

ax.plot([alphas.min(), alphas.max()], [wmax_abs, wmax_abs], 'r--', lw=2)

ax.set(xlabel=r'$\alpha$', ylabel=r'arg max$_{\omega} \, \dot{S}$')
ax.tick_params(which='both', direction='in')
plt.tight_layout()

fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_argmax_eprDensity_vs_alpha.pdf'), format='pdf')

fig2, ax2 = plt.subplots(figsize=(6, 5))
a = ax2.scatter(np.mean(epr_spectral[np.argsort(alphas)], axis=1),
                np.mean(abs(wmax)[np.argsort(alphas)], axis=1),
                c=np.sort(alphas), cmap='cool')
ax2.errorbar(np.mean(epr_spectral[np.argsort(alphas)], axis=1),
             np.mean(abs(wmax)[np.argsort(alphas)], axis=1),
             yerr=np.std(abs(wmax)[np.argsort(alphas)], axis=1),
             xerr=np.std(epr_spectral[np.argsort(alphas)], axis=1),
             fmt='k.', markersize=0, alpha=0.5)

ax2.set(xlabel=r'$\hat{\dot{S}}$', ylabel=r'arg max$_{\omega} \, \dot{S}$',
        xscale='log', yscale='log')
ax2.tick_params(which='both', direction='in')

cbar = fig2.colorbar(a)
cbar.ax.set_title(r'$\alpha$')
plt.tight_layout()
fig2.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_argmax_eprDensity_vs_eprSpectral.pdf'), format='pdf')

plt.show()
