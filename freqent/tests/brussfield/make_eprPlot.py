import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datetime import datetime
import matplotlib as mpl

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2

if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/fieldSims/191028/'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brussfield/'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/fieldSims/191028/'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/brussfield/'

folders = glob(os.path.join(datapath, 'alpha*'))
alphas = np.asarray([float(a.split(os.path.sep)[-1].split('_')[0][5:]) for a in folders])

epr_spectral = np.zeros((len(alphas), 10))
epr_blind = np.zeros(len(alphas))
epr = np.zeros(len(alphas))

for fInd, f in enumerate(folders):
    with h5py.File(os.path.join(f, 'data.hdf5'), 'r') as d:
        epr[fInd] = d['data']['epr'][()]
        epr_blind[fInd] = d['data']['epr_blind'][()]
        epr_spectral[fInd] = d['data']['s'][:]

        if fInd == 0:
            lCompartment = d['params']['lCompartment'][()]
            nCompartments = d['params']['nCompartments'][()]
            T = d['data']['t_points'][:].max()
            dw = np.diff(d['data']['omega'][:])[0]
            dk = np.diff(d['data']['k'][:])[0]
            k_max = d['data']['k'][:].max()
            w_max = d['data']['omega'][:].max()
            sigma = d['params']['sigma'][:]

V = lCompartment * nCompartments
mu = np.log(alphas)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(mu[mu <= 8], epr[mu <= 8] / V, 'o', label=r'$\dot{S}_{true}$')
ax.plot(mu[mu <= 8], epr_blind[mu <= 8] / V, 'o', label=r'$\dot{S}_{blind}$')
# ax.errorbar(alphas, np.mean(epr_spectral, axis=1) - bias, yerr=np.std(epr_spectral, axis=1), fmt='ko', label='epr_spectral', capsize=5)

ax.plot(mu[mu <= 8], np.mean(epr_spectral[mu <= 8], axis=1), 'ko', label=r'$\hat{\dot{S}}$')
ax.fill_between(mu[[mu <= 8]][np.argsort(mu[mu <= 8])],
                np.mean(epr_spectral, axis=1)[mu <= 8][np.argsort(mu[mu <= 8])] + np.std(epr_spectral[mu <= 8], axis=1)[np.argsort(mu[mu <= 8])],
                np.mean(epr_spectral[mu <= 8], axis=1)[np.argsort(mu[mu <= 8])] - np.std(epr_spectral[mu <= 8], axis=1)[np.argsort(mu[mu <= 8])],
                color='k', alpha=0.5)
# ax.plot(alphas[np.argsort(alphas)] , epr_blind[np.argsort(alphas)] / V, 'r', lw=3, label=r'$\dot{S}_{thry}$')

# ax.plot(np.log(np.repeat(np.sort(alphas), 10)), np.ravel(epr_spectral[np.argsort(alphas), :]), 'k.')

ax.set(xlabel=r'$\Delta \mu$', ylabel=r'$\dot{S}$')
# ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
ax.tick_params(which='both', direction='in')
ax.set(yscale='log')
ax.legend(loc='lower right')
plt.tight_layout()

# fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_eprPlot.pdf'), format='pdf')
plt.show()
