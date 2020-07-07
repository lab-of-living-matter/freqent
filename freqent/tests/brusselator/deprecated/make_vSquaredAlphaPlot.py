import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import sys
import os
from glob import glob
from datetime import datetime

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1


if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/reactionsOnly/191026/'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations/'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/reactionsOnly/191026/'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations/'

files = glob(os.path.join(datapath, 'alpha*', 'data.hdf5'))

alphas = np.zeros(len(files))
vsqs = np.zeros(len(files))
epr_blind = np.zeros(len(files))
epr = np.zeros(len(files))
epr_spectral = np.zeros((len(alphas), 50))

for ind, file in enumerate(files):
    with h5py.File(file) as d:
        alphas[ind] = (d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
                       (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))
        dx, dy = np.diff(d['data']['edges_x'])[0], np.diff(d['data']['edges_y'])[0]
        dudy, dudx = np.gradient(d['data']['flux_field'][0].T, 4)
        dvdy, dvdx = np.gradient(d['data']['flux_field'][1].T, 4)
        vsqs[ind] = np.sum(d['data']['flux_field'][0].T**2 + d['data']['flux_field'][1].T**2) / (dx * dy)
        epr[ind] = d['data']['epr'][()]
        epr_blind[ind] = d['data']['epr_blind'][()]
        epr_spectral[ind] = d['data']['s'][:]

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(np.log(alphas), epr, 'o', label=r'$\dot{S}_{true}$')
ax.plot(np.log(alphas), epr_blind, 'o', label=r'$\dot{S}_{blind}$')
ax.plot(np.log(alphas), np.mean(epr_spectral, axis=1), 'ko', label=r'$\hat{\dot{S}}$')
ax.fill_between(np.log(alphas)[np.argsort(alphas)],
                np.mean(epr_spectral, axis=1)[np.argsort(alphas)] + np.std(epr_spectral, axis=1)[np.argsort(alphas)],
                np.mean(epr_spectral, axis=1)[np.argsort(alphas)] - np.std(epr_spectral, axis=1)[np.argsort(alphas)],
                color='k', alpha=0.5)
ax.plot(np.log(alphas), vsqs, 'o', color='C3', label=r'$\dot{S}_{\int v^2}$', alpha=0.7)
ax.set(xlabel=r'$\Delta \mu$', ylabel=r'$\dot{S}$',
       yscale='log')
ax.tick_params(which='both', direction='in')
ax.legend(loc='lower right')
plt.tight_layout()

fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_vSquared_vs_alpha.pdf'), format='pdf')

plt.show()
