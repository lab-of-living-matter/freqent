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
    datapath = '/mnt/llmStorage203/Danny/freqent/gaussfield'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/gaussfield/'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/freqent/gaussfield'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/gaussfield/'

folders = glob(os.path.join(datapath, 'alpha*'))
alphas = np.asarray([float(a.split(os.path.sep)[-1].split('_')[0][5:]) for a in folders])

eprs = np.zeros((len(alphas), 10))
for fInd, f in enumerate(folders):
    with h5py.File(os.path.join(f, 'data.hdf5'), 'r') as d:
        eprs[fInd] = d['data']['epr'][()]

        if fInd == 0:
            T = d['data']['t'][:].max()
            L = d['data']['L'][:].max()
            dw = np.diff(d['data']['omega'][:])[0]
            dk = np.diff(d['data']['k'][:])[0]
            k_max = d['data']['k'][:].max()
            w_max = d['data']['omega'][:].max()
            sigma = d['params']['sigma'][()]

nrep = 1
nvar = 2
sigma_w = 5 * dw
sigma_k = 5 * dk
bias = (1 / nrep) * (nvar * (nvar - 1) / 2) * (w_max / (T * sigma_w * np.sqrt(np.pi))) * (k_max / (L * sigma_k * np.sqrt(np.pi)))

fig, ax = plt.subplots(figsize=(5, 5))
#ax.errorbar(alphas, np.nanmean(eprs, axis=1), yerr=np.nanstd(eprs, axis=1), color='k', fmt='o', label='data', capsize=5, alpha=0.5)

inds = [np.where(np.sort(alphas) == n)[0][0] for n in range(26)]
ax.plot(np.sort(alphas)[inds], np.nanmean(eprs, axis=1)[np.argsort(alphas)][inds] - bias, 'ko', label=r'$\dot{S}_{spectral}$')
ax.fill_between(np.sort(alphas)[inds],
                np.nanmean(eprs, axis=1)[np.argsort(alphas)][inds] - bias + np.nanstd(eprs, axis=1)[np.argsort(alphas)][inds],
                np.nanmean(eprs, axis=1)[np.argsort(alphas)][inds] - bias - np.nanstd(eprs, axis=1)[np.argsort(alphas)][inds],
                color='k', alpha=0.5)
ax.plot(np.sort(alphas), np.sort(alphas)**2, 'r', lw=3, label=r'$\dot{S}_{thry} = \alpha^2$')
ax.set(xlabel=r'$\alpha$', ylabel=r'$\dot{S}$')
ax.tick_params(which='both', direction='in')
ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
plt.legend()

plt.tight_layout()

fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_eprPlot_shaded.pdf'), format='pdf')
plt.show()
