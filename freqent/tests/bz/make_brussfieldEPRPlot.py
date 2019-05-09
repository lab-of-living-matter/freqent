import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

parentFolder = '/mnt/llmStorage203/Danny/freqent/brusselatorSims/fieldSims/190429/brussfield'
folders = glob(os.path.join(parentFolder, 'alpha*'))
alphas = np.asarray([float(a.split(os.path.sep)[-1].split('_')[0][5:]) for a in folders])

epr = np.zeros(len(alphas))
epr_blind = np.zeros(len(alphas))
epr_spectral = np.zeros(len(alphas))

for fInd, f in enumerate(folders):
    with h5py.File(os.path.join(f, 'data.hdf5'), 'r') as d:
        epr[fInd] = d['data']['epr'][()]
        epr_blind[fInd] = d['data']['epr_blind'][()]
        epr_spectral[fInd] = d['data']['epr_spectral'][()]

fig, ax = plt.subplots()
ax.plot(alphas, epr, 'o', label='epr')
ax.plot(alphas, epr_blind, 'o', label='epr_blind')
ax.plot(alphas, epr_spectral, 'o', label='epr_spectral')

ax.set(xlabel=r'$\alpha$', ylabel=r'$\dot{\Sigma}$')
ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
plt.legend()

plt.tight_layout()

# fig.savefig(os.path.join(parentFolder, 'eprPlot.pdf'), format='pdf')
plt.show()
