import os, sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if sys.platform == 'linux':
    parentFolder = '/mnt/llmStorage203/Danny/brusselatorSims/fieldSims/190905/'
elif sys.platform == 'darwin':
    parentFolder = '/Volumes/Storage/Danny/brusselatorSims/fieldSims/190905/'

folders = glob(os.path.join(parentFolder, 'alpha*'))
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

V = lCompartment * nCompartments

fig, ax = plt.subplots()
ax.plot(np.log(alphas), epr / V, '.', label='epr')
ax.plot(np.log(alphas), epr_blind / V, '.', label='epr_blind')
# ax.errorbar(alphas, np.mean(epr_spectral, axis=1), yerr=np.std(epr_spectral, axis=1), fmt='ko', label='epr_spectral')
ax.plot(np.log(np.repeat(np.sort(alphas), 10)), np.ravel(epr_spectral[np.argsort(alphas), :]), 'k.')

ax.set(xlabel=r'$\alpha$', ylabel=r'$\dot{\Sigma}$')
# ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
ax.set(yscale='log')
plt.legend()

plt.tight_layout()

# fig.savefig(os.path.join(parentFolder, 'eprPlot.pdf'), format='pdf')
plt.show()
