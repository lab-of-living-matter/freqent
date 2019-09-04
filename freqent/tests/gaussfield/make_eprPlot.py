import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

parentFolder = '/mnt/llmStorage203/Danny/freqent/gaussfield'
folders = glob(os.path.join(parentFolder, 'alpha*'))
alphas = np.asarray([float(a.split(os.path.sep)[-1].split('_')[0][5:]) for a in folders])

eprs = np.zeros((len(alphas), 10))
for fInd, f in enumerate(folders):
    with h5py.File(os.path.join(f, 'data.hdf5'), 'r') as d:
        eprs[fInd] = d['data']['epr'][()]

fig, ax = plt.subplots()
ax.errorbar(alphas, np.nanmean(eprs, axis=1), yerr=np.nanstd(eprs, axis=1), color='k', fmt='o', label='data')
ax.plot(np.sort(alphas), np.sort(alphas)**2, 'r-', label=r'$\alpha^2$')
ax.set(xlabel=r'$\alpha$', ylabel=r'$\dot{\Sigma}$')
ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
plt.legend()

plt.tight_layout()

# fig.savefig(os.path.join(parentFolder, 'eprPlot.pdf'), format='pdf')
plt.show()
