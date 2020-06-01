import os
from glob import glob
import numpy as np
import h5py
import argparse
from scipy import signal
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str,
                    help='parent directory of data')
args = parser.parse_args()

fig, ax = plt.subplots(1, 3)
volFolders = ['v10', 'v50', 'v100', 'v500', 'v1000', 'v5000', 'v10000']

for vInd, v in enumerate(volFolders):
    f = os.path.join(args.file, v)
    muFolders = glob(os.path.join(f, 'mu*'))
    for m in muFolders:
        with h5py.File(os.path.join(m, 'data.hdf5')) as d:
            epf = d['data']['s_density'][:].mean(axis=0)
            w = d['data']['omega'][:]
            mu = float(m.split(os.path.sep)[-1][2:-7])
        peak_ind = np.argmax(epf[w > 0])
        fwhm = signal.peak_widths(epf[w > 0], peaks=np.array([peak_ind]), rel_height=0.5)
        q_factor = w[w > 0][peak_ind] / (np.diff(w)[0] * fwhm[0])
        ax[0].plot(mu, w[w > 0][peak_ind], 'ko', alpha=(vInd + 1) / len(volFolders))
        ax[1].plot(mu, np.diff(w)[0] * fwhm[0], 'ko', alpha=(vInd + 1) / len(volFolders))
        ax[2].plot(mu, q_factor, 'ko', alpha=(vInd + 1) / len(volFolders))

ax[0].set(title=r'$\omega_\mathrm{peak}$')
ax[1].set(title=r'$\Delta \omega_\mathrm{FWHM}$')
ax[2].set(title=r'$Q = \omega_\mathrm{peak} / \Delta \omega_\mathrm{FWHM}$')
plt.show()
