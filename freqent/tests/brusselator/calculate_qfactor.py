import os
from glob import glob
import numpy as np
import h5py
import argparse
from scipy import signal
import matplotlib.pyplot as plt
import freqent.freqent as fe

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str,
                    help='parent directory of data')
args = parser.parse_args()

fig, ax = plt.subplots(2, 2, sharex=True)
# volFolders = ['v10', 'v50', 'v100', 'v500', 'v1000', 'v5000', 'v10000']
volFolders = ['v10', 'v100', 'v1000', 'v10000']

for vInd, v in enumerate(volFolders):
    f = os.path.join(args.file, v)
    muFolders = glob(os.path.join(f, 'mu*'))
    qs = np.zeros(len(muFolders))
    mus = np.zeros(len(muFolders))
    for muInd, m in enumerate(muFolders):
        print(m.split(os.path.sep)[-2:])
        with h5py.File(os.path.join(m, 'data.hdf5')) as d:
            t_points = d['data']['t_points'][:]
            t_epr = np.where(t_points > 100)[0]
            dt = np.diff(t_points)[0]
            s, epf, w = fe.entropy(d['data']['trajs'][..., t_epr], sample_spacing=dt, smooth_corr=False, return_density=True)
            # epf = d['data']['s_density'][:].mean(axis=0)
            # w = d['data']['omega'][:]
            mu = float(m.split(os.path.sep)[-1][2:-7])
        peak_ind = np.argmax(epf[w > 0])
        peak = np.max(epf[w > 0])
        fwhm = signal.peak_widths(epf[w > 0], peaks=np.array([peak_ind]), rel_height=0.5)
        q_factor = w[w > 0][peak_ind] / (np.diff(w)[0] * fwhm[0])
        ax[0, 0].plot(mu, w[w > 0][peak_ind], 'ko', alpha=(vInd + 1) / len(volFolders))
        ax[0, 1].plot(mu, np.diff(w)[0] * fwhm[0], 'ko', alpha=(vInd + 1) / len(volFolders))
        ax[1, 0].plot(mu, q_factor, 'ko', alpha=(vInd + 1) / len(volFolders))
        ax[1, 1].plot(mu, peak / q_factor, 'ko', alpha=(vInd + 1) / len(volFolders))

ax[0, 0].set(title=r'$\omega_\mathrm{peak}$')
ax[0, 1].set(title=r'$\Delta \omega_\mathrm{FWHM}$')
ax[1, 0].set(title=r'$Q = \omega_\mathrm{peak} / \Delta \omega_\mathrm{FWHM}$', xlabel=r'$\Delta \mu$')
ax[1, 1].set(title=r'$\mathrm{max}\left(\mathcal{E} / Q \right)$', xlabel=r'$\Delta \mu$')
plt.show()
