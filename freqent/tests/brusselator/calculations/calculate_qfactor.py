import os
from glob import glob
import numpy as np
import h5py
import argparse
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import freqent.freqent as fe

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str,
                    help='parent directory of data')
args = parser.parse_args()

fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 7))
# volFolders = ['v10', 'v50', 'v100', 'v500', 'v1000', 'v5000', 'v10000']
volFolders = ['v50', 'v500', 'v5000']

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
            s, epf, w = fe.entropy(d['data']['trajs'][..., t_epr], sample_spacing=dt,
                                   sigma=1, return_epf=True)
            # epf = d['data']['s_density'][:].mean(axis=0)
            # w = d['data']['omega'][:]
            mu = float(m.split(os.path.sep)[-1][2:-7])
            V = d['params']['V'][()]

        peak_ind = np.argmax(epf[w > 0])
        peak = np.max(epf[w > 0])
        fwhm = signal.peak_widths(epf[w > 0], peaks=np.array([peak_ind]), rel_height=0.5)
        q_factor = w[w > 0][peak_ind] / (np.diff(w)[0] * fwhm[0])
        ax[0, 0].semilogy(mu, w[w > 0][peak_ind] / V, 'o', color='C' + str(vInd),
                          alpha=(vInd + 1) / len(volFolders))
        ax[0, 1].semilogy(mu, np.diff(w)[0] * fwhm[0] / V, 'o', color='C' + str(vInd),
                          alpha=(vInd + 1) / len(volFolders))
        ax[1, 0].semilogy(mu, q_factor / V, 'o', color='C' + str(vInd),
                          alpha=(vInd + 1) / len(volFolders))
        ax[1, 1].semilogy(mu, peak / V, 'o', color='C' + str(vInd),
                          alpha=(vInd + 1) / len(volFolders))

ax[0, 0].set(ylabel=r'$\omega_\mathrm{peak}/V$')
ax[0, 1].set(ylabel=r'$\Delta \omega_\mathrm{FWHM}/V$')
ax[1, 0].set(ylabel=r'$Q/V$', xlabel=r'$\Delta \mu$')
ax[1, 1].set(ylabel=r'$\mathrm{max}\left( \mathcal{E} \right)/V$', xlabel=r'$\Delta \mu$')

elements = [mpl.lines.Line2D([0], [0], color='C' + str(vInd),
                             marker='o', lw=0,
                             label='V={V}'.format(V=int(v[1:]))) for vInd, v in enumerate(volFolders)]
ax[1, 1].legend(handles=elements)
plt.tight_layout()
plt.show()
