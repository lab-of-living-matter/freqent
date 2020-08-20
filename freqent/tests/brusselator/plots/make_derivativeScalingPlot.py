import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py
import os
from glob import glob
import argparse
from scipy import ndimage
from scipy import stats

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
parser.add_argument('--folder', '-f', type=str,
                    help='parent folder of data')
args = parser.parse_args()

volFolders = ['v10', 'v50', 'v100', 'v500', 'v1000', 'v5000', 'v10000']
max_derivative = np.zeros(len(volFolders))
mu_max_derivative = np.zeros(len(volFolders))
V = np.zeros(len(volFolders))

for vInd, v in enumerate(volFolders):
    f = os.path.join(args.folder, v)
    muFolders = glob(os.path.join(f, 'mu*'))
    mus = np.zeros(len(muFolders))
    epr_blinds = np.zeros(len(muFolders))

    for muInd, m in enumerate(muFolders):
        with h5py.File(os.path.join(m, 'data.hdf5')) as d:
            mus[muInd] = float(m.split(os.path.sep)[-1][2:-7])
            epr_blinds[muInd] = d['data']['epr_blind'][()].mean()
            V[vInd] = d['params']['V'][()]

    mu_order = np.argsort(mus)
    dmu = np.diff(mus[mu_order])[0]
    max_derivative[vInd] = ndimage.gaussian_filter1d(epr_blinds[mu_order], sigma=1, order=1).max()
    mu_max_derivative[vInd] = mus[mu_order][np.argmax(ndimage.gaussian_filter1d(epr_blinds[mu_order],
                                                                                sigma=1, order=1))]

# perform linear fit to logged data
m, b, r, p, sig = stats.linregress(np.log10(V[:-1]), np.log10(max_derivative[:-1]))

fig, ax = plt.subplots(figsize=(4.9, 4.6))
ax.plot(V, V**m * 10**b, 'r--')
ax.plot(V, max_derivative, 'ko')
ax.set(xscale='log', yscale='log', xlabel=r'$V$',
       ylabel=r'max$\left( \partial \dot{S}_{\mathrm{blind}} / \partial \Delta \mu \right)$')
ax.text(V[1], max_derivative[1] * 0.5, r'$\propto V^{{{m:0.2f} \pm {sigma:0.2f}}}$'.format(m=m, sigma=sig**2), color='r')
plt.tight_layout()

# fig2, ax2 = plt.subplots(figsize=(4.9, 4.6))
# ax.plot(V, V**m * 10**b, 'r--')
# ax2.plot(V, mu_max_derivative, 'ko')
# ax2.set(xscale='log', yscale='log', xlabel=r'$V$')
        # ylabel=r'max$\left( \partial \dot{S}_{\mathrm{blind}} / \partial \Delta \mu \right)$')
# ax.text(0.5e3, 1e2, r'$\propto V^{{{m:0.2f} \pm {sigma:0.2f}}}$'.format(m=m, sigma=sig), color='r')
plt.tight_layout()
plt.show()
