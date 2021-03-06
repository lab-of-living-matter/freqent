import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py
import os
from glob import glob
import argparse

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

volFolders = ['v10', 'v100', 'v1000']

fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=150)
for vInd, v in enumerate(volFolders):
    f = os.path.join(args.folder, v)
    muFolders = glob(os.path.join(f, 'mu*'))
    mus = np.zeros(len(muFolders))
    eprs = np.zeros(len(muFolders))
    epr_blinds = np.zeros(len(muFolders))
    s_mean = np.zeros(len(muFolders))
    s_std = np.zeros(len(muFolders))
    for muInd, m in enumerate(muFolders):
        with h5py.File(os.path.join(m, 'data.hdf5')) as d:
            # mus[muInd] = np.log((d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4]) /
            #                     (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))
            mus[muInd] = float(m.split(os.path.sep)[-1][2:-7])
            eprs[muInd] = d['data']['epr'][()]
            epr_blinds[muInd] = d['data']['epr_blind'][()]
            #s_mean[muInd] = d['data']['s'][()].mean()
            #s_std[muInd] = d['data']['s'][()].std()
            V = d['params']['V'][()]
            nCompartments = d['params']['nCompartments'][()]

    mu_order = np.argsort(mus)
    # ax.semilogy(mus[mu_order], eprs[mu_order] / (nCompartments * V), 'o-',
                # color='C0', alpha=(vInd + 1) / len(volFolders))
    ax.semilogy(mus[mu_order], epr_blinds[mu_order] / (nCompartments * V), '-o',
                color='C1', alpha=(vInd + 1) / len(volFolders))
    # ax.errorbar(mus[mu_order], s_mean[mu_order] / V, yerr=s_std[mu_order] / V,
                # marker='o', color='k', capsize=5, alpha=(vInd + 1) / len(volFolders))

legend_elements = [mpl.lines.Line2D([0], [0], color='k', marker='o', alpha=1 / len(volFolders), label=r'V=10'),
                   mpl.lines.Line2D([0], [0], color='k', marker='o', alpha=2 / len(volFolders), label=r'V=100'),
                   mpl.lines.Line2D([0], [0], color='k', marker='o', alpha=3 / len(volFolders), label=r'V=1000')]

ax.legend(handles=legend_elements)
ax.set(xlabel=r'$\Delta \mu$', ylabel=r'$\dot{S}/V$')

# ax.text(6.5, 5, r'$\dot{S}_\mathrm{blind}$', color='C1', fontsize=15)
# ax.text(6.5, 50, r'$\dot{S}_\mathrm{true}$', color='C0', fontsize=15)

plt.show()
