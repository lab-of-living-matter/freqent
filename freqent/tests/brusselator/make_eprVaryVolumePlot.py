import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py
import os
from glob import glob

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

folder = '/Volumes/Storage/Danny/brusselatorSims/reactionsOnly/200422/'
volFolders = ['v10', 'v50', 'v100', 'v500', 'v1000', 'v5000', 'v10000']

fig, ax = plt.subplots(figsize=(6.5, 6.2))
for vInd, v in enumerate(volFolders):
    f = os.path.join(folder, v)
    muFolders = glob(os.path.join(f, 'mu*'))
    mus = np.zeros(len(muFolders))
    eprs = np.zeros(len(muFolders))
    epr_blinds = np.zeros(len(muFolders))
    for muInd, m in enumerate(muFolders):
        with h5py.File(os.path.join(m, 'data.hdf5')) as d:
            # mus[muInd] = np.log((d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4]) /
            #                     (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))
            mus[muInd] = float(m.split(os.path.sep)[-1][2:-7])
            eprs[muInd] = d['data']['epr'][()]
            epr_blinds[muInd] = d['data']['epr_blind'][()]
            V = d['params']['V'][()]
    mu_order = np.argsort(mus)
    ax.semilogy(mus[mu_order], eprs[mu_order] / V, 'o-', color='C0', alpha=(vInd + 1) / len(volFolders))
    ax.semilogy(mus[mu_order], epr_blinds[mu_order] / V, '-o', color='C1', alpha=(vInd + 1) / len(volFolders))

legend_elements = [mpl.lines.Line2D([0], [0], color='k', marker='o', alpha=1 / len(volFolders), label=r'V=10'),
                   mpl.lines.Line2D([0], [0], color='k', marker='o', alpha=2 / len(volFolders), label=r'V=50'),
                   mpl.lines.Line2D([0], [0], color='k', marker='o', alpha=3 / len(volFolders), label=r'V=100'),
                   mpl.lines.Line2D([0], [0], color='k', marker='o', alpha=4 / len(volFolders), label=r'V=500'),
                   mpl.lines.Line2D([0], [0], color='k', marker='o', alpha=5 / len(volFolders), label=r'V=1000'),
                   mpl.lines.Line2D([0], [0], color='k', marker='o', alpha=6 / len(volFolders), label=r'V=5000'),
                   mpl.lines.Line2D([0], [0], color='k', marker='o', alpha=7 / len(volFolders), label=r'V=10000')]

ax.legend(handles=legend_elements)
ax.set(xlabel=r'$\Delta \mu$', ylabel=r'$dS/dt$')

ax.text(6.5, 0.75, r'$\dot{S}_\mathrm{blind}$', color='C1', fontsize=15)
ax.text(6.5, 50, r'$\dot{S}_\mathrm{true}$', color='C0', fontsize=15)

plt.show()
