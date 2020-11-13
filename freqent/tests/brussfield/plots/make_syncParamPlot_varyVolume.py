import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from datetime import datetime
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
parser.add_argument('parentFolder', type=str,
                    help='parent folder to plot sync param of')
args = parser.parse_args()

volFolders = ['v10', 'v100', 'v1000']


mean_syncParam = np.zeros(len(volFolders), dtype=object)
std_syncParam = np.zeros(len(volFolders), dtype=object)
mus = np.zeros(len(volFolders), dtype=object)

for vol_index, vol in enumerate(volFolders):
    folder = os.path.join(args.parentFolder, vol)
    files = [os.path.join(f, 'data.hdf5') for f in glob(os.path.join(folder, 'mu*'))]
    mean = np.zeros(len(files))
    std = np.zeros(len(files))
    mu = np.zeros(len(files))
    for file_index, file in enumerate(files):
        mu[file_index] = float(file.split(os.path.sep)[-2][2:-7])
        with h5py.File(file) as d:
            mean[file_index] = d['data']['syncParam'][()].mean(axis=1).mean()
            std[file_index] = d['data']['syncParam'][()].mean(axis=1).std()
    mean_syncParam[vol_index] = mean
    std_syncParam[vol_index] = std
    mus[vol_index] = mu

fig, ax = plt.subplots()
colors = ['C0', 'C1', 'C2']
vol = [10, 100, 1000]
legend = [r'$V = 10$', r'$V = 100$', r'$V = 1000$']
markers = ['o', 's', '^']
for ii in range(len(volFolders)):
    ax.plot(np.sort(mus[ii]),
            mean_syncParam[ii][np.argsort(mus[ii])],
            '-', marker=markers[ii], color=colors[ii], markeredgecolor='k',
            label=legend[ii])
    ax.fill_between(np.sort(mus[ii]),
                    mean_syncParam[ii][np.argsort(mus[ii])] - std_syncParam[ii][np.argsort(mus[ii])],
                    mean_syncParam[ii][np.argsort(mus[ii])] + std_syncParam[ii][np.argsort(mus[ii])],
                    color=colors[ii], alpha=0.5)
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
ax.set(xlabel=r'$\Delta \mu$', ylabel=r'$\langle r \rangle$')
ax.legend()
plt.tight_layout()

plt.show()
