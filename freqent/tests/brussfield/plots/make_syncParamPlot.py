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

files = [os.path.join(f, 'data.hdf5') for f in glob(os.path.join(args.parentFolder, 'mu*'))]
mean_syncParam = np.zeros(len(files))
std_syncParam = np.zeros(len(files))
mus = np.zeros(len(files))

fig, ax = plt.subplots()
for file_index, file in enumerate(files):
    mus[file_index] = float(file.split(os.path.sep)[-2][2:-7])
    with h5py.File(file) as d:
        mean_syncParam[file_index] = d['data']['syncParam'][()].mean(axis=1).mean()
        std_syncParam[file_index] = d['data']['syncParam'][()].mean(axis=1).std()

ax.plot(sorted(mus), mean_syncParam[np.argsort(mus)], 'ko-')
ax.fill_between(sorted(mus),
                mean_syncParam[np.argsort(mus)] - std_syncParam[np.argsort(mus)],
                mean_syncParam[np.argsort(mus)] + std_syncParam[np.argsort(mus)],
                color='k', alpha=0.5)
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
ax.set(xlabel=r'$\Delta \mu$', ylabel=r'$\langle r \rangle$')
plt.tight_layout()

plt.show()
