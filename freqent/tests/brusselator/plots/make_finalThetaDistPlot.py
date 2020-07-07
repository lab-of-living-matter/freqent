import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import os
import argparse
from datetime import datetime

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
parser.add_argument('--files', '-f', type=str, nargs='+',
                    help='absolute paths to files to make plot with')
parser.add_argument('--savepath', '-s', type=str, default=None,
                    help='absolute path of where to save plot')
args = parser.parse_args()

fig, ax = plt.subplots()
for file in args.files:
    theta_ends = np.zeros(50)
    with h5py.File(os.path.join(file, 'data.hdf5')) as d:
        t = d['data']['t_points'][:]
        mu = np.log((d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4]) /
                    (d['params']['C'] * d['params']['rates'][3] * d['params']['rates'][5]))
        for ind, (x, y) in enumerate(zip(d['data']['trajs'][:, 0, len(t) // 2:],
                                         d['data']['trajs'][:, 1, len(t) // 2:])):
            theta_ends[ind] = np.unwrap(np.arctan2(y - y.mean(), x - x.mean()))[-1] / (2 * np.pi)
            ax.errorbar(mu, np.mean(theta_ends), yerr=np.std(theta_ends),
                        marker='o', capsize=7, color='k', lw=2)

ax.plot(ax.set_xlim(), [0, 0], 'r--', alpha=0.5)
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
ax.set(xlabel=r'$\Delta \mu$', ylabel=r'$\theta(T) / 2 \pi$')

if args.savepath is not None:
    fig.savefig(os.path.join(args.saveapth, datetime.now().strftime('%y%m%d') + '_finalThetaDist.pdf'),
                format='pdf')
