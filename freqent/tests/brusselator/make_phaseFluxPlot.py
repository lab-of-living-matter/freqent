import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import argparse
from probabilityFlux import *

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', '-file', type=str,
                    help='absolute path to hdf5 file with data')
parser.add_argument('--savepath', '-save', type=str,
                    help='absolute path to save plot')

args = parser.parse_args()

with h5py.File(args.filepath) as d:
    edges_x = d['data']['edges_x'][:]
    edges_y = d['data']['edges_y'][:]
    flux_field = d['data']['flux_field'][:]
    prob_map = d['data']['prob_map'][:]
    alpha = d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] / (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5])

xx, yy = np.meshgrid(edges_x[:-1], edges_y[:-1])
dx, dy = [np.diff(edges_x)[0], np.diff(edges_y)[0]]
yrange = edges_y[-1] - edges_y[0]

fig, ax = plt.subplots(figsize=(5, 5))
ax.pcolormesh(edges_x, edges_y, prob_map.T, rasterized=True)
ax.quiver(xx[::2, ::2] + dx / 2, yy[::2, ::2] + dy / 2, flux_field[0][::2, ::2].T, flux_field[1][::2, ::2].T,
          color='w', scale=1)

ax.set_aspect('equal')
ax.set(xlabel='X', ylabel='Y', title=r'$\alpha = {a}$'.format(a=alpha))
ax.tick_params(which='both', direction='in')
plt.tight_layout()

plt.show()
