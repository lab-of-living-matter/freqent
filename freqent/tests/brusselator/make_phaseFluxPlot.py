import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import argparse
from probabilityFlux import *

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42

parser = argparse.ArgumentParser()
parser.add_argument('--filePath', '-file', type=str,
                    help='absolute path to hdf5 file with data')
parser.add_argument('--dbin', type=float,
                    help='size of bins used to discretize space')
parser.add_argument('--savePath', '-save', type=str,)

args = parser.parse_args()

with h5py.File(args.filePath) as d:
    alpha = ((d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4]) /
             (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))
    data = d['data']['trajs'][:, :, d['data']['t_points'][:] > 10]
    nrep, nvar, nt = data.shape
    data = np.reshape(np.moveaxis(data, 1, 0), (nvar, nrep * nt))
    dt = np.diff(d['data']['t_points'][:])[0]

edges = [np.arange(data[0].min() - 5.5, data[0].max() + 5.5, args.dbin),
         np.arange(data[1].min() - 5.5, data[1].max() + 5.5, args.dbin)]

prob_map, flux_field, edges = probabilityFlux(data.T[::10], dt=dt, bins=edges)

xx, yy = np.meshgrid(edges[0][:-1], edges[1][:-1])
dx, dy = [np.diff(edges[0])[0], np.diff(edges[1])[0]]
yrange = edges[1][-1] - edges[1][0]

fig, ax = plt.subplots(figsize=(5, 5))
ax.pcolormesh(edges[0], edges[1], prob_map.T, rasterized=True)
ax.quiver(xx + dx / 2, yy + dy / 2, flux_field[0].T, flux_field[1].T,
          color='w', scale=1)

ax.set_aspect('equal')
ax.set(xlabel='X', ylabel='Y', title=r'$\alpha = {a}$'.format(a=alpha),
       xlim=[200 - yrange / 2, 200 + yrange / 2])
ax.tick_params(which='both', direction='in')
plt.tight_layout()

plt.show()
