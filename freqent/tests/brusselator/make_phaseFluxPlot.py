import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import argparse
import os
from glob import glob

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', '-file', type=str,
                    help='absolute path to parent folder of all alpha* folders with data')
parser.add_argument('--savepath', type=str)

args = parser.parse_args()
files = glob(os.path.join(args.datapath, 'alpha*', 'data.hdf5'))

for file in files:
    print(file.split(os.path.sep)[-2])
    with h5py.File(file) as d:
        edges_x = d['data']['edges_x'][:]
        edges_y = d['data']['edges_y'][:]
        flux_field = d['data']['flux_field'][:]
        prob_map = d['data']['prob_map'][:]
        alpha = (d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
                 (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))

    xx, yy = np.meshgrid(edges_x[:-1], edges_y[:-1])
    dx, dy = [np.diff(edges_x)[0], np.diff(edges_y)[0]]
    yrange = edges_y[-1] - edges_y[0]

    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.pcolormesh(edges_x, edges_y, prob_map.T, rasterized=True, cmap='Blues')
    ax.quiver(xx[::2, ::2] + dx / 2, yy[::2, ::2] + dy / 2, flux_field[0][::2, ::2].T, flux_field[1][::2, ::2].T,
              color='k', alpha=0.5, scale=1)

    ax.set_aspect('equal')
    ax.set(xlabel='X', ylabel='Y', title=r'$\alpha = {a}$'.format(a=alpha), xlim=[0, 800], ylim=[0, 1100])
    ax.tick_params(which='both', direction='in')
    plt.tight_layout()

    fig.savefig(os.path.join(args.savepath, 'phaseFluxPlot_alpha{a}.png'.format(a=alpha)), format='png')

