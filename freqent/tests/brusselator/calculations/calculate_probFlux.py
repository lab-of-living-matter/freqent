import numpy as np
import h5py
import argparse
import glob
import os
from probabilityFlux import *
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', '-file', type=str,
                    help='absolute path to parent folder with all data')
parser.add_argument('--dbin', type=float,
                    help='size of bins used to discretize space')

args = parser.parse_args()


def calculate_probFlux(file):
    print(file.split(os.path.sep)[-2])
    with h5py.File(file) as d:
        # alpha = ((d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4]) /
        #          (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))
        data = d['data']['trajs'][:, :, d['data']['t_points'][:] > 10]
        nrep, nvar, nt = data.shape
        data = np.reshape(np.moveaxis(data, 1, 0), (nvar, nrep * nt))
        dt = np.diff(d['data']['t_points'][:])[0]

        edges = [np.arange(data[0].min() - 5.5, data[0].max() + 5.5, args.dbin),
                 np.arange(data[1].min() - 5.5, data[1].max() + 5.5, args.dbin)]

        prob_map, flux_field, edges = probabilityFlux(data.T[::10], dt=dt, bins=edges)

        if '/data/prob_map' in d:
            del d['data']['prob_map']
        d['data'].create_dataset('prob_map', data=prob_map)

        if '/data/flux_field' in d:
            del d['data']['flux_field']
        d['data'].create_dataset('flux_field', data=flux_field)

        if '/data/edges_x' in d:
            del d['data']['edges_x']
        d['data'].create_dataset('edges_x', data=edges[0])

        if '/data/edges_y' in d:
            del d['data']['edges_y']
        d['data'].create_dataset('edges_y', data=edges[1])


files = glob.glob(os.path.join(args.dataPath, '*', '*hdf5'))

print('Calculating phase fluxes of:')
with multiprocessing.Pool(processes=4) as pool:
    result = pool.map(calculate_probFlux, files)
print('Done.')
