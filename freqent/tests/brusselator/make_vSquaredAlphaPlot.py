import numpy as np
import matplotlib.pyplot as plt
import h5py





alphas = np.zeros(len(files))
vsqs = np.zeros(len(files))
for ind, file in enumerate(files):
    with h5py.File(file) as d:
        alphas[ind] = d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] / (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5])
        dx, dy = np.diff(d['data']['edges_x'])[0], np.diff(d['data']['edges_y'])[0]
        dudy, dudx = np.gradient(d['data']['flux_field'][0].T, 4)
        dvdy, dvdx = np.gradient(d['data']['flux_field'][1].T, 4)
        curls[ind] = np.sum(dudx * dvdy - dudy * dvdx)
        vsqs[ind] = np.sum(d['data']['flux_field'][0].T**2 + d['data']['flux_field'][1].T**2)
