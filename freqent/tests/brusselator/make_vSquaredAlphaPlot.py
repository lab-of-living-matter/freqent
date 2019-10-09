import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os
from glob import glob

if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/reactionsOnly/190904/'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations/'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/reactionsOnly/190904/'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations/'

files = glob(os.path.join(datapath, 'alpha*', 'data.hdf5'))

alphas = np.zeros(len(files))
vsqs = np.zeros(len(files))
for ind, file in enumerate(files):
    with h5py.File(file) as d:
        alphas[ind] = (d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
                       (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))
        dx, dy = np.diff(d['data']['edges_x'])[0], np.diff(d['data']['edges_y'])[0]
        dudy, dudx = np.gradient(d['data']['flux_field'][0].T, 4)
        dvdy, dvdx = np.gradient(d['data']['flux_field'][1].T, 4)
        vsqs[ind] = np.sum(d['data']['flux_field'][0].T**2 + d['data']['flux_field'][1].T**2)


fig, ax = plt.subplots()
ax.plot(alphas, vsqs, 'ko')
