import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import os
import sys
import freqent.freqent as fe
from datetime import datetime
mpl.rcParams['pdf.fonttype'] = 42


if sys.platform == 'linux':
    dataFolder = '/mnt/llmStorage203/Danny/brusselatorSims/reactionsOnly/190904/'
    saveFolder = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/brusselator/'
if sys.platform == 'darwin':
    dataFolder = '/Volumes/Storage/Danny/brusselatorSims/reactionsOnly/190904/'
    saveFolder = '/Users/Danny/Dropbox/LLM_Danny/freqent/figures/brusselator/'

fig, ax = plt.subplots(1, 2)
for file in os.listdir(parentFolder):
    if file.endswith('.hdf5'):
        ax[0].cla(), ax[1].cla()
        with h5py.File(os.path.join(parentFolder, file), 'r') as f:
            nt = len(f['data']['t_points'][:])
            dt = np.diff(f['data']['t_points'][:])[0]
            s, rhos, w = fe.entropy(f['data']['trajs'][..., nt // 2:], dt,
                                    sigma=f['params']['sigma'][()], return_epf=True)
            ax[0].plot(f['data']['trajs'][0, 0], f['data']['trajs'][0, 1])
            ax[1].plot(w[w != 0], rhos[w != 0])
            ax[1].set(xlim=[-20, 20])
        plt.pause(0.0001)

