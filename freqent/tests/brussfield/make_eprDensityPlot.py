import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from datetime import datetime

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2

if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/fieldSims/190910'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/brussfield/eprDensity/'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/fieldSims/190910'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/figures/brussfield/eprDensity/'

files = glob(os.path.join(datapath, 'alpha*', 'data.hdf5'))
alphas = np.zeros(len(files))

for file in files:
    with h5py.File(file, 'r') as d:
        epr_density = np.mean(d['data']['rhos'][:], axis=0)
        alpha = (d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
                 (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))
        t = d['data']['t_points'][:]
        nCompartments = d['params']['nCompartments'][()]
        traj = d['data']['trajs'][0, 0]
        k = d['data']['k'][:] * 100
        w = d['data']['omega'][:]

    fig, ax = plt.subplots()
    a = ax.pcolormesh(k, w, epr_density, rasterized=True,
                      vmin=0, vmax=1e-6)
    ax.set(xlabel=r'$k$', ylabel=r'$\omega$', title=r'$\rho_{\dot{s}}$')
    plt.tight_layout()
    fig.colorbar(a, ax=ax, extend='max')
    ax.set_aspect(np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0])
    fig.suptitle(r'$\alpha = {a}$'.format(a=alpha))
    fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_alpha{a:0.2f}.png'.format(a=alpha)), format='png')

    plt.close('all')

plt.show()
