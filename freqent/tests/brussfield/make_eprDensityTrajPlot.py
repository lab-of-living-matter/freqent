import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from datetime import datetime
mpl.rcParams['pdf.fonttype'] = 42

if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/fieldSims/190910'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/brussfield/eprDensity_traj/'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/fieldSims/190910'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/figures/brussfield/eprDensity_traj/'

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
    ax[0].cla(), ax[1].cla()
    ax[0].pcolormesh(np.arange(nCompartments), t, traj, cmap='cividis')
    ax[0].set(xlabel=r'$r$', ylabel=r'$t$', title=r'$X(r,t)$')

    a = ax[1].pcolormesh(k, w, epr_density, rasterized=True,
                         vmin=0, vmax=1e-6)
    ax[1].set(xlabel=r'$k$', ylabel=r'$\omega$', title=r'$\rho_{\dot{s}}$')
    # ax[1].text(0.01, 50, r'$\alpha = {a}$'.format(a=alpha), color='w', size=12)
    plt.tight_layout()
    fig.colorbar(a, ax=ax[1], extend='max')
    ax[1].set_aspect(np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0])
    fig.suptitle(r'$\alpha = {a}$'.format(a=alpha))
    fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_alpha{a:0.2f}.png'.format(a=alpha)), format='png')

    plt.close('all')

plt.show()
