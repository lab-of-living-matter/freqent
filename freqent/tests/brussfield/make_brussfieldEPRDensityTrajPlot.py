import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
# import freqent.freqentn as fen
mpl.rcParams['pdf.fonttype'] = 42

parentFolder = '/mnt/llmStorage203/Danny/brusselatorSims/fieldSims/190509/brussfield'
savePath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/brussfield/eprDensity_traj/'
folders = glob(os.path.join(parentFolder, 'alpha*'))
alphas = np.asarray([float(f.split(os.path.sep)[-1].split('_')[0][5:]) for f in folders])

for ind in np.argsort(alphas[1:]):
    f = folders[1:][ind]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    with h5py.File(os.path.join(f, 'data.hdf5'), 'r') as d:
        ax[0].cla(), ax[1].cla()
        ax[0].pcolormesh(np.arange(d['params']['nCompartments'][()]), d['data']['t_points'][:], d['data']['trajs'][0, 0], cmap='Reds')
        ax[0].set(xlabel=r'$r$', ylabel=r'$t$', title=r'$X(r,t)$')
        a = ax[1].pcolormesh(d['data']['k'][:], d['data']['omega'], d['data']['epr_density'],
                             vmin=0, vmax=4 * 10**-5, rasterized=True)

    ax[1].set(xlabel=r'$k$', ylabel=r'$\omega$', title='EPR density')
    ax[1].text(0.01, 50, r'$\alpha = {a}$'.format(a=alphas[1:][ind]), color='w', size=12)
    fig.colorbar(a, ax=ax[1], extend='max')
    ax[1].set_aspect(np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0])
    fig.savefig(os.path.join(savePath, 'alpha{a}.png'.format(a=alphas[1:][ind])), format='png')

    plt.close('all')
