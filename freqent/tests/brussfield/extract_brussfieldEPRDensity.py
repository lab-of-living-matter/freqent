import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import freqent.freqentn as fen
mpl.rcParams['pdf.fonttype'] = 42

parentFolder = '/mnt/llmStorage203/Danny/brusselatorSims/fieldSims/190509/brussfield'
# parentFolder = '/home/daniel/Desktop/'
folders = glob(os.path.join(parentFolder, 'alpha*'))
alphas = np.asarray([float(f.split(os.path.sep)[-1].split('_')[0][5:]) for f in folders])

for fInd, f in enumerate(folders):
    with h5py.File(os.path.join(f, 'data.hdf5'), 'r+') as d:
        dt = np.diff(d['data']['t_points'][:])[0]
        dx = d['params']['lCompartment'][()]
        nt = d['params']['n_t_points'][()]
        epr, epr_density, w = fen.entropy(d['data']['trajs'][..., nt // 2:, :-1],
                                          sample_spacing=[dt, dx],
                                          window='boxcar',
                                          detrend='constant',
                                          smooth_corr=True,
                                          nfft=None,
                                          sigma=int(d['params']['sigma'][()]),
                                          subtract_bias=True,
                                          many_traj=True,
                                          return_density=True)

        if '/data/epr_density' in d:
            d['data']['epr_density'][...] = epr_density
        else:
            d['data'].create_dataset('epr_density', data=epr_density)

        if '/data/omega' in d:
            d['data']['omega'][...] = w[0]
        else:
            d['data'].create_dataset('omega', data=w[0])

        if '/data/k' in d:
            d['data']['k'][...] = w[0]
        else:
            d['data'].create_dataset('k', data=w[1])

        print('alpha = {a}, epr_spectral={s}, epr={s1}'.format(a=alphas[fInd], s=d['data']['epr_spectral'][()], s1=epr))

        # plot data
        fig, ax = plt.subplots()
        a = ax.pcolormesh(w[1], w[0], epr_density, rasterized=True)
        fig.colorbar(a, ax=ax)
        ax.set(xlabel=r'$k$', ylabel=r'$\omega$')
        ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
        fig.savefig(os.path.join(f, 'epr_density.pdf'), format='pdf')
        plt.close('all')
