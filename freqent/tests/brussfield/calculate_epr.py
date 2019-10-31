import os
import sys
from glob import glob
import numpy as np
import h5py
import freqent.freqentn as fen
import multiprocessing


def calc_epr_spectral(file):
    '''
    function to pass to multiprocessing pool to calculate epr in parallel
    '''
    print('Reading {f}'.format(f=file.split(os.path.sep)[-2]))
    with h5py.File(file) as d:
        t_points = d['data']['t_points'][:]
        t_epr = np.where(t_points > 10)[0]
        dt = np.diff(t_points)[0]
        dx = d['params']['lCompartment'][()]
        # nCompartments = d['params']['nCompartments'][()]
        nSim = d['params']['nSim'][()]

        s = np.zeros(nSim)
        nt, nx = d['data']['trajs'][0, 0, t_epr, :].shape
        rhos = np.zeros((nSim, nt - (nt + 1) % 2, nx - (nx + 1) % 2))

        for ind, traj in enumerate(d['data']['trajs'][..., t_epr, :]):
            s[ind], rhos[ind], w = fen.entropy(traj, sample_spacing=[dt, dx],
                                               window='boxcar', detrend='constant',
                                               smooth_corr=True, nfft=None,
                                               sigma=sigma,
                                               subtract_bias=True,
                                               many_traj=False,
                                               return_density=True)

        if '/data/s' in d:
            del d['data']['s']
        d['data'].create_dataset('s', data=s)

        if '/data/rhos' in d:
            del d['data']['rhos']
        d['data'].create_dataset('rhos', data=rhos)

        if '/data/omega' in d:
            del d['data']['omega']
        d['data'].create_dataset('omega', data=w[0])

        if '/data/k' in d:
            del d['data']['k']
        d['data'].create_dataset('k', data=w[1])

        if '/params/sigma/' in d:
            del d['params']['sigma']
        d['params'].create_dataset('sigma', data=sigma)

    return s, rhos, w


if sys.platform == 'darwin':
    dataFolder = '/Volumes/Storage/Danny/brusselatorSims/fieldSims/191028'
if sys.platform == 'linux':
    dataFolder = '/mnt/llmStorage203/Danny/brusselatorSims/fieldSims/191028'

files = glob(os.path.join(dataFolder, 'alpha*', 'data.hdf5'))
sigma = [75, 5]

print('Calculating eprs...')
with multiprocessing.Pool(processes=4) as pool:
    result = pool.map(calc_epr_spectral, files)
print('Done.')
