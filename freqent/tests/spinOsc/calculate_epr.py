'''
Use this function to calculate the epr for every trajecotry
in spinOsc simulations
'''
import os
import sys
from glob import glob
import numpy as np
import h5py
import freqent.freqent as fe
import multiprocessing


def calc_epr_spectral(file):
    '''
    function to pass to multiprocessing pool to calculate epr in parallel
    '''
    print('Reading {f}'.format(f=file.split(os.path.sep)[-1]))
    with h5py.File(file) as d:
        t = d['data']['t'][:]
        t_epr = np.where(t > 10)[0]  # only calculate epr after t = 10
        dt = np.diff(t)[0]
        nsim = d['params']['nsim'][()]

        s = np.zeros(nsim)
        rhos = np.zeros((nsim, len(t_epr)))

        for ind, traj in enumerate(d['data']['trajs'][..., t_epr]):
            s[ind], rhos[ind], w = fe.entropy(traj, sample_spacing=dt,
                                              sigma=sigma, return_epf=True)

        if '/data/s' in d:
            d['data']['s'][...] = s
        else:
            d['data'].create_dataset('s', data=s)

        if '/data/s_density' in d:
            d['data']['s_density'][...] = rhos
        else:
            d['data'].create_dataset('s_density', data=rhos)

        if '/data/omega' in d:
            d['data']['omega'][...] = w
        else:
            d['data'].create_dataset('omega', data=w)

        if '/params/sigma' in d:
            d['params']['sigma'][...] = sigma
        else:
            d['params'].create_dataset('sigma', data=sigma)
    return s, rhos, w


if sys.platform == 'linux':
    dataFolder = '/mnt/llmStorage203/Danny/freqent/spinOsc/190709/'
if sys.platform == 'darwin':
    dataFolder = '/Volumes/Storage/Danny/freqent/spinOsc/190709/'

files = glob(os.path.join(dataFolder, '*.hdf5'))
sigma = 30

print('Calculating eprs...')
with multiprocessing.Pool(processes=4) as pool:
    result = pool.map(calc_epr_spectral, files)
print('Done.')
