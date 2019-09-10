import os
import sys
from glob import glob
import numpy as np
# import matplotlib.pyplot as plt
import h5py
import freqent.freqent as fe
import multiprocessing


def calc_epr_spectral(file):
    '''
    function to pass to multiprocessing pool to run parallel simulations
    '''
    print('Reading {f}'.format(f=file.split(os.path.sep)[-2]))
    with h5py.File(file) as d:
        t_points = d['data']['t_points'][:]
        t_epr = np.where(t_points > 1000)[0]  # only calculate epr after t = 10
        dt = np.diff(t_points)[0]
        nSim = d['params']['nSim'][()]
        s = np.zeros(nSim)
        rhos = np.zeros((nSim, len(t_epr)))

        for ind, traj in enumerate(d['data']['trajs'][..., t_epr]):
            s[ind], rhos[ind], w = fe.entropy(traj, sample_spacing=dt,
                                              sigma=sigma, return_density=True)

        # if '/data/s' in d:
        #     d['data']['s'][...] = s
        # else:
        #     d['data'].create_dataset('s', data=s)

        # if '/data/s_density' in d:
        #     d['data']['s_density'][...] = rhos
        # else:
        #     d['data'].create_dataset('s_density', data=rhos)

        # if '/data/omega' in d:
        #     d['data']['omega'][...] = w
        # else:
        #     d['data'].create_dataset('omega', data=w)

        # if '/params/sigma' in d:
        #     d['params']['sigma'][...] = sigma
        # else:
        #     d['params'].create_dataset('sigma', data=sigma)
    return s, rhos, w


if sys.platform == 'linux':
    folder = '/mnt/llmStorage203/Danny/brusselatorSims/reactionsOnly/190904/'
if sys.platform == 'darwin':
    folder = '/Volumes/Storage/Danny/brusselatorSims/reactionsOnly/190904/'

files = glob(os.path.join(folder, 'alpha*', 'data.hdf5'))
sigma = 50

print('Calculating eprs...')
with multiprocessing.Pool(processes=4) as pool:
    result = pool.map(calc_epr_spectral, files)
print('Done.')
