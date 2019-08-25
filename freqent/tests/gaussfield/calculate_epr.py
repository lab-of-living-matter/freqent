import glob
import os
import numpy as np
import h5py
import freqent.freqentn as fen
import multiprocessing


def calculate_epr(f):
    print(f)
    with h5py.File(os.path.join(f, 'data.hdf5')) as data:
        s, rhos, w = fen.entropy(data['data']['trajs'][..., data['params']['nsteps'][()] // 2::tFactor, :],
                                 sample_spacing=[data['params']['dt'][()] * tFactor, data['params']['dx'][()]],
                                 detrend='constant',
                                 many_traj=True,
                                 return_density=True,
                                 sigma=int(data['params']['sigma'][()]))
    return s, rhos, w

parentDir = '/mnt/llmStorage203/Danny/freqent/gaussfield/'

folders = glob.glob(os.path.join(parentDir, 'alpha*'))
tFactor = 10

alphas = np.array([float(f.split('alpha')[1].split('_')[0]) for f in folders])
epr = np.zeros(len(alphas))

with multiprocessing.Pool(processes=4) as pool:
    result = pool.map(calculate_epr, folders)
