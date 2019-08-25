import glob
import os
import numpy as np
import h5py
import freqent.freqentn as fen

parentDir = '/mnt/llmStorage203/Danny/freqent/gaussfield/190501/gaussfield/'

folders = glob.glob(os.path.join(parentDir, 'alpha*'))
tFactor = 10

alphas = np.array([float(f.split('alpha')[1].split('_')[0]) for f in folders])
epr = np.zeros(len(alphas))

for fInd, f in enumerate(folders):
    print(f)
    with h5py.File(os.path.join(f, 'data.hdf5')) as data:
        s, rhos, w = fen.entropy(data['data']['trajs'][..., data['params']['nsteps'][()] // 2::tFactor, :],
                                 sample_spacing=[data['params']['dt'][()] * tFactor, data['params']['dx'][()]],
                                 detrend='constant',
                                 many_traj=True,
                                 return_density=True,
                                 sigma=[10, 0.1])
    epr[fInd] = s
