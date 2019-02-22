import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import freqent.freqent as fe

folder = '/media/daniel/storage11/local_LLM_Danny/freqent/190220/highResBruss'

alpha = np.array([float(x.split(os.path.sep)[-1].split('_')[0].split('alpha')[-1]) for x in glob(os.path.join(folder, '*.hdf5'))])

epr = np.zeros(len(alpha))

sigmaArray = list(range(1, 11))
epr_spectral = np.zeros((len(sigmaArray), len(alpha)))


for fInd, fullPath in enumerate(glob(os.path.join(folder, '*.hdf5'))):
    f = fullPath.split(os.path.sep)[-1]
    print('Opening {f}...'.format(f=f))
    with h5py.File(fullPath, 'r') as d:
        epr[fInd] = d['data']['epr'][()]
        # trajs = d['data']['trajs'][:]
        # t_points = d['data']['t_points'][:]
        dt = np.diff(d['data']['t_points'][:])[0]

        for sigInd, sigma in enumerate(sigmaArray):
            epr_spectral[sigInd, fInd] = fe.entropy(d['data']['trajs'][:],
                                                    sample_spacing=dt,
                                                    sigma=sigma)

# fig, ax = plt.subplots()
# ax.loglog(alpha, epr, '.', label='True EPR')
# ax.loglog(alpha, epr_spectral, '.', label='Spectral EPR')
# ax.set(xlabel=r'$\alpha$', ylabel=r'$\dot{\Sigma}$')
# plt.legend()

# plt.show()
