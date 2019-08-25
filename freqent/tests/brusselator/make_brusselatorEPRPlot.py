import os
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
    print('Reading {f}'.format(f=file))
    with h5py.File(file, 'r') as d:
        t_points = d['data']['t_points'][:]
        T = t_points.max()
        tStart = T / 2
        dt = np.diff(d['data']['t_points'][:])[0]
        epr = d['data']['epr'][()]
        epr_blind = d['data']['epr_blind'][()]
        # epr_spectral = d['data']['epr_spectral'][()]
        epr_spectral = [(fe.entropy(traj, sample_spacing=dt, sigma=10)).real for traj in d['data']['trajs'][..., t_points > tStart]]
    return [epr, epr_blind, np.mean(epr_spectral), np.std(epr_spectral)]


folder = '/media/daniel/storage11/local_LLM_Danny/freqent/190313/blindBruss'

alpha = np.array([float(x.split(os.path.sep)[-1].split('_')[0].split('alpha')[-1]) for x in glob(os.path.join(folder, '*.hdf5'))])

# epr = np.zeros(len(alpha))

# sigmaArray = [10]  # list(range(1, 11))
# epr_spectral = np.zeros((len(sigmaArray), len(alpha)))
# # epr_spectral = np.zeros(len(alpha))

files = glob(os.path.join(folder, '*.hdf5'))

print('Getting and calculating eprs...')
with multiprocessing.Pool(processes=8) as pool:
    result = pool.map(calc_epr_spectral, files)
print('Done.')


# for fInd, fullPath in enumerate(glob(os.path.join(folder, '*.hdf5'))):
#     f = fullPath.split(os.path.sep)[-1]
#     print('Opening {f}...'.format(f=f))
#     with h5py.File(fullPath, 'r') as d:
#         epr[fInd] = d['data']['epr'][()]
#         # epr_spectral[fInd] = d['data']['epr_spectral'][()]
#         trajs = d['data']['trajs'][:]
#         t_points = d['data']['t_points'][:]
#         T = t_points.max()
#         tStart = T / 10
#         dt = np.diff(d['data']['t_points'][:])[0]

#         for sigInd, sigma in enumerate(sigmaArray):
#             epr_spectral[sigInd, fInd] = (fe.entropy(d['data']['trajs'][..., t_points > tStart],
#                                                      sample_spacing=dt,
#                                                      sigma=sigma)).real

# fig, ax = plt.subplots()
# ax.loglog(alpha, epr, '.', label='True EPR')
# ax.loglog(alpha, epr_spectral.T, '.', label='Spectral EPR')
# ax.set(xlabel=r'$\alpha$', ylabel=r'$\dot{\Sigma}$')
# plt.legend()

# plt.show()
