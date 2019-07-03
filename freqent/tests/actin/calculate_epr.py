import os
import numpy as np
import h5py
import freqent.freqentn as fen
from datetime import datetime
import pandas as pd
import sys

if sys.platform == 'darwin':
  parentDir = '/Users/Danny/Dropbox/LLM_Danny/freqent/actin/'
if sys.platform == 'linux':
  parentDir = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/actin/'

noncontractile_exps = os.listdir(os.path.join(parentDir, 'noncontractile'))
thermal_exps = os.listdir(os.path.join(parentDir, 'thermal'))

files_noncontractile = ['051612_skmmII_noncontr.hdf5',
                        '053112_smmm_noncontr_Fig4.hdf5',
                        '071813_2.hdf5',
                        '111116_2_NC_spun_skmm.hdf5',
                        '111116_3_NC_spun_skmm.hdf5',
                        '112916_1_NC_spun_647_skmm.hdf5',
                        '112916_3_NC_spun_647_skmm.hdf5',
                        '120216_2_NC_spun_647_skmm.hdf5',
                        '120216_3_NC_spun_647_skmm.hdf5',
                        '120916_3-2_NC_spun_647_skmm_0pt75mMATP.hdf5',
                        '121516_1_NC_unspun_647_skmm.hdf5',
                        '121516_4_NC_unspun_647_skmm.hdf5']
#                         '052012_sm.hdf5',
#   These data sets       '053112_2_sm_nocontract.hdf5',
#   all have dt=30s       '060312_sm_nocontract.hdf5',
#   so may not see        '060412_2_nmmII_1to300x_noncontr.hdf5',
#   relevant time-        '071813_2_nmmII.hdf5',
#   scales                '071913_2_nmmII_Fig4.hdf5',
#                         '729130_sm_nocontract.hdf5']

files_thermal = ['112916_2_imaging.hdf5',
                 '121516_3_imaging.hdf5',
                 '111116_3_imaging.hdf5',
                 '111116_2_imaging.hdf5',
                 # '120216_2_imaging.hdf5',
                 '120916_1_imaging.hdf5',
                 '112916_3_imaging.hdf5',
                 '121516_2_imaging.hdf5',
                 '121516_4_imaging.hdf5',
                 '120216_2_noskmm_thermalFactinNetwork.hdf5',
                 '120216_3_imaging.hdf5']


epr_nc = []  #np.zeros(len(files_noncontractile))
epr_thermal = []  #np.zeros(len(files_thermal))

window = 'boxcar'
nfft = [2**8, 2**6, 2**6]
detrend = 'linear'
smooth_corr = True
sigma = [10, 5, 5]
subtract_bias = True
many_traj = False

for ind, file in enumerate(files_noncontractile):
    with h5py.File(os.path.join(parentDir, 'noncontractile', file)) as f:
        data = np.stack((f['data']['ordermat'][:, 2:-2, 2:-2], np.transpose(f['data']['cgim'][:, 2:-2, 2:-2], axes=[0, 2, 1])))
        # data = np.stack((f['data']['cgim'], np.transpose(f['data']['anglemat'], axes=[0, 2, 1])))
        dx = f['params']['dx'][()]
        dt = f['params']['dt'][()]
        winsize = f['params']['winsize'][()]
        overlap = f['params']['overlap'][()]
        winspace = int(winsize - np.ceil(winsize * overlap))
        s, sdensity = fen.entropy(data, [dt, dx * winspace, dx * winspace],
                                  window=window,
                                  nfft=nfft,
                                  detrend=detrend,
                                  smooth_corr=smooth_corr,
                                  sigma=sigma,
                                  subtract_bias=subtract_bias,
                                  many_traj=many_traj)

        epr_nc.append(s.real)

for ind, file in enumerate(files_thermal):
    with h5py.File(os.path.join(parentDir, 'thermal', file)) as f:
        data = np.stack((f['data']['ordermat'][:, 2:-2, 2:-2], np.transpose(f['data']['cgim'][:, 2:-2, 2:-2], axes=[0, 2, 1])))
        # data = np.stack((f['data']['cgim'], np.transpose(f['data']['anglemat'], axes=[0, 2, 1])))
        dx = f['params']['dx'][()]
        dt = f['params']['dt'][()]
        winsize = f['params']['winsize'][()]
        overlap = f['params']['overlap'][()]
        winspace = int(winsize - np.ceil(winsize * overlap))
        s, sdensity = fen.entropy(data, [dt, dx * winspace, dx * winspace],
                                  window=window,
                                  nfft=nfft,
                                  detrend=detrend,
                                  smooth_corr=smooth_corr,
                                  sigma=sigma,
                                  subtract_bias=subtract_bias,
                                  many_traj=many_traj)

        epr_thermal.append(s.real)

labels = ['noncontractile'] * len(epr_nc) + ['thermal'] * len(epr_thermal)
epr = epr_nc + epr_thermal
files = [file.split('.')[0] for file in files_noncontractile + files_thermal]

if nfft is None:
    nfft = ''

params = {'window': window,
          'nfft': nfft,
          'detrend': detrend,
          'smooth_corr': smooth_corr,
          'sigma': sigma,
          'subtract_bias': subtract_bias,
          'many_traj': many_traj}

paramsattrs = {'window': 'window used in calculating fft of signal',
               'nfft': 'size of nfft window if zero-padding wanted',
               'detrend': 'how to detrend signal',
               'smooth_corr': 'boolean to either smooth correlation functions or not',
               'sigma': 'size of gaussian used to smooth in each dimension',
               'subtract_bias': 'boolean of whether to subtract bias off epr estimate',
               'many_traj': 'boolean of whether passing multiple trajectories into freqent.freqentn.entropy()'}


# with h5py.File(os.path.join(parentDir, datetime.today().strftime('%y%m%d') + '_epr.hdf5'), 'w') as f:
#     datagrp = f.create_group('epr')
#     paramsgrp = f.create_group('params')

#     for ind, file in enumerate(files):
#         d = datagrp.create_dataset(file, data=epr[ind])
#         d.attrs['experiment type'] = labels[ind]

#     for paramname in params.keys():
#         p = paramsgrp.create_dataset(paramname, data=params[paramname])
#         p.attrs['description'] = paramsattrs[paramname]
