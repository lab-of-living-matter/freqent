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
                        # '071813_2.hdf5',   # small muscle myosin, all others skmm
                        '111116_2_NC_spun_skmm.hdf5',
                        '111116_3_NC_spun_skmm.hdf5',
                        '112916_1_NC_spun_647_skmm.hdf5',
                        '112916_3_NC_spun_647_skmm.hdf5',
                        '120216_2_NC_spun_647_skmm.hdf5',
                        '120216_3_NC_spun_647_skmm.hdf5',
                        '120916_3-2_NC_spun_647_skmm_0pt75mMATP.hdf5',
                        '121516_1_NC_unspun_647_skmm.hdf5',
                        '121516_4_NC_unspun_647_skmm.hdf5']
                        # '053112_smmm_noncontr_Fig4.hdf5',        #
                        # '052012_sm.hdf5',                        #
                        # '053112_2_sm_nocontract.hdf5',           # These data sets
                        # '060312_sm_nocontract.hdf5',             # all have dt=30s
                        # '060412_2_nmmII_1to300x_noncontr.hdf5',  # so may not see
                        # '071813_2_nmmII.hdf5',                   # relevant time-
                        # '071913_2_nmmII_Fig4.hdf5',              # scales
                        # '729130_sm_nocontract.hdf5']             #

files_thermal = ['112916_2_imaging.hdf5',
                 '121516_3_imaging.hdf5',
                 '111116_3_imaging.hdf5',
                 '111116_2_imaging.hdf5',
                 # '120216_2_imaging.hdf5',
                 '120916_1_imaging.hdf5',
                 '112916_3_imaging.hdf5',
                 # '121516_2_imaging.hdf5',
                 '121516_4_imaging.hdf5',
                 '120216_2_noskmm_thermalFactinNetwork.hdf5',
                 '120216_3_imaging.hdf5']


epr_nc = []
epr_nc_density = []
nc_freqs = []
epr_thermal = []
epr_thermal_density = []
thermal_freqs = []

window = 'boxcar'
nfft = [2**8, 2**6, 2**6]
detrend = 'linear'
smooth_corr = True
sigma = [1, 2]
subtract_bias = False
many_traj = False
azimuthal_average = True

for ind, file in enumerate(files_noncontractile):
    with h5py.File(os.path.join(parentDir, 'noncontractile', file)) as f:
        print(file)
        data = np.stack((f['data']['ordermat'][:, 2:-2, 2:-2], np.transpose(f['data']['cgim'][:, 2:-2, 2:-2], axes=[0, 2, 1])))
        dx = f['params']['dx'][()]
        dt = f['params']['dt'][()]
        winsize = f['params']['winsize'][()]
        overlap = f['params']['overlap'][()]
        winspace = int(winsize - np.ceil(winsize * overlap))
        s, sdensity, freqs = fen.entropy(data, [dt, dx * winspace, dx * winspace],
                                         window=window,
                                         nfft=nfft,
                                         detrend=detrend,
                                         smooth_corr=smooth_corr,
                                         sigma=sigma,
                                         subtract_bias=subtract_bias,
                                         many_traj=many_traj,
                                         return_density=True,
                                         azimuthal_average=azimuthal_average)

        epr_nc.append(s.real)
        epr_nc_density.append(sdensity.real)
        nc_freqs.append(freqs)

for ind, file in enumerate(files_thermal):
    with h5py.File(os.path.join(parentDir, 'thermal', file)) as f:
        print(file)
        data = np.stack((f['data']['ordermat'][:, 2:-2, 2:-2], np.transpose(f['data']['cgim'][:, 2:-2, 2:-2], axes=[0, 2, 1])))
        dx = f['params']['dx'][()]
        dt = f['params']['dt'][()]
        winsize = f['params']['winsize'][()]
        overlap = f['params']['overlap'][()]
        winspace = int(winsize - np.ceil(winsize * overlap))
        s, sdensity, freqs = fen.entropy(data, [dt, dx * winspace, dx * winspace],
                                         window=window,
                                         nfft=nfft,
                                         detrend=detrend,
                                         smooth_corr=smooth_corr,
                                         sigma=sigma,
                                         subtract_bias=subtract_bias,
                                         many_traj=many_traj,
                                         return_density=True,
                                         azimuthal_average=azimuthal_average)

        epr_thermal.append(s.real)
        epr_thermal_density.append(sdensity.real)
        thermal_freqs.append(freqs)

labels = ['noncontractile'] * len(epr_nc) + ['thermal'] * len(epr_thermal)
epr = epr_nc + epr_thermal
epr_density = epr_nc_density + epr_thermal_density
freqs = nc_freqs + thermal_freqs
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


with h5py.File(os.path.join(parentDir, datetime.today().strftime('%y%m%d') + '_epr.hdf5'), 'w') as f:
    # eprgrp = f.create_group('epr')
    # eprdensitygrp = f.create_group('epr_density')
    paramsgrp = f.create_group('params')

    for ind, file in enumerate(files):
        g = f.create_group(file)
        g.attrs['experiment type'] = labels[ind]
        d = g.create_dataset('epr', data=epr[ind])
        d.attrs['description'] = 'entropy production rate'

        d2 = g.create_dataset('epr_density', data=epr_density[ind])
        d2.attrs['description'] = 'entropy production rate density'

        d3 = g.create_dataset('omega', data=freqs[ind][0])
        d3.attrs['description'] = 'temporal frequency bins for epr_density'

        d4 = g.create_dataset('k', data=freqs[ind][1])
        d4.attrs['description'] = 'spatial frequency bins for epr_density'

    for paramname in params.keys():
        p = paramsgrp.create_dataset(paramname, data=params[paramname])
        p.attrs['description'] = paramsattrs[paramname]
