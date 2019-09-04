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

files_nc = ['051612_skmmII_noncontr.hdf5',
            '071813_2.hdf5',   # small muscle myosin, all others skmm
            '111116_2_NC_spun_skmm.hdf5',
            '111116_3_NC_spun_skmm.hdf5',
            '112916_1_NC_spun_647_skmm.hdf5',
            '112916_3_NC_spun_647_skmm.hdf5',
            '120216_2_NC_spun_647_skmm.hdf5',
            '120216_3_NC_spun_647_skmm.hdf5',
            '120916_3-2_NC_spun_647_skmm_0pt75mMATP.hdf5',
            '121516_1_NC_unspun_647_skmm.hdf5',
            '121516_4_NC_unspun_647_skmm.hdf5',
            '053112_smmm_noncontr_Fig4.hdf5',        #
            '052012_sm.hdf5',                        #
            '053112_2_sm_nocontract.hdf5',           # These data sets
            '060312_sm_nocontract.hdf5',             # all have dt=30s
            '060412_2_nmmII_1to300x_noncontr.hdf5',  # so may not see
            '071813_2_nmmII.hdf5',                   # relevant time-
            '071913_2_nmmII_Fig4.hdf5',              # scales
            '729130_sm_nocontract.hdf5']             #

files_thermal = ['112916_2_imaging.hdf5',
                 '121516_3_imaging.hdf5',
                 '111116_3_imaging.hdf5',
                 '111116_2_imaging.hdf5',
                 '120216_2_imaging.hdf5',
                 '120916_1_imaging.hdf5',
                 '112916_3_imaging.hdf5',
                 '121516_2_imaging.hdf5',
                 '121516_4_imaging.hdf5',
                 '120216_2_noskmm_thermalFactinNetwork.hdf5',
                 '120216_3_imaging.hdf5']


window = 'boxcar'
nfft = None
detrend = 'constant'
smooth_corr = True
sigma = [2, 1, 1]
subtract_bias = True
many_traj = False
azimuthal_average = False
tile_data = False

noise_reps = 1
shuffle_reps = 100

# actual data
epr_nc = np.zeros(len(files_nc))
epr_nc_density = []
nc_freqs = []
dt_nc = np.zeros(len(files_nc))
dx_nc = np.zeros(len(files_nc))
winspace_nc = np.zeros(len(files_nc))

epr_thermal = np.zeros(len(files_thermal))
epr_thermal_density = []
thermal_freqs = []
dt_thermal = np.zeros(len(files_thermal))
dx_thermal = np.zeros(len(files_thermal))
winspace_thermal = np.zeros(len(files_thermal))

# noise and shuffled data
epr_nc_noise = np.zeros((len(files_nc), noise_reps))
epr_nc_shuffle = np.zeros((len(files_nc), shuffle_reps))

epr_thermal_noise = np.zeros((len(files_thermal), noise_reps))
epr_thermal_shuffle = np.zeros((len(files_thermal), shuffle_reps))


for ind, file in enumerate(files_nc):
    with h5py.File(os.path.join(parentDir, 'noncontractile', file)) as f:
        print(file)
        data = np.stack((f['data']['ordermat'][:, 2:-2, 2:-2], np.transpose(f['data']['cgim'][:, 2:-2, 2:-2], axes=[0, 2, 1])))
        # data = np.stack((f['data']['vt'][1:, 2:-2, 2:-2],
        #                  f['data']['ordermat'][1:, 2:-2, 2:-2],
        #                  np.transpose(f['data']['cgim'][1:, 2:-2, 2:-2], axes=[0, 2, 1])))

        if tile_data:
            data = np.concatenate((np.flip(data, axis=-1), data), axis=-1)  # flip over positive y-axis to fill upper half-plane
            data = np.concatenate((np.flip(data, axis=-2), data), axis=-2)  # flip over x-axis to fill all four quadrants
            print(data.shape)
            nfft = [int(d - (1 - np.mod(d, 2))) for d in data.shape[1:]]
            print(nfft)

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

        epr_nc[ind] = s.real
        epr_nc_density.append(sdensity.real)
        nc_freqs.append(freqs)
        dt_nc[ind] = dt
        dx_nc[ind] = dx
        winspace_nc[ind] = (winspace)

        for ii in range(noise_reps):
            noise0 = np.std(data[0]) * np.random.randn(*data[0].shape) + np.mean(data[0])
            noise1 = np.std(data[1]) * np.random.randn(*data[1].shape) + np.mean(data[1])
            noise = np.stack((noise0, noise1))
            s_noise, sdensity_noise, freqs_noise = fen.entropy(noise, [dt, dx * winspace, dx * winspace],
                                                               window=window,
                                                               nfft=nfft,
                                                               detrend=detrend,
                                                               smooth_corr=smooth_corr,
                                                               sigma=sigma,
                                                               subtract_bias=subtract_bias,
                                                               many_traj=many_traj,
                                                               return_density=True,
                                                               azimuthal_average=azimuthal_average)
            epr_nc_noise[ind, ii] = s_noise.real

        for ii in range(shuffle_reps):
            tInds1 = np.arange(data.shape[1])
            tInds2 = np.arange(data.shape[1])
            np.random.shuffle(tInds1), np.random.shuffle(tInds2)

            shuffled_data = np.stack((data[0, tInds1], data[1, tInds1]))
            s_shuffle, sdensity_shuffle, freqs_shuffle = fen.entropy(shuffled_data,
                                                                     [dt, dx * winspace, dx * winspace],
                                                                     window=window,
                                                                     nfft=nfft,
                                                                     detrend=detrend,
                                                                     smooth_corr=smooth_corr,
                                                                     sigma=sigma,
                                                                     subtract_bias=subtract_bias,
                                                                     many_traj=many_traj,
                                                                     return_density=True,
                                                                     azimuthal_average=azimuthal_average)
            epr_nc_shuffle[ind, ii] = s_shuffle.real

for ind, file in enumerate(files_thermal):
    with h5py.File(os.path.join(parentDir, 'thermal', file)) as f:
        print(file)
        data = np.stack((f['data']['ordermat'][:, 2:-2, 2:-2], np.transpose(f['data']['cgim'][:, 2:-2, 2:-2], axes=[0, 2, 1])))
        # data = np.stack((f['data']['vt'][1:, 2:-2, 2:-2],
        #                  f['data']['ordermat'][1:, 2:-2, 2:-2],
        #                  np.transpose(f['data']['cgim'][1:, 2:-2, 2:-2], axes=[0, 2, 1])))

        if tile_data:
            data = np.concatenate((np.flip(data, axis=-1), data), axis=-1)  # flip over positive y-axis to fill upper half-plane
            data = np.concatenate((np.flip(data, axis=-2), data), axis=-2)  # flip over x-axis to fill all four quadrants
            print(data.shape)
            nfft = [int(d - (1 - np.mod(d, 2))) for d in data.shape[1:]]
            print(nfft)

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

        epr_thermal[ind] = s.real
        epr_thermal_density.append(sdensity.real)
        thermal_freqs.append(freqs)
        dt_thermal[ind] = dt
        dx_thermal[ind] = dx
        winspace_thermal[ind] = (winspace)

        for ii in range(noise_reps):
            noise0 = np.std(data[0]) * np.random.randn(*data[0].shape) + np.mean(data[0])
            noise1 = np.std(data[1]) * np.random.randn(*data[1].shape) + np.mean(data[1])
            noise = np.stack((noise0, noise1))
            s_noise, sdensity_noise, freqs_noise = fen.entropy(noise, [dt, dx * winspace, dx * winspace],
                                                               window=window,
                                                               nfft=nfft,
                                                               detrend=detrend,
                                                               smooth_corr=smooth_corr,
                                                               sigma=sigma,
                                                               subtract_bias=subtract_bias,
                                                               many_traj=many_traj,
                                                               return_density=True,
                                                               azimuthal_average=azimuthal_average)
            epr_thermal_noise[ind, ii] = s_noise.real

        for ii in range(shuffle_reps):
            tInds1 = np.arange(data.shape[1])
            tInds2 = np.arange(data.shape[1])
            np.random.shuffle(tInds1), np.random.shuffle(tInds2)

            shuffled_data = np.stack((data[0, tInds1], data[1, tInds1]))
            s_shuffle, sdensity_shuffle, freqs_shuffle = fen.entropy(shuffled_data,
                                                                     [dt, dx * winspace, dx * winspace],
                                                                     window=window,
                                                                     nfft=nfft,
                                                                     detrend=detrend,
                                                                     smooth_corr=smooth_corr,
                                                                     sigma=sigma,
                                                                     subtract_bias=subtract_bias,
                                                                     many_traj=many_traj,
                                                                     return_density=True,
                                                                     azimuthal_average=azimuthal_average)
            epr_thermal_shuffle[ind, ii] = s_shuffle.real

# put everything together
labels = ['noncontractile'] * len(epr_nc) + ['thermal'] * len(epr_thermal)
epr = np.concatenate((epr_nc, epr_thermal))
epr_density = epr_nc_density + epr_thermal_density
epr_noise = np.concatenate((epr_nc_noise, epr_thermal_noise))
epr_shuffle = np.concatenate((epr_nc_shuffle, epr_thermal_shuffle))

freqs = nc_freqs + thermal_freqs
files = [file.split('.')[0] for file in files_nc + files_thermal]
dt = np.concatenate((dt_nc, dt_thermal))

if nfft is None:
    nfft = ''

params = {'window': window,
          'nfft': nfft,
          'detrend': detrend,
          'smooth_corr': smooth_corr,
          'sigma': sigma,
          'subtract_bias': subtract_bias,
          'many_traj': many_traj,
          'shuffle_reps': shuffle_reps,
          'noise_reps': noise_reps}

paramsattrs = {'window': 'window used in calculating fft of signal',
               'nfft': 'size of nfft window if zero-padding wanted',
               'detrend': 'how to detrend signal',
               'smooth_corr': 'boolean to either smooth correlation functions or not',
               'sigma': 'size of gaussian used to smooth in each dimension',
               'subtract_bias': 'boolean of whether to subtract bias off epr estimate',
               'many_traj': 'boolean of whether passing multiple trajectories into freqent.freqentn.entropy()',
               'shuffle_reps': 'number of times to shuffle data',
               'noise_reps': 'number of times random noise data is generated'}


with h5py.File(os.path.join(parentDir, datetime.today().strftime('%y%m%d') + '_epr.hdf5'), 'w') as f:
    # eprgrp = f.create_group('epr')
    # eprdensitygrp = f.create_group('epr_density')
    paramsgrp = f.create_group('params')

    for ind, file in enumerate(files):
        file_group = f.create_group(file)
        file_group.attrs['experiment type'] = labels[ind]

        d = file_group.create_dataset('epr', data=epr[ind])
        d.attrs['description'] = 'entropy production rate'

        d2 = file_group.create_dataset('epr_density', data=epr_density[ind])
        d2.attrs['description'] = 'entropy production rate density'

        d3 = file_group.create_dataset('omega', data=freqs[ind][0])
        d3.attrs['description'] = 'temporal frequency bins for epr_density'

        d4 = file_group.create_dataset('k', data=freqs[ind][1])
        d4.attrs['description'] = 'spatial frequency bins for epr_density'

        d5 = file_group.create_dataset('epr_shuffle', data=epr_shuffle[ind])
        d5.attrs['description'] = 'epr from shuffled data'

        d6 = file_group.create_dataset('epr_noise', data=epr_noise[ind])
        d6.attrs['description'] = 'epr from noise'

        d7 = file_group.create_dataset('dt', data=dt[ind])
        d7.attrs['description'] = 'time step between frames in seconds'

    for paramname in params.keys():
        p = paramsgrp.create_dataset(paramname, data=params[paramname])
        p.attrs['description'] = paramsattrs[paramname]
