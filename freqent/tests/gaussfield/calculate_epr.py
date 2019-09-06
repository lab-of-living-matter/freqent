import glob
import os
import numpy as np
import h5py
import freqent.freqentn as fen
import multiprocessing


def calculate_epr(f):
    print(f)
    with h5py.File(os.path.join(f, 'data.hdf5'), 'a') as d:
        nSim = d['params']['nSim'][()]
        T = d['params']['nsteps'][()] * d['params']['dt'][()]  # total time of simulations
        # starting simulation step after a total time of 10 has passed, used to assume steady state
        steady_state_start = int(d['params']['nsteps'][()] / (T / 10))

        epr_array = np.zeros(nSim)
        rhos_array = np.zeros((nSim,
                               int((d['params']['nsteps'][()] - steady_state_start + 1) / tFactor) + 1,
                               d['params']['nsites'][()] + 1))

        for ind, traj in enumerate(d['data']['trajs'][..., steady_state_start::tFactor, :]):
            try:
                s, rhos, w = fen.entropy(traj,
                                         sample_spacing=[d['params']['dt'][()] * tFactor, d['params']['dx'][()]],
                                         detrend='constant',
                                         many_traj=False,
                                         return_density=True,
                                         sigma=[10, 3],
                                         subtract_bias=True)
                epr_array[ind] = s
                rhos_array[ind] = rhos
            except np.linalg.LinAlgError:
                epr_array[ind] = np.nan
                rhos = np.zeros((int((d['params']['nsteps'][()] - steady_state_start + 1) / tFactor) + 1,
                                d['params']['nsites'][()] + 1))
                rhos[:] = np.nan
                rhos_array[ind] = rhos
            else:
                pass
            finally:
                pass

        # save data to hdf5 file
        if '/data/epr_spectral' in d:
            del d['data']['epr_spectral']

        if '/data/epr' in d:
            d['data']['epr'][...] = epr_array
        else:
            epr_dset = d['data'].create_dataset('epr', data=epr_array)
            epr_dset.attrs['description'] = 'epr of each trajectory'

        if '/data/epr_density' in d:
            d['data']['epr_density'][...] = rhos_array
        else:
            epr_density_dset = d['data'].create_dataset('epr_density', data=rhos_array)
            epr_density_dset.attrs['description'] = 'epr density of each trajectory'

        if '/data/omega' in d:
            d['data']['omega'][...] = w[0]
        else:
            omega_dset = d['data'].create_dataset('omega', data=w[0])
            omega_dset.attrs['description'] = 'temporal frequency bins for epr density'

        if '/data/k' in d:
            d['data']['k'][...] = w[1]
        else:
            k_dset = d['data'].create_dataset('k', data=w[1])
            k_dset.attrs['description'] = 'spatial frequency bins for epr density'

    return epr_array, rhos_array, w


parentDir = '/mnt/llmStorage203/Danny/freqent/gaussfield/'
folders = glob.glob(os.path.join(parentDir, 'alpha*'))
tFactor = 10

alphas = np.array([float(f.split('alpha')[1].split('_')[0]) for f in folders])
epr = np.zeros(len(alphas))

with multiprocessing.Pool(processes=4) as pool:
    result = pool.map(calculate_epr, folders)
