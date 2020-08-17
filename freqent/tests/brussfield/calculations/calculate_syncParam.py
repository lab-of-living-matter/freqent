import os
import numpy as np
import h5py
import multiprocessing
import argparse


def calc_syncParam(file):
    '''
    function to pass to multiprocessing pool to calculate
    order paramter in parallel
    '''
    print('Reading {f}'.format(f=file.split(os.path.sep)[-2]))
    with h5py.File(file) as d:
        t_points = d['data']['t_points'][:]
        t_epr = np.where(t_points > 10)[0]

        nCompartments = d['params']['nCompartments'][()]
        nSim = d['params']['nSim'][()]

        phases = np.zeros((nSim, len(t_epr), nCompartments))
        syncParam = np.zeros((nSim, len(t_epr)))
        # s = np.zeros(nSim)
        # nt, nx = d['data']['trajs'][0, 0, t_epr, :].shape
        # rhos = np.zeros((nSim, nt - (nt + 1) % 2, nx - (nx + 1) % 2))

        for traj_index, traj in enumerate(d['data']['trajs'][..., t_epr, :]):
            delta_x = traj[0] - traj[0].mean()
            delta_y = traj[1] - traj[1].mean()
            phases[traj_index] = np.unwrap(np.arctan2(delta_y, delta_x), axis=0)
            syncParam[traj_index] = np.abs(np.mean(np.exp(1j * phases[traj_index]), axis=1))

        if '/data/phases' in d:
            del d['data']['phases']
        d['data'].create_dataset('phases', data=phases)

        if '/data/syncParam' in d:
            del d['data']['syncParam']
        d['data'].create_dataset('syncParam', data=syncParam)

    return phases, syncParam


parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, nargs='+',
                    help='files to calculate entropy for')
args = parser.parse_args()

files = args.files

if len(files) < 4:
    nProcesses = len(files)
else:
    nProcesses = 4

print('Calculating synchronization order parameters...')
with multiprocessing.Pool(processes=nProcesses) as pool:
    result = pool.map(calc_syncParam, files)
print('Done.')
