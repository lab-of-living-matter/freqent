import os
import sys
from glob import glob
import numpy as np
import h5py
import freqent.freqent as fe
import multiprocessing
import argparse


def calc_epr_spectral(file):
    '''
    function to pass to multiprocessing pool to calculate epr in parallel
    '''
    print('Reading {f}'.format(f=file.split(os.path.sep)[-2]))
    with h5py.File(os.path.join(file, 'data.hdf5')) as d:
        t_points = d['data']['t_points'][:]
        t_epr = np.where(t_points > 10)[0]  # only calculate epr after t = 10
        dt = np.diff(t_points)[0]
        nSim = d['params']['nSim'][()]

        s = np.zeros(nSim)
        rhos = np.zeros((nSim, len(t_epr[::t_factor])))

        for ind, traj in enumerate(d['data']['trajs'][..., t_epr[::t_factor]]):
            s[ind], rhos[ind], w = fe.entropy(traj, sample_spacing=dt * t_factor,
                                              sigma=sigma, return_density=True)

        if '/data/s' in d:
            del d['data']['s']
        d['data'].create_dataset('s', data=s)

        if '/data/s_density' in d:
            del d['data']['s_density']
        d['data'].create_dataset('s_density', data=rhos)

        if '/data/omega' in d:
            del d['data']['omega']
        d['data'].create_dataset('omega', data=w)

        if '/params/sigma' in d:
            del d['params']['sigma']
        d['params'].create_dataset('sigma', data=sigma)

        if '/params/t_factor' in d:
            del d['params']['t_factor']
        d['params'].create_dataset('t_factor', data=t_factor)
    return s, rhos, w


parser = argparse.ArgumentParser()
parser.add_argument('--files', '-f', type=str, nargs='+',
                    help='files to calculate entropy for')
parser.add_argument('--sigma', '-sig', type=float, default=10,
                    help='size of Gaussian to smooth correlation functions with')
parser.add_argument('--t_factor', '-dt', type=float, default=1,
                    help='spacing of subsampling')
args = parser.parse_args()

sigma = args.sigma
t_factor = args.t_factor

if len(args.files) <= 4:
    nProcessses = len(args.files)
else:
    nProcessses = 4

print('Calculating eprs...')
with multiprocessing.Pool(processes=nProcessses) as pool:
    result = pool.map(calc_epr_spectral, args.files)
print('Done.')
