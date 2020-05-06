import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
import os

parser = argparse.ArgumentParser()
parser.add_argument('--files', '-f', nargs='+',
                    help='absolute path to file to run test on')
parser.add_argument('--alpha', '-a', type=float,
                    help='tolerable error rate')

args = parser.parse_args()
files = args.files
a = args.alpha


def L(a):
    '''
    Threshold for making decision on arrow of time with error a
    See Roldan, et al, PRL 115, 250602 (2015)
    '''
    return np.log((1 - a) / a)


fig, ax = plt.subplots()
for file in files:
    with h5py.File(os.path.join(file, 'data.hdf5')) as d:
        t = d['data']['t_points'][:]
        mu = np.log((d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4]) /
                    (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))
        fpts = []
        for x, y in zip(d['data']['trajs'][:, 0, len(t) // 2:],
                        d['data']['trajs'][:, 1, len(t) // 2:]):
            theta = np.arctan2((y - y.mean()) / y.std(), (x - x.mean()) / x.std())
            # rescaled_x = (x - x.mean()) / x.std()
            crossing_inds = np.logical_or(np.cumsum(theta) > L(a), np.cumsum(theta) < -L(a))
            crossing_times = t[len(t) // 2:][crossing_inds] - t[len(t) // 2]

            if len(crossing_times) > 0:
                fpt = crossing_times.min()
            else:
                fpt = np.nan

            fpts.append(fpt)

        mean_fpt = np.nanmean(fpts)

        s_dot_estimate = L(a) * (1 - 2 * a) / mean_fpt
        ax.semilogy(mu, s_dot_estimate, 'ko')
        ax.semilogy(mu, d['data']['epr'][()], 'o', color='C0')
        ax.semilogy(mu, d['data']['epr_blind'][()], 'o', color='C1')

plt.show()
