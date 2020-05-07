import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import h5py
import os

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

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
        for x, y in zip(d['data']['trajs'][:, 0, t > 100],
                        d['data']['trajs'][:, 1, t > 100]):
            theta = np.unwrap(np.arctan2(y - y.mean(), x - x.mean()))
            # rescaled_x = (x - x.mean()) / x.std()
            crossing_inds = np.logical_or((theta - theta[0]) / (2 * np.pi) > L(a),
                                          (theta - theta[0]) / (2 * np.pi) < -L(a))
            crossing_times = t[t > 100][crossing_inds] - 100

            if len(crossing_times) > 0:
                fpt = crossing_times.min()
            else:
                fpt = t.max() - 100

            fpts.append(fpt)

        mean_fpt = np.mean(fpts)

        s_dot_estimate = L(a) * (1 - 2 * a) / mean_fpt
        ax.semilogy(mu, s_dot_estimate, 'ko')
        ax.semilogy(mu, d['data']['epr'][()], 'o', color='C0')
        ax.semilogy(mu, d['data']['epr_blind'][()], 'o', color='C1')

legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color='C0', lw=0, label=r'$\dot{S}_{\mathrm{true}}$'),
                   mpl.lines.Line2D([0], [0], marker='o', color='C1', lw=0, label=r'$\dot{S}_{\mathrm{blind}}$'),
                   mpl.lines.Line2D([0], [0], marker='o', color='k', lw=0, label=r'$\dot{S}_{\mathrm{MFPT}}$')]

ax.legend(legend_elements, loc='upper left')

plt.show()
