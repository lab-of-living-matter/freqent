import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from datetime import datetime
import freqent.freqentn as fen
import argparse
from scipy.ndimage import gaussian_filter

#plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
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
parser.add_argument('--datapath', '-data', type=str,
                    help='absolute path to data')
parser.add_argument('--savepath', '-save', type=str, default=None,
                    help='absolute path to save data')
args = parser.parse_args()
datapath = args.datapath
savepath = args.savepath


# alpha = 65.24
files = glob(os.path.join(datapath, 'mu*', 'data.hdf5'))
sigma = [1, 1]
hopf = 6.16

mus = np.array([float(f.split(os.path.sep)[-2][2:-7]) for f in files])
# inds = np.logical_and(mus >= 4, mus <= 7)
# files = np.array(files)[inds]
kmax = np.zeros((len(files), 10))
# mus = mus[inds]

# kmax = np.zeros(len(files))
# mus = np.zeros(len(files[inds]))


for fInd, file in enumerate(files):
    print('reading {f}'.format(f=file.split(os.path.sep)[-2]))
    #mus[fInd] = float(file.split(os.path.sep)[-2][2:-7])
    with h5py.File(file, 'r') as d:
        t_points = d['data']['t_points'][:]
        t_epr = np.where(t_points > 10)[0]
        dt = np.diff(t_points)[0]
        dx = d['params']['lCompartment'][()]
        # nCompartments = d['params']['nCompartments'][()]
        nSim = d['params']['nSim'][()]
        #mu[fInd] = np.log(d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
        #                 (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))

        s = np.zeros(nSim)
        nt, nx = d['data']['trajs'][0, 0, t_epr, :].shape
        rhos = np.zeros((nSim, nt - (nt + 1) % 2, nx - (nx + 1) % 2))

        for ind, traj in enumerate(d['data']['trajs'][..., t_epr, :]):
            s[ind], rhos[ind], w = fen.entropy(traj, sample_spacing=[dt, dx],
                                               window='boxcar', detrend='constant',
                                               smooth_corr=True, nfft=None,
                                               sigma=sigma,
                                               subtract_bias=True,
                                               many_traj=False,
                                               return_density=True)
            kmax[fInd, ind] = w[1][np.unravel_index(np.argmax(rhos[ind]), rhos[ind].shape)[1]]
            # kmax[fInd, ind] = w[1][np.argmax(rhos[ind].sum(axis=0))]
        # rhos_mean = np.mean(rhos, axis=0)
        # kmax[fInd] = w[1][np.unravel_index(np.argmax(rhos_mean), rhos_mean.shape)[1]]
        # for ind, rhos in enumerate(d['data']['rhos']):
        #     kmax[fInd, ind] = d['data']['k'][np.unravel_index(np.argmax(rhos), rhos.shape)[1]]
        # rhos_mean = np.mean(d['data']['rhos'][:], axis=0)
        # kmax[fInd] = d['data']['k'][np.unravel_index(np.argmax(rhos_mean), rhos_mean.shape)[1]]


# order kmax
kmax = kmax[np.argsort(mus)]
mus = np.sort(mus)

# first plot data below hopf bifurcation.
# Here, kmax is bimodal with peaks at +- max(k), so take abs before taking mean
pre_hopf_mean = np.mean(np.abs(kmax[mus < hopf]), axis=1)
pre_hopf_std = np.std(np.abs(kmax[mus < hopf]), axis=1)

# after the bifurcation, distribution is unimodal, so able to be described by a mean
post_hopf_mean = np.mean(kmax[mus > hopf], axis=1)
post_hopf_std = np.std(kmax[mus > hopf], axis=1)

kmax_mean = np.concatenate((pre_hopf_mean, post_hopf_mean))
kmax_std = np.concatenate((pre_hopf_std, post_hopf_std))

fig, ax = plt.subplots()
ax.plot(mus[np.logical_and(mus >= 0, mus <= 8)], kmax_mean[np.logical_and(mus >= 0, mus <= 8)], 'ko')
# ax.errorbar(mus[np.logical_and(mus >= 0, mus <= 8)], kmax_mean[np.logical_and(mus >= 0, mus <= 8)], yerr=kmax_std[np.logical_and(mus >= 0, mus <= 8)], fmt='ko')
ax.fill_between(mus[np.logical_and(mus >= 0, mus <= 8)],
                kmax_mean[np.logical_and(mus >= 0, mus <= 8)] - kmax_std[np.logical_and(mus >= 0, mus <= 8)],
                kmax_mean[np.logical_and(mus >= 0, mus <= 8)] + kmax_std[np.logical_and(mus >= 0, mus <= 8)],
                color='k', alpha=0.5)
ax.plot([hopf, hopf], [-0.05, np.pi], 'r--', label=r'$\mu_{HB}$')
ax.tick_params(which='both', direction='in')
ax.set(xlabel=r'$\Delta \mu$', ylabel=r'$\mathrm{argmax}_{q} \; \mathcal{E}$')
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
#fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_epf_kmax.pdf'), format='pdf')

plt.show()
