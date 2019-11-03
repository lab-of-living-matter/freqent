import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from datetime import datetime
import freqent.freqentn as fen

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2

if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/fieldSims/191028'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/brussfield/'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/fieldSims/191028'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/figures/brussfield/'

# alpha = 65.24
files = glob(os.path.join(datapath, 'alpha*', 'data.hdf5'))
sigma = [20, 5]
hopf = 6.16

kmax = np.zeros((len(files), 10))
alphas = np.zeros(len(files))

for fInd, file in enumerate(files):
    print('reading {f}'.format(f=file.split(os.path.sep)[-2]))
    with h5py.File(file, 'r') as d:
        t_points = d['data']['t_points'][:]
        t_epr = np.where(t_points > 50)[0]
        dt = np.diff(t_points)[0]
        dx = d['params']['lCompartment'][()]
        # nCompartments = d['params']['nCompartments'][()]
        nSim = d['params']['nSim'][()]
        alphas[fInd] = (d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
                        (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5]))

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
        # for ind, rhos in enumerate(d['data']['rhos']):
        #     kmax[fInd, ind] = d['data']['k'][np.unravel_index(np.argmax(rhos), rhos.shape)[1]]

# order kmax
kmax = kmax[np.argsort(alphas)] * 100
mu = np.sort(np.log(alphas))

# first plot data below hopf bifurcation.
# Here, kmax is bimodal with peaks at +- max(k), so take abs before taking mean
pre_hopf_mean = np.mean(np.abs(kmax[mu < hopf]), axis=1)
pre_hopf_std = np.std(np.abs(kmax[mu < hopf]), axis=1)

# after the bifurcation, distribution is unimodal, so able to be described by a mean
post_hopf_mean = np.mean(kmax[mu > hopf], axis=1)
post_hopf_std = np.std(kmax[mu > hopf], axis=1)

kmax_mean = np.concatenate((pre_hopf_mean, post_hopf_mean))
kmax_std = np.concatenate((pre_hopf_std, post_hopf_std))

fig, ax = plt.subplots()
# ax.errorbar(alphas, kmax_mean, kmax_std, fmt='ko')
ax.errorbar(mu[mu >= 0], kmax_mean[mu >= 0], yerr=kmax_std[mu >= 0], fmt='ko')
# ax.fill_between(mu[mu >= 0],
#                 kmax_mean[mu >= 0] - kmax_std[mu >= 0],
#                 kmax_mean[mu >= 0] + kmax_std[mu >= 0],
#                 color='k', alpha=0.5)
ax.plot([hopf, hopf], [-0.05, np.pi], 'r--', label=r'$\mu_{HB}$')
ax.tick_params(which='both', direction='in')
ax.set(xlabel=r'$\Delta \mu$', ylabel=r'$\mathrm{argmax}_{q} \; \rho_{\dot{s}}$')
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
# fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_eprDensity_kmax.pdf'), format='pdf')

plt.show()
