import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datetime import datetime
import matplotlib as mpl
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
    datapath = '/mnt/llmStorage203/Danny/brusselatorSims/fieldSims/191028/'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brussfield/'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/brusselatorSims/fieldSims/191028/'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/brussfield/'

folders = np.asarray(glob(os.path.join(datapath, 'alpha*')))
mu = np.log(np.asarray([float(a.split(os.path.sep)[-1].split('_')[0][5:]) for a in folders]))

epr_spectral = np.zeros((len(mu), 10))
epr_blind = np.zeros(len(mu))
epr = np.zeros(len(mu))

sigma_array = [[]]

for fInd, f in enumerate(folders[list(np.argsort(mu))]):
    with h5py.File(os.path.join(f, 'data.hdf5'), 'r') as d:
        delta_mu = np.log((d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
                          (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5])))

        if delta_mu > 5.65 or abs(delta_mu) < 1:
            if np.isclose(delta_mu, 5.7):
                sigma = [50, 4]
            elif np.isclose(delta_mu, 5.8):
                sigma = [25, 3]
            elif np.isclose(delta_mu, 5.9):
                sigma = [10, 1]
            elif np.isclose(delta_mu, 6.0):
                sigma = [5, 0.5]
            elif np.isclose(delta_mu, 6.1):
                sigma = [2, 0.2]
            elif np.isclose(delta_mu, 6.2):
                sigma = [0.5, 0.05]
            # elif np.isclose(delta_mu, 6.3):
            #     sigma = [0.5, 0.05]
            elif delta_mu > 6.3:
                sigma = [0.5, 0.05]
            elif abs(delta_mu) < 1 and abs(delta_mu) > 0.5:
                sigma = [100, 10]
            elif abs(delta_mu) < 0.5:
                sigma = [200, 20]


            t_points = d['data']['t_points'][:]
            t_epr = np.where(t_points > 20)[0]
            dt = np.diff(t_points)[0]
            dx = d['params']['lCompartment'][()]
            nSim = d['params']['nSim'][()]
            # s = np.zeros(nSim)
            for ind, traj in enumerate(d['data']['trajs'][..., t_epr, :]):
                epr_spectral[fInd, ind] = fen.entropy(traj, sample_spacing=[dt, dx],
                                                      window='boxcar', detrend='constant',
                                                      smooth_corr=True, nfft=None,
                                                      sigma=sigma,
                                                      subtract_bias=True,
                                                      many_traj=False,
                                                      return_density=False)
        else:
            epr_spectral[fInd] = d['data']['s'][:]

        epr[fInd] = d['data']['epr'][()]
        epr_blind[fInd] = d['data']['epr_blind'][()]

        if fInd == 0:
            lCompartment = d['params']['lCompartment'][()]
            nCompartments = d['params']['nCompartments'][()]
            # T = d['data']['t_points'][:].max()
            # dw = np.diff(d['data']['omega'][:])[0]
            # dk = np.diff(d['data']['k'][:])[0]
            # k_max = d['data']['k'][:].max()
            # w_max = d['data']['omega'][:].max()
            # sigma = d['params']['sigma'][:]

V = lCompartment * nCompartments
mu = np.sort(mu)


fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(mu[mu <= 8], epr[mu <= 8] / V, 'o', label=r'$\dot{S}_{true}$')
ax.plot(mu[mu <= 8], epr_blind[mu <= 8] / V, 'o', label=r'$\dot{S}_{blind}$')
# ax.errorbar(alphas, np.mean(epr_spectral, axis=1) - bias, yerr=np.std(epr_spectral, axis=1), fmt='ko', label='epr_spectral', capsize=5)

ax.plot(mu[mu <= 8], np.mean(epr_spectral[mu <= 8], axis=1), 'ko', label=r'$\hat{\dot{S}}$')
ax.fill_between(mu[[mu <= 8]][np.argsort(mu[mu <= 8])],
                np.mean(epr_spectral, axis=1)[mu <= 8][np.argsort(mu[mu <= 8])] + np.std(epr_spectral[mu <= 8], axis=1)[np.argsort(mu[mu <= 8])],
                np.mean(epr_spectral[mu <= 8], axis=1)[np.argsort(mu[mu <= 8])] - np.std(epr_spectral[mu <= 8], axis=1)[np.argsort(mu[mu <= 8])],
                color='k', alpha=0.5)

ax.set(xlabel=r'$\Delta \mu$', ylabel=r'$\dot{S}$')
# ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
ax.tick_params(which='both', direction='in')
ax.set(yscale='log')
ax.legend(loc='lower right')
plt.tight_layout()

fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_eprPlot.pdf'), format='pdf')
plt.show()
