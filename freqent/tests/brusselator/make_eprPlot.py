import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
from datetime import datetime
import freqent.freqent as fe

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2

if sys.platform == 'linux':
    parentFolder = '/mnt/llmStorage203/Danny/brusselatorSims/reactionsOnly/191026/'
    saveFolder = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations/'
elif sys.platform == 'darwin':
    parentFolder = '/Volumes/Storage/Danny/brusselatorSims/reactionsOnly/191026/'
    saveFolder = '/Users/Danny/Dropbox/LLM_Danny/freqent/brusselator/stochastic_simulations/'

folders = glob(os.path.join(parentFolder, 'alpha*'))
mu = np.log(np.asarray([float(a.split(os.path.sep)[-1].split('_')[0][5:]) for a in folders]))

epr_spectral = np.zeros((len(mu), 50))
epr_blind = np.zeros(len(mu))
epr = np.zeros(len(mu))

for fInd, f in enumerate(folders):
    with h5py.File(os.path.join(f, 'data.hdf5'), 'r') as d:
        delta_mu = np.log((d['params']['B'][()] * d['params']['rates'][2] * d['params']['rates'][4] /
                    (d['params']['C'][()] * d['params']['rates'][3] * d['params']['rates'][5])))
        if delta_mu > 5 and delta_mu < 5.8:
            sigma = 10
            t_factor = 10
            t_points = d['data']['t_points'][:]
            t_epr = np.where(t_points > 10)[0]  # only calculate epr after t = 1000
            dt = np.diff(t_points)[0]
            nSim = d['params']['nSim'][()]
            s = np.zeros(nSim)
            rhos = np.zeros((nSim, len(t_epr[::t_factor])))
            for ind, traj in enumerate(d['data']['trajs'][..., t_epr[::t_factor]]):
                epr_spectral[fInd, ind] = fe.entropy(traj, sample_spacing=dt * t_factor,
                                                     sigma=sigma, return_density=False)
        elif delta_mu > 5.8:
            sigma = 5
            t_factor = 10
            t_points = d['data']['t_points'][:]
            t_epr = np.where(t_points > 10)[0]  # only calculate epr after t = 1000
            dt = np.diff(t_points)[0]
            nSim = d['params']['nSim'][()]
            s = np.zeros(nSim)
            rhos = np.zeros((nSim, len(t_epr[::t_factor])))
            for ind, traj in enumerate(d['data']['trajs'][..., t_epr[::t_factor]]):
                epr_spectral[fInd, ind] = fe.entropy(traj, sample_spacing=dt * t_factor,
                                                     sigma=sigma, return_density=False)
        else:
            epr_spectral[fInd] = d['data']['s'][:]

        epr[fInd] = d['data']['epr'][()]
        epr_blind[fInd] = d['data']['epr_blind'][()]

epr = epr[np.argsort(mu)]
epr_blind = epr_blind[np.argsort(mu)]
epr_spectral = epr_spectral[np.argsort(mu)]
mu = np.sort(mu)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(mu, epr, 'o', label=r'$\dot{S}_{true}$')
ax.plot(mu, epr_blind, 'o', label=r'$\dot{S}_{blind}$')
# ax.errorbar(alphas, np.mean(epr_spectral, axis=1),
#             yerr=np.std(epr_spectral, axis=1), fmt='ko',
#             label=r'$\dot{S}_{spectral}$', capsize=5)

ax.plot(mu[mu < 5], np.mean(epr_spectral, axis=1)[mu < 5],
        'o', color='0.0', label=r'$\sigma = {s}$'.format(s=dt * t_factor * 200))
ax.fill_between(mu[mu < 5],
                np.mean(epr_spectral, axis=1)[mu < 5] + np.std(epr_spectral, axis=1)[mu < 5],
                np.mean(epr_spectral, axis=1)[mu < 5] - np.std(epr_spectral, axis=1)[mu < 5],
                color='0.0', alpha=0.5)

ax.plot(mu[np.logical_and(mu > 5, mu < 5.8)], np.mean(epr_spectral, axis=1)[np.logical_and(mu > 5, mu < 5.8)],
        'o', color='0.4', label=r'$\sigma = {s}$'.format(s=dt * t_factor * 10))
ax.fill_between(mu[np.logical_and(mu > 5, mu < 5.8)],
                np.mean(epr_spectral, axis=1)[np.logical_and(mu > 5, mu < 5.8)] + np.std(epr_spectral, axis=1)[np.logical_and(mu > 5, mu < 5.8)],
                np.mean(epr_spectral, axis=1)[np.logical_and(mu > 5, mu < 5.8)] - np.std(epr_spectral, axis=1)[np.logical_and(mu > 5, mu < 5.8)],
                color='0.4', alpha=0.5)

ax.plot(mu[mu > 5.8], np.mean(epr_spectral, axis=1)[mu > 5.8],
        'o', color='0.6', label=r'$\sigma = {s}$'.format(s=dt * t_factor * 5))
ax.fill_between(mu[mu > 5.8],
                np.mean(epr_spectral, axis=1)[mu > 5.8] + np.std(epr_spectral, axis=1)[mu > 5.8],
                np.mean(epr_spectral, axis=1)[mu > 5.8] - np.std(epr_spectral, axis=1)[mu > 5.8],
                color='0.6', alpha=0.5)

# ax.plot(np.repeat(np.sort(alphas), 50), np.ravel(epr_spectral[np.argsort(alphas), :]), 'k.', alpha=0.5)

ax.set(xlabel=r'$\Delta \mu$', ylabel=r'$\dot{S}$')
# ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
ax.set(yscale='log', xscale='linear')
ax.tick_params(which='both', direction='in')
ax.legend(loc='lower right')
plt.tight_layout()

# fig.savefig(os.path.join(saveFolder, datetime.now().strftime('%y%m%d') + '_eprPlot.pdf'), format='pdf')
plt.show()
