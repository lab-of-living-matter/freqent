import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import sys
import os
import h5py
from datetime import datetime
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if sys.platform == 'darwin':
    parentDir = '/Users/Danny/Dropbox/LLM_Danny/freqent/actin/'
if sys.platform == 'linux':
    parentDir = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/actin/'

myo_data = pd.read_excel(os.path.join(parentDir, 'myosinData/myosinData.xlsx'), usecols="A:M")
relevant_myo_data = myo_data.where(myo_data['isoform'] == 'skmm').dropna()
myo_density = relevant_myo_data['meanMyoDensity_perMicron'].values

# extract epr data
epr = np.zeros(len(relevant_myo_data))
epr_shuffle = np.zeros((len(relevant_myo_data), 100))
epr_shuffle_mean = np.zeros(len(relevant_myo_data))
epr_shuffle_std = np.zeros(len(relevant_myo_data))
with h5py.File(os.path.join(parentDir, datetime.today().strftime('%y%m%d') + '_epr.hdf5')) as f:
    for ind, (index, row) in enumerate(relevant_myo_data.iterrows()):
        epr[ind] = f[row['Experiment']]['epr'][()]
        epr_shuffle[ind] = f[row['Experiment']]['epr_shuffle'][:]
        epr_shuffle_mean[ind] = np.mean(epr_shuffle[ind])
        epr_shuffle_std[ind] = np.std(epr_shuffle[ind])

# plot the data
fig, ax = plt.subplots()
ax.errorbar(myo_density[np.where(myo_density)], epr_shuffle_mean[np.where(myo_density)],
            yerr=epr_shuffle_std[np.where(myo_density)], fmt='o', label='shuffled data', color='C1', capsize=7)
for ind, e in enumerate(epr_shuffle[np.where(myo_density)]):
    ax.plot(np.random.randn(100) * 0.0003 + np.repeat(myo_density[np.where(myo_density)][ind], 100), e, '.', color='C1', alpha=0.5)

ax.plot(myo_density[np.where(myo_density)], epr[np.where(myo_density)], 'o', label='real data', color='C0')

# perform and plot linear regressions
real_regression = stats.linregress(myo_density[np.where(myo_density)],
                                   epr[np.where(myo_density)])
shuffle_regression = stats.linregress(myo_density[np.where(myo_density)],
                                      epr_shuffle_mean[np.where(myo_density)])

x = np.array([myo_density[np.where(myo_density)].min() * 0.9,
              myo_density[np.where(myo_density)].max() * 1.1])
ax.plot(x, real_regression.slope * x + real_regression.intercept, '--', color='C0')
ax.plot(x, shuffle_regression.slope * x + shuffle_regression.intercept, '--', color='C1')

# plotting parameters
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
ax.tick_params(which='both', direction='in')
ax.legend()
ax.text(0.08, 0.00075, 'p={p:0.3f}'.format(p=real_regression.pvalue), color='C0')
ax.text(0.08, 0.00070, 'p={p:0.3f}'.format(p=shuffle_regression.pvalue), color='C1')
ax.set(xlabel='myosin density  [um^{-2}]', ylabel='epr [s^{-1}  um^{-2}]')

# fig.savefig(os.path.join(parentDir, 'myosinData', datetime.today().strftime('%y%m%d') + '_myo_vs_epr_skmm_real+shuffle.pdf'), format='pdf')
# fig.savefig(os.path.join(parentDir, 'myosinData', datetime.today().strftime('%y%m%d') + '_myo_vs_epr_skmm_real+shuffle.png'), format='png')

plt.show()
