import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

myoData = pd.read_excel('/media/daniel/storage11/Dropbox/LLM_Danny/freqent/actin/myosinData/myosinData.xlsx', usecols="A:M")

myoData['epr_shuffle_mean'] = epr_nc_shuffle_mean
myoData['epr_shuffle_std'] = epr_nc_shuffle_std
myoData['epr'] = epr_nc

skmmEPR = myoData.where(myoData['isoform'] == 'skmm')['epr'].dropna().values
skmm_myoDensity = myoData.where(myoData['isoform'] == 'skmm')['meanMyoDensity_perMicron'].dropna().values
skmmEPR_shuffle_mean = myoData.where(myoData['isoform'] == 'skmm')['epr_shuffle_mean'].dropna().values
skmmEPR_shuffle_std = myoData.where(myoData['isoform'] == 'skmm')['epr_shuffle_std'].dropna().values
x = np.linspace(skmm_myoDensity.min(), skmm_myoDensity.max())

fig, ax = plt.subplots()
ax.plot(skmm_myoDensity[np.where(skmm_myoDensity)], skmmEPR[np.where(skmm_myoDensity)], 'o', label='real data', color='C0')
ax.errorbar(skmm_myoDensity[np.where(skmm_myoDensity)], skmmEPR_shuffle_mean[np.where(skmm_myoDensity)],
            yerr=skmmEPR_shuffle_std[np.where(skmm_myoDensity)], fmt='o', label='shuffled data', color='C1', capsize=7)

m, b, r_value, p_value, std_err = stats.linregress(skmm_myoDensity[np.where(skmm_myoDensity)],
                                                   skmmEPR[np.where(skmm_myoDensity)])
m_shuffle, b_shuffle, r_value_shuffle, p_value_shuffle, std_err_shuffle = stats.linregress(skmm_myoDensity[np.where(skmm_myoDensity)],
                                                                                           skmmEPR_shuffle_mean[np.where(skmm_myoDensity)])

p_value_shuffle
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
ax.tick_params(which='both', direction='in')
ax.plot(x, m * x + b, '--', color='C0')
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
ax.plot(x, m_shuffle * x + b_shuffle, '--', color='C1')
ax.legend()
ax.text(0.08, 0.00075, 'p={p:0.3f}'.format(p=p_value), color='C0')
ax.text(0.08, 0.00070, 'p={p:0.3f}'.format(p=p_value_shuffle), color='C1')
ax.set(xlabel=r'myosin density $[\#/\mu m]$', ylabel=r'epr $[s^{-1} \mu m ^{-2}]$')

# fig.savefig('/media/daniel/storage11/Dropbox/LLM_Danny/freqent/actin/myosinData/190819_myo_vs_epr_skmm_real+shuffle.pdf', format='pdf')
# fig.savefig('/media/daniel/storage11/Dropbox/LLM_Danny/freqent/actin/myosinData/190819_myo_vs_epr_skmm_real+shuffle.png', format='png')
