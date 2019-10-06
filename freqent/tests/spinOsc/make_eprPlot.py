import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import os
import sys
from glob import glob
from datetime import datetime

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2

if sys.platform == 'linux':
    dataPath = '/mnt/llmStorage203/Danny/freqent/spinOsc/190709/'
    savePath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/spinOsc/'
elif sys.platform == 'darwin':
    dataPath = '/Volumes/Storage/Danny/freqent/spinOsc/190709/'
    savePath = '/Users/Danny/Dropbox/LLM_Danny/freqent/spinOsc/'

alphas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ndims = [2, 3, 4]
nsim = 64
s_array = np.zeros((len(ndims), len(alphas), nsim))

for file in glob(os.path.join(dataPath, '*.hdf5')):
    with h5py.File(file, 'r') as d:
        dim = d['params']['ndim'][()]
        alpha = d['params']['alpha'][()]
        s_array[ndims.index(dim), alphas.index(alpha)] = d['data']['s'][:]

fig2, ax2 = plt.subplots(figsize=(5, 5))
# ax2.errorbar(alphas, np.mean(s_array[0], axis=1), yerr=np.std(s_array[0], axis=1),
#              fmt='ko', capsize=5, lw=2, label='data')
ax2.plot(alphas, np.mean(s_array[0], axis=1), 'ko', label=r'$\dot{S}_{spectral}$')
ax2.fill_between(alphas, np.mean(s_array[0], axis=1) + np.std(s_array[0], axis=1),
                 np.mean(s_array[0], axis=1) - np.std(s_array[0], axis=1),
                 color='k', alpha=0.5)
# ax2.plot(alphas, s_array[0], 'k.')
ax2.plot(np.arange(0, 10, 0.01), 2 * np.arange(0, 10, 0.01)**2, 'r-',
         lw=2, label=r'$\dot{S}_{thry} = 2 \alpha^2$')
ax2.tick_params(which='both', direction='in')
ax2.set(xlabel=r'$\alpha$', ylabel=r'$\dot{S}$', title='2 dimensions', ylim=[-14, 260])
ax2.set_aspect(np.diff(ax2.set_xlim())[0] / np.diff(ax2.set_ylim())[0])
ax2.legend()
plt.tight_layout()
# fig2.savefig(os.path.join(savePath, datetime.now().strftime('%y%m%d') + '_eprPlot_2dim.pdf'), format='pdf')


fig3, ax3 = plt.subplots(figsize=(5, 5))
# ax3.errorbar(alphas, np.mean(s_array[1], axis=1), yerr=np.std(s_array[1], axis=1),
#              fmt='k^', capsize=5, lw=2, label='data')
ax3.plot(alphas, np.mean(s_array[1], axis=1), 'k^', label=r'$\dot{S}_{spectral}$')
ax3.fill_between(alphas, np.mean(s_array[1], axis=1) + np.std(s_array[1], axis=1),
                 np.mean(s_array[1], axis=1) - np.std(s_array[1], axis=1),
                 color='k', alpha=0.5)
# ax3.plot(alphas, s_array[1], 'k.')
ax3.plot(np.arange(0, 10, 0.01), 2 * np.arange(0, 10, 0.01)**2, 'r-',
         lw=2, label=r'$\dot{S}_{thry} = 2 \alpha^2$')
ax3.tick_params(which='both', direction='in')
ax3.set(xlabel=r'$\alpha$', ylabel=r'$\dot{S}$', title='3 dimensions', ylim=[-14, 260])
ax3.set_aspect(np.diff(ax3.set_xlim())[0] / np.diff(ax3.set_ylim())[0])
ax3.legend()
plt.tight_layout()
# fig3.savefig(os.path.join(savePath, datetime.now().strftime('%y%m%d') + '_eprPlot_3dim.pdf'), format='pdf')

fig4, ax4 = plt.subplots(figsize=(5, 5))
# ax4.errorbar(alphas, np.mean(s_array[2], axis=1), yerr=np.std(s_array[2], axis=1),
#              fmt='ks', capsize=5, lw=2, label='data')
ax4.plot(alphas, np.mean(s_array[2], axis=1), 'ks', label=r'$\dot{S}_{spectral}$')
ax4.fill_between(alphas, np.mean(s_array[2], axis=1) + np.std(s_array[2], axis=1),
                 np.mean(s_array[2], axis=1) - np.std(s_array[2], axis=1),
                 color='k', alpha=0.5)
# ax4.plot(alphas, s_array[2], 'k.')
ax4.plot(np.arange(0, 10, 0.01), 2 * np.arange(0, 10, 0.01)**2, 'r-',
         lw=2, label=r'$\dot{S}_{thry} = 2 \alpha^2$')
ax4.tick_params(which='both', direction='in')
ax4.set(xlabel=r'$\alpha$', ylabel=r'$\dot{S}$', title='4 dimensions', ylim=[-14, 260])
ax4.set_aspect(np.diff(ax4.set_xlim())[0] / np.diff(ax4.set_ylim())[0])
ax4.legend()
plt.tight_layout()
# fig4.savefig(os.path.join(savePath, datetime.now().strftime('%y%m%d') + '_eprPlot_4dim.pdf'), format='pdf')

figall, axall = plt.subplots(figsize=(5, 5))
axall.errorbar(alphas, np.mean(s_array[0], axis=1), yerr=np.std(s_array[2], axis=1),
               fmt='o', capsize=5, lw=2, alpha=0.5, label='2D')
# axall.plot(alphas, s_array[2], 'k.')
axall.errorbar(alphas, np.mean(s_array[1], axis=1), yerr=np.std(s_array[2], axis=1),
               fmt='^', capsize=5, lw=2, alpha=0.5, label='3D')
# axall.plot(alphas, s_array[2], 'k.')
axall.errorbar(alphas, np.mean(s_array[2], axis=1), yerr=np.std(s_array[2], axis=1),
               fmt='s', capsize=5, lw=2, alpha=0.5, label='4D')
# axall.plot(alphas, s_array[2], 'k.')

axall.plot(np.arange(0, 10, 0.01), 2 * np.arange(0, 10, 0.01)**2, 'r-', lw=2, label=r'$2 \alpha^2$')
axall.tick_params(which='both', direction='in')
axall.set(xlabel=r'$\alpha$', ylabel=r'$\dot{S}$', title='2,3,4 dimensions', ylim=[-14, 260])
axall.set_aspect(np.diff(axall.set_xlim())[0] / np.diff(axall.set_ylim())[0])
axall.legend()
plt.tight_layout()
# figall.savefig(os.path.join(savePath, datetime.now().strftime('%y%m%d') + '_eprPlot_alldim.pdf'), format='pdf')

plt.show()
