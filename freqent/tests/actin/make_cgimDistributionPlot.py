import os
import numpy as np
import h5py
import freqent.freqentn as fen
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

plt.close('all')
if sys.platform == 'darwin':
    parentDir = '/Users/Danny/Dropbox/LLM_Danny/freqent/actin/'
if sys.platform == 'linux':
    parentDir = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/actin/'

bins = np.linspace(-5200, 5200, 50)
binwidth = np.diff(bins)[0]

thermalpdf = np.zeros(len(bins) - 1)
noncontractilepdf = np.zeros(len(bins) - 1)

fig, ax = plt.subplots(2, 1, sharex=True)
for file in os.listdir(os.path.join(parentDir, 'thermal')):
    with h5py.File(os.path.join(parentDir, 'thermal', file), 'r') as f:
        [n, _, _] = ax[0].hist(np.ravel(f['data']['cgim'][:] - np.nanmean(f['data']['cgim'])),
                               histtype='step',
                               color='k',
                               density=True,
                               bins=bins,
                               linewidth=1.5,
                               alpha=0.3)
        thermalpdf += n
for file in os.listdir(os.path.join(parentDir, 'noncontractile')):
    with h5py.File(os.path.join(parentDir, 'noncontractile', file), 'r') as f:
        [n, _, _] = ax[1].hist(np.ravel(f['data']['cgim'][:] - np.nanmean(f['data']['cgim'])),
                               histtype='step',
                               color='k',
                               density=True,
                               bins=bins,
                               linewidth=1.5,
                               alpha=0.3)
        noncontractilepdf += n

thermalpdf /= len(os.listdir(os.path.join(parentDir, 'thermal')))
noncontractilepdf /= len(os.listdir(os.path.join(parentDir, 'noncontractile')))

ax[0].plot(bins[:-1] + 0.5 * binwidth, thermalpdf, color='r', linewidth=2)
ax[1].plot(bins[:-1] + 0.5 * binwidth, noncontractilepdf, color='r', linewidth=2)
ax[0].text(-4750, 0.0008, 'thermal')
ax[1].text(-4750, 0.004, 'noncontractile')
ax[1].set(xlabel=r'$\rho - \langle \rho \rangle$', ylabel='pdf')
ax[0].set(ylabel='pdf', title='Coarse-grained image intensity distribution')

legend_elements = [mpl.lines.Line2D([0], [0], color='k', linewidth=2, label='data'),
                   mpl.lines.Line2D([0], [0], color='r', linewidth=2, label='mean')]

ax[0].legend(handles=legend_elements, loc='best')
ax[1].legend(handles=legend_elements, loc='best')

plt.tight_layout()

fig.savefig(os.path.join(parentDir, datetime.today().strftime('%y%m%d') + '_cgimDist.pdf'), format='pdf')

plt.show()
