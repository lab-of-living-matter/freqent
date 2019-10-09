import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import os
import sys
import freqent.freqent as fe
from datetime import datetime

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2

if sys.platform == 'linux':
    dataPath = '/mnt/llmStorage203/Danny/freqent/spinOsc/190709/'
    savePath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/spinOsc/'
elif sys.platform == 'darwin':
    dataPath = '/Volumes/Storage/Danny/freqent/spinOsc/190709/'
    savePath = '/Users/Danny/Dropbox/LLM_Danny/freqent/figures/spinOsc/'

cmap = mpl.cm.get_cmap('tab10')
normalize = mpl.colors.Normalize(vmin=-0.5, vmax=9.5)
colors = [cmap(normalize(value)) for value in np.arange(11)]

ndim = 2
alphas = [1, 3, 9]
# alphas = [1, 5, 9]

fig, ax = plt.subplots(figsize=(5.5, 5))
for file in os.listdir(dataPath):
    if file.endswith('.hdf5'):
        if int(file.split('dim')[1][0]) == ndim:
            with h5py.File(os.path.join(dataPath, file), 'r') as f:
                nt = len(f['data']['t'][:])
                alpha = f['params']['alpha'][()]
                if alpha in alphas:
                    print(file)
                    s, rhos, w = fe.entropy(f['data']['trajs'][..., nt // 2:], f['params']['dt'][()],
                                            sigma=f['params']['sigma_array'][0], return_density=True)
                    ax.semilogy(w[w != 0], rhos[w != 0].real / np.diff(w)[0],
                                color=colors[int(alpha)],
                                label=r'$\alpha$ = {a}'.format(a=alpha), alpha=0.5)
                    sig = (8 * alpha**2 * w**2) / (2 * np.pi * ((1 + 1j * w)**2 + alpha**2) * ((1 - 1j * w)**2 + alpha**2))
                    ax.semilogy(w[w != 0], sig[w != 0], '--', color=colors[int(alpha)], lw=2)

ax.set(xlabel=r'$\omega$', ylabel='epr density',
       xlim=[-20, 20], ylim=[1e-5, 1e2],
       xticks=[-20, -10, 0, 10, 20])
# ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
ax.tick_params(which='both', direction='in')
ax.legend()

# cax, _ = mpl.colorbar.make_axes(ax)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize, ticks=alphas)
# cbar.ax.set_title(r'$\alpha$')
# cbar.ax.tick_params(which='both', direction='in')

# fig.savefig(os.path.join(savePath, datetime.now().strftime('%y%m%d') + '_eprDensity_vs_alpha.pdf'), format='pdf')

plt.show()
