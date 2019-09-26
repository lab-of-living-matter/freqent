import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import os
import freqent.freqent as fe
from datetime import datetime

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
# fig, ax = plt.subplots(figsize=(8, 6))
dataPath = '/mnt/llmStorage203/Danny/freqent/spinOsc/190709/'
savePath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/spinOsc/'

cmap = mpl.cm.get_cmap('tab10')
normalize = mpl.colors.Normalize(vmin=0, vmax=10)
colors = [cmap(normalize(value)) for value in np.arange(11)]

ndim = 2
# alphas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
alphas = [1, 5, 9]

fig, ax = plt.subplots()
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
                    ax.plot(w[w != 0], rhos[w != 0] / np.diff(w)[0],
                            color=colors[int(alpha)],
                            label=r'$\alpha$ = {a}'.format(a=alpha), alpha=0.5)
                    sig = (8 * alpha**2 * w**2) / (2 * np.pi * ((1 + 1j * w)**2 + alpha**2) * ((1 - 1j * w)**2 + alpha**2))
                    ax.plot(w, sig, '--', color=colors[int(alpha)])

ax.set(xlabel=r'$\omega$', ylabel='epr density', xlim=[-20, 20])
# ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
ax.tick_params(which='both', direction='in')

cax, _ = mpl.colorbar.make_axes(ax)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
cbar.ax.set_title(r'$\alpha$')

# fig.savefig(os.path.join(savePath, datetime.now().strftime('%y%m%d') + '_eprDensity_vs_alpha.pdf'), format='pdf')

plt.show()
