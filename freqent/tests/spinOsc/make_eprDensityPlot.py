import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib as mpl
import os
import freqent.freqent as fe
from datetime import datetime

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
fig, ax = plt.subplots(figsize=(8, 6))
parentDir = '/mnt/llmStorage203/Danny/freqent/spinOsc/190709/'
savePath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/spinOsc/'

cmap = mpl.cm.get_cmap('cividis')
normalize = mpl.colors.Normalize(vmin=0, vmax=10)
colors = [cmap(normalize(value)) for value in np.arange(11)]

ndim = 2

fig, ax = plt.subplots()
for file in os.listdir(parentDir):
    if file.endswith('.hdf5'):
        if int(file.split('dim')[1][0]) == ndim:
            print(file)
            with h5py.File(os.path.join(parentDir, file), 'r') as f:
                nt = len(f['data']['t'][:])
                s, rhos, w = fe.entropy(f['data']['trajs'][..., nt // 2:], f['params']['dt'][()],
                                        sigma=f['params']['sigma'][0], return_density=True)
                ax.plot(w[w != 0], rhos[w != 0],
                        color=colors[int(f['params']['alpha'][()])],
                        label=r'$\alpha$ = {a}'.format(a=f['params']['alpha'][()]))

ax.set(xlabel=r'$\omega$', ylabel='epr density', xlim=[-20, 20])
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
ax.tick_params(which='both', direction='in')

cax, _ = mpl.colorbar.make_axes(ax)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
cbar.ax.set_title(r'$\alpha$')

fig.savefig(os.path.join(savePath, datetime.now().strftime('%y%m%d') + '_eprDensity_vs_alpha.pdf'), format='pdf')
