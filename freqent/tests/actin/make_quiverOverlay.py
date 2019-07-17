import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pims
import os
mpl.rcParams['pdf.fonttype'] = 42

plt.close('all')
imagePath = '/mnt/llmStorage203/Ian/FluctuationCorrelations/2016-addExpts/111116_2_imaging/561-700sq-StackReg.tif'
dataPath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/actin/thermal/111116_2_imaging.hdf5'
savePath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/actin/111116_2_imaging_quiverOverlay'

f = h5py.File(dataPath)
dt = f['params']['dt'][()]
im = pims.open(imagePath)[0]
mi, ma = im.min(), im.max()

winrad = int(np.floor(f['params']['winsize'][()] / 2))
winspace = int(f['params']['winsize'][()] - np.ceil(f['params']['winsize'][()] * f['params']['overlap']))
sample_rows = np.arange(winrad, im.shape[-2] - winrad, winspace)
sample_cols = np.arange(winrad, im.shape[-1] - winrad, winspace)
xx, yy = np.meshgrid(sample_rows, sample_cols)

cmap = mpl.cm.get_cmap('viridis')
normalize = mpl.colors.Normalize(vmin=0, vmax=1)
colors = [cmap(normalize(value)) for value in np.linspace(0, 1)]

ind = 0
for j, theta, q in zip(im[:1], f['data']['anglemat'][:1] * np.pi / 180, f['data']['ordermat'][:1]):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.pcolormesh(j, cmap='Greys_r', vmin=mi, vmax=ma, rasterized=True)
    a = ax.quiver(xx, yy, np.cos(theta), np.sin(theta), q, cmap=cmap, headwidth=1, headlength=1, pivot='mid')
    ax.axis('off')
    ax.text(550, 650, '{0} s'.format(ind * dt), color='w', size=20)
    ax.plot([550, 650], [35, 35], 'w', linewidth=3)
    ax.text(570, 10, r'$10 \ \mu m$', color='w', size=12)
    ax.quiverkey(a, 0.03, 0.05, 2, r'$\vec{{n}}$', labelpos='S', coordinates='axes', color='w', labelcolor='w')
    ax.set_aspect('equal')
    cax, _ = mpl.colorbar.make_axes(ax)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
    cbar.ax.set_title(r'$q$', {'fontsize': 15})
    cbar.ax.tick_params(direction='in', labelsize='large')
    fig.savefig(os.path.join(savePath, 'frame_{:03d}.png'.format(ind)), format='png')
    ind += 1

    plt.close('all')
