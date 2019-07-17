import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
mpl.rcParams['pdf.fonttype'] = 42

plt.close('all')
imagePath = '/mnt/llmStorage203/Ian/FluctuationCorrelations/2016-addExpts/111116_2_imaging/561-700sq-StackReg.tif'
dataPath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/actin/thermal/111116_2_imaging.hdf5'
savePath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/actin/111116_2_imaging_cgimOrder'

f = h5py.File(dataPath)
dt = f['params']['dt'][()]
# im = pims.open(imagePath)[0]
# mi, ma = im.min(), im.max()

# winrad = int(np.floor(f['params']['winsize'][()] / 2))
# winspace = int(f['params']['winsize'][()] - np.ceil(f['params']['winsize'][()] * f['params']['overlap']))
# sample_rows = np.arange(winrad, im.shape[-2] - winrad, winspace)
# sample_cols = np.arange(winrad, im.shape[-1] - winrad, winspace)
# xx, yy = np.meshgrid(sample_rows, sample_cols)

order_cmap = mpl.cm.get_cmap('viridis')
order_norm = mpl.colors.Normalize(vmin=0, vmax=1)
order_colors = [order_cmap(order_norm(value)) for value in np.linspace(0, 1)]

cgim_min, cgim_max = f['data']['cgim'][:].min(), f['data']['cgim'][:].max()
cgim_cmap = mpl.cm.get_cmap('Greys_r')
cgim_norm = mpl.colors.Normalize(vmin=cgim_min, vmax=cgim_max)
cgim_colors = [cgim_cmap(cgim_norm(value)) for value in np.linspace(cgim_min, cgim_max)]

ind = 7
for p, q in zip(f['data']['cgim'][7:8, 2:-2, 2:-2], f['data']['ordermat'][7:8, 2:-2, 2:-2]):
    fig, (cgim_ax, order_ax) = plt.subplots(1, 2, figsize=(14, 5))

    cgim_ax.pcolormesh(p, cmap=cgim_cmap, vmin=cgim_min, vmax=cgim_max, rasterized=True)
    cgim_ax.axis('off'), cgim_ax.set_aspect('equal')
    cgim_ax.plot([30, 36], [4, 4], color='w', linewidth=3)
    cgim_ax.text(30, 2, r'$10 \ \mu m$', color='w', size=12)
    cgim_ax.text(30, 35, '{0} s'.format(ind * dt), color='w', size=15)
    # cgim_ax.set(title=r'$\rho$')
    cgim_cax, _ = mpl.colorbar.make_axes(cgim_ax)
    cgim_cbar = mpl.colorbar.ColorbarBase(cgim_cax, cmap=cgim_cmap, norm=cgim_norm)
    cgim_cbar.ax.set_title(r'$\rho$', {'fontsize': 15})
    cgim_cbar.ax.tick_params(direction='in', labelsize='large')

    order_ax.pcolormesh(q.T, cmap=order_cmap, vmin=0, vmax=1, rasterized=True)
    order_ax.axis('off'), order_ax.set_aspect('equal')
    # order_ax.set(title=r'$q$')
    order_cax, _ = mpl.colorbar.make_axes(order_ax)
    order_cbar = mpl.colorbar.ColorbarBase(order_cax, cmap=order_cmap, norm=order_norm)
    order_cbar.ax.set_title(r'$q$', {'fontsize': 15})
    order_cbar.ax.tick_params(direction='in', labelsize='large')

    fig.savefig(os.path.join(savePath, 'frame_{:03d}.pdf'.format(ind)), format='pdf')
    ind += 1

    # plt.show()
    plt.close('all')


