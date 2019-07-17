import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import freqent.freqentn as fen
mpl.rcParams['pdf.fonttype'] = 42

dataPath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/actin/noncontractile/111116_3_NC_spun_skmm.hdf5'
savePath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/actin/111116_3_NC_spun_skmm_corrMat'
savePath_aziavg = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/actin/111116_3_NC_spun_skmm_corrMatAziAvg'

f = h5py.File(dataPath)

data = np.stack((f['data']['cgim'][:, 2:-2, 2:-2],
                 np.transpose(f['data']['ordermat'][:, 2:-2, 2:-2], [0, 2, 1])))

c, w = fen.corr_matrix(data,
                       sample_spacing=[f['params']['dt'][()],
                                       f['params']['dx'][()] * 16,
                                       f['params']['dx'][()] * 16],
                       nfft=[2**8, 2**6, 2**6],
                       azimuthal_average=False)

cmap = mpl.cm.get_cmap('Blues')
normalize = mpl.colors.Normalize(vmin=0, vmax=4000000)

c_aa, w_aa = fen.corr_matrix(data,
                             sample_spacing=[f['params']['dt'][()],
                                             f['params']['dx'][()] * 16,
                                             f['params']['dx'][()] * 16],
                             nfft=[2**8, 2**6, 2**6],
                             azimuthal_average=True)


for ind, corr in enumerate(c):
    fig, ax = plt.subplots()
    ax.pcolormesh(w[-2], w[-1], corr[..., 0, 0].real,
                  vmin=0, vmax=4000000, cmap=cmap,
                  rasterized=True)
    ax.set(title=r'$\omega = {:.3f} \ 2\pi/s$'.format(w[0][ind]),
           xlabel=r'$k_x \ [2 \pi / \mu m]$', ylabel=r'$k_y \ [2 \pi / \mu m]$')
    ax.tick_params(which='both', direction='in')
    ax.set_aspect('equal')

    cax, _ = mpl.colorbar.make_axes(ax)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize, extend='max')
    cbar.ax.set_title(r'$\Re\{C_{\rho \rho}\}$', {'fontsize': 12})

    fig.savefig(os.path.join(savePath, 'frame_{:03d}.png'.format(ind)), format='png')

    plt.close('all')

fig, ax = plt.subplots()
ax.pcolormesh(w_aa[1], w_aa[0], c_aa[..., 0, 0].real,
              cmap=cmap, vmin=0, vmax=4000000,
              rasterized=True)
ax.set(xlabel=r'$k_r \ [2 \pi / \mu m]$', ylabel=r'$\omega \ [2 \pi / s]$')
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
cax, _ = mpl.colorbar.make_axes(ax)
cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize, extend='max')
cbar.ax.set_title(r'$\Re\{C_{\rho \rho}\}$', {'fontsize': 12})
fig.savefig(os.path.join(savePath_aziavg, 'C00_real.pdf'), format='pdf')
