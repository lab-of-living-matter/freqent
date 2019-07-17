import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import freqent.freqentn as fen
mpl.rcParams['pdf.fonttype'] = 42

dataPath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/actin/noncontractile/111116_3_NC_spun_skmm.hdf5'
savePath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/figures/actin/111116_3_NC_spun_skmm_corrMat'

f = h5py.File(dataPath)

data = np.stack((f['data']['cgim'][:, 2:-2, 2:-2],
                 np.transpose(f['data']['ordermat'][:, 2:-2, 2:-2], [0, 2, 1])))

c, w = fen.corr_matrix(data,
                       sample_spacing=[f['params']['dt'][()],
                                       f['params']['dx'][()] * 16,
                                       f['params']['dx'][()] * 16],
                       nfft=[2**8, 2**6, 2**6],
                       azimuthal_average=False)

c_aa, w_aa = fen.corr_matrix(data,
                             sample_spacing=[f['params']['dt'][()],
                                             f['params']['dx'][()] * 16,
                                             f['params']['dx'][()] * 16],
                             nfft=[2**8, 2**6, 2**6],
                             azimuthal_average=True)

crr_cmap = mpl.cm.get_cmap('Blues')
crr_vmin, crr_vmax = 0, 4000000
crr_normalize = mpl.colors.Normalize(vmin=crr_vmin, vmax=crr_vmax)

cqq_cmap = mpl.cm.get_cmap('Blues')
cqq_vmin, cqq_vmax = 0, 2
cqq_normalize = mpl.colors.Normalize(vmin=cqq_vmin, vmax=cqq_vmax)

crq_cmap = mpl.cm.get_cmap('RdBu_r')
crq_vmin, crq_vmax = -2500, 2500
crq_normalize = mpl.colors.Normalize(vmin=crq_vmin, vmax=crq_vmax)


for ind, corr in enumerate(c):
    # plot image intensity correlation functions
    fig_crr, ax_crr = plt.subplots()
    ax_crr.pcolormesh(w[-2], w[-1], corr[..., 0, 0].real,
                      vmin=crr_vmin, vmax=crr_vmax, cmap=crr_cmap,
                      rasterized=True)
    ax_crr.set(title=r'$\omega = {:.3f} \ 2\pi/s$'.format(w[0][ind]),
               xlabel=r'$k_x \ [2 \pi / \mu m]$', ylabel=r'$k_y \ [2 \pi / \mu m]$')
    ax_crr.tick_params(which='both', direction='in')
    ax_crr.set_aspect('equal')

    cax_crr, _ = mpl.colorbar.make_axes(ax_crr)
    cbar_crr = mpl.colorbar.ColorbarBase(cax_crr,
                                         cmap=crr_cmap,
                                         norm=crr_normalize,
                                         extend='max')
    cbar_crr.ax.set_title(r'$\Re\{C_{\rho \rho}\}$', {'fontsize': 12})

    fig_crr.savefig(os.path.join(savePath, 'real_Crr', 'frame_{:03d}.png'.format(ind)),
                    format='png')

    # plot nematic order param correlation functions
    fig_cqq, ax_cqq = plt.subplots()
    ax_cqq.pcolormesh(w[-2], w[-1], corr[..., 1, 1].real,
                      vmin=cqq_vmin, vmax=cqq_vmax, cmap=cqq_cmap,
                      rasterized=True)
    ax_cqq.set(title=r'$\omega = {:.3f} \ 2\pi/s$'.format(w[0][ind]),
               xlabel=r'$k_x \ [2 \pi / \mu m]$',
               ylabel=r'$k_y \ [2 \pi / \mu m]$')
    ax_cqq.tick_params(which='both', direction='in')
    ax_cqq.set_aspect('equal')

    cax_cqq, _ = mpl.colorbar.make_axes(ax_cqq)
    cbar_cqq = mpl.colorbar.ColorbarBase(cax_cqq,
                                         cmap=cqq_cmap,
                                         norm=cqq_normalize,
                                         extend='max')
    cbar_cqq.ax.set_title(r'$\Re\{C_{q q}\}$', {'fontsize': 12})

    fig_cqq.savefig(os.path.join(savePath, 'real_Cqq', 'frame_{:03d}.png'.format(ind)),
                    format='png')

    # Plot cross correlation
    fig_crq, ax_crq = plt.subplots()
    ax_crq.pcolormesh(w[-2], w[-1], corr[..., 0, 1].imag,
                      vmin=crq_vmin, vmax=crq_vmax, cmap=crq_cmap,
                      rasterized=True)
    ax_crq.set(title=r'$\omega = {:.3f} \ 2\pi/s$'.format(w[0][ind]),
               xlabel=r'$k_x \ [2 \pi / \mu m]$', ylabel=r'$k_y \ [2 \pi / \mu m]$')
    ax_crq.tick_params(which='both', direction='in')
    ax_crq.set_aspect('equal')

    cax_crq, _ = mpl.colorbar.make_axes(ax_crq)
    cbar_crq = mpl.colorbar.ColorbarBase(cax_crq,
                                         cmap=crq_cmap,
                                         norm=crq_normalize,
                                         extend='both')
    cbar_crq.ax.set_title(r'$\Im\{C_{\rho q}\}$', {'fontsize': 12})

    fig_crq.savefig(os.path.join(savePath, 'imag_Crq', 'frame_{:03d}.png'.format(ind)), format='png')

    plt.close('all')


# plot azimuthal average of Re C_rr
fig_crr, ax_crr = plt.subplots()
ax_crr.pcolormesh(w_aa[1], w_aa[0], c_aa[..., 0, 0].real,
                  cmap=crr_cmap, vmin=crr_vmin, vmax=crr_vmax,
                  rasterized=True)
ax_crr.set(xlabel=r'$k_r \ [2 \pi / \mu m]$', ylabel=r'$\omega \ [2 \pi / s]$')
ax_crr.tick_params(which='both', direction='in')
ax_crr.set_aspect(np.diff(ax_crr.set_xlim())[0] / np.diff(ax_crr.set_ylim())[0])
cax_crr, _ = mpl.colorbar.make_axes(ax_crr)
cbar_crr = mpl.colorbar.ColorbarBase(cax_crr,
                                     cmap=crr_cmap,
                                     norm=crr_normalize,
                                     extend='max')
cbar_crr.ax.set_title(r'$\Re\{C_{\rho \rho}\}$', {'fontsize': 12})
fig_crr.savefig(os.path.join(savePath, 'Crr_real.pdf'), format='pdf')

# plot azimuthal average of Re C_qq
fig_cqq, ax_cqq = plt.subplots()
ax_cqq.pcolormesh(w_aa[1], w_aa[0], c_aa[..., 1, 1].real,
                  cmap=cqq_cmap, vmin=cqq_vmin, vmax=cqq_vmax,
                  rasterized=True)
ax_cqq.set(xlabel=r'$k_r \ [2 \pi / \mu m]$', ylabel=r'$\omega \ [2 \pi / s]$')
ax_cqq.tick_params(which='both', direction='in')
ax_cqq.set_aspect(np.diff(ax_cqq.set_xlim())[0] / np.diff(ax_cqq.set_ylim())[0])
cax_cqq, _ = mpl.colorbar.make_axes(ax_cqq)
cbar_cqq = mpl.colorbar.ColorbarBase(cax_cqq,
                                     cmap=cqq_cmap,
                                     norm=cqq_normalize,
                                     extend='max')
cbar_cqq.ax.set_title(r'$\Re\{C_{q q}\}$', {'fontsize': 12})
fig_cqq.savefig(os.path.join(savePath, 'Cqq_real.pdf'), format='pdf')

# plot azimuthal average of Im C_rq
fig_crq, ax_crq = plt.subplots()
ax_crq.pcolormesh(w_aa[1], w_aa[0], c_aa[..., 0, 1].imag,
                  cmap=crq_cmap, vmin=crq_vmin, vmax=crq_vmax,
                  rasterized=True)
ax_crq.set(xlabel=r'$k_r \ [2 \pi / \mu m]$', ylabel=r'$\omega \ [2 \pi / s]$')
ax_crq.tick_params(which='both', direction='in')
ax_crq.set_aspect(np.diff(ax_crq.set_xlim())[0] / np.diff(ax_crq.set_ylim())[0])
cax_crq, _ = mpl.colorbar.make_axes(ax_crq)
cbar_crq = mpl.colorbar.ColorbarBase(cax_crq,
                                     cmap=crq_cmap,
                                     norm=crq_normalize,
                                     extend='both')
cbar_crq.ax.set_title(r'$\Im\{C_{\rho q}\}$', {'fontsize': 12})
fig_crq.savefig(os.path.join(savePath, 'Crq_imag.pdf'), format='pdf')
