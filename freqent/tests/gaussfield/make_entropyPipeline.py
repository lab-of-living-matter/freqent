import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import freqent.freqentn as fen

plt.close('all')

if sys.platform == 'linux':
    datapath = '/mnt/llmStorage203/Danny/freqent/gaussfield/'
    savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/gaussfield'
elif sys.platform == 'darwin':
    datapath = '/Volumes/Storage/Danny/freqent/gaussfield/'
    savepath = '/Users/Danny/Dropbox/LLM_Danny/freqent/gaussfield'

files = glob(os.path.join(datapath, 'alpha*', 'data.hdf5'))

with h5py.File(files[9]) as d:
    traj = d['data']['trajs'][0, :, ::10, :]
    t = d['data']['t'][::10]
    dx = d['params']['dx'][()]
    dt = d['params']['dt'][()] * 10

sigma = [10, 3]
t_epr = t > 10
mode = 'reflect'

c, freqs = fen.corr_matrix(traj[:, t_epr, :], sample_spacing=[dt, dx])

# check size of c and freqs to make sure there are an odd number of elements
cndim = c.ndim - 2  # get number of frequency dimensions in correlation matrix
for ndim, n in enumerate(c.shape[:-2]):
    inds = [slice(None)] * cndim  # get first elements in the appropriate dimension
    singletonInds = [slice(None)] * cndim  # use this to expand selected slice for
    if n % 2 == 0:
        inds[ndim] = 0
        singletonInds[ndim] = np.newaxis
        c = np.concatenate((c, np.conj(c[tuple(inds)][tuple(singletonInds)])), axis=ndim)
        freqs[ndim] = np.concatenate((freqs[ndim], -freqs[ndim][0][np.newaxis]))

# plot raw covariance matrix
fig, ax = plt.subplots(2, 2)

ax[0, 0].pcolorfast(freqs[1], freqs[0], np.log(c[:, :, 0, 0].real))
ax[0, 0].set(ylabel=r'$\omega$', title=r'$ \log (Re [C_{00}])$', ylim=[-50, 50])

ax[0, 1].pcolorfast(freqs[1], freqs[0], c[..., 0, 1].real)
ax[0, 1].set(title=r'$Re [C_{01}]$', ylim=[-50, 50])

ax[1, 0].pcolorfast(freqs[1], freqs[0], c[..., 0, 0].imag, cmap='RdBu_r')
ax[1, 0].set(xlabel=r'$k$', ylabel=r'$\omega$', title=r'$Im [C_{00}]$', ylim=[-50, 50])

ax[1, 1].pcolorfast(freqs[1], freqs[0], c[..., 0, 1].imag, cmap='RdBu_r')
ax[1, 1].set(xlabel=r'$k$', title=r'$Im [C_{01}])$', ylim=[-50, 50])

fig.suptitle('Raw covariance matrix')
# plt.tight_layout()

# smooth c
c_smooth = fen._nd_gauss_smooth(c, stddev=sigma, mode=mode)

# plot smoothed covariance matrix
fig_smooth, ax_smooth = plt.subplots(2, 2)

ax_smooth[0, 0].pcolorfast(freqs[1], freqs[0], np.log(c_smooth[:, :, 0, 0].real))
ax_smooth[0, 0].set(ylabel=r'$\omega$', title=r'$ \log (Re [C_{00}])$', ylim=[-50, 50])

ax_smooth[0, 1].pcolorfast(freqs[1], freqs[0], c_smooth[..., 0, 1].real)
ax_smooth[0, 1].set(title=r'$Re [C_{01}]$', ylim=[-50, 50])

ax_smooth[1, 0].pcolorfast(freqs[1], freqs[0], c_smooth[..., 0, 0].imag, cmap='RdBu_r')
ax_smooth[1, 0].set(xlabel=r'$k$', ylabel=r'$\omega$', title=r'$Im [C_{00}]$', ylim=[-50, 50])

ax_smooth[1, 1].pcolorfast(freqs[1], freqs[0], c_smooth[..., 0, 1].imag, cmap='RdBu_r')
ax_smooth[1, 1].set(xlabel=r'$k$', title=r'$Im [C_{01}])$', ylim=[-50, 50])

fig_smooth.suptitle(r'Smoothed covariance matrix, $\sigma = ({w}, {k})$'.format(w=sigma[0], k=sigma[1]))
# plt.tight_layout()


# invert c
c_smooth_inv = np.linalg.inv(c_smooth)

# plot difference between c_inv(-w) - c_inv(w)
c_smooth_inv_diff = (np.flip(c_smooth_inv, axis=0) - c_smooth_inv)

fig_smooth_inv_diff, ax_smooth_inv_diff = plt.subplots(2, 2)

ax_smooth_inv_diff[0, 0].pcolorfast(freqs[1], freqs[0], c_smooth_inv_diff[..., 0, 0].real)
ax_smooth_inv_diff[0, 0].set(ylabel=r'$\omega$', title=r'$ \log (Re [C_{00}])$')

ax_smooth_inv_diff[0, 1].pcolorfast(freqs[1], freqs[0], c_smooth_inv_diff[..., 0, 1].real)
ax_smooth_inv_diff[0, 1].set(title=r'$Re [C_{01}]$')

ax_smooth_inv_diff[1, 0].pcolorfast(freqs[1], freqs[0], c_smooth_inv_diff[..., 0, 0].imag, cmap='RdBu_r')
ax_smooth_inv_diff[1, 0].set(xlabel=r'$k$', ylabel=r'$\omega$', title=r'$Im [C_{00}]$')

ax_smooth_inv_diff[1, 1].pcolorfast(freqs[1], freqs[0], c_smooth_inv_diff[..., 0, 1].imag, cmap='RdBu_r')
ax_smooth_inv_diff[1, 1].set(xlabel=r'$k$', title=r'$Im [C_{01}])$')

fig_smooth_inv_diff.suptitle(r'$C^{-1}(k, -\omega) - C^{-1}(k, \omega)$')
# plt.tight_layout()

# calculate the matrix that will be summed over to get the epr density
axes = list(range(c.ndim))
axes[-2:] = [axes[-1], axes[-2]]
# z_ratio = np.log(np.linalg.det(np.flip(c_smooth, axis=0)) / np.linalg.det(c_smooth))
# sdensity_matrix = (np.flip(c_smooth_inv, axis=0) - c_smooth_inv) * np.transpose(c_smooth, axes=axes)
sdensity_matrix = np.matmul((np.flip(c_smooth_inv, axis=0) - c_smooth_inv), c_smooth)

# plot inverted smoothed covariance matrix
fig_sdensity, ax_sdensity = plt.subplots(2, 2)

ax_sdensity[0, 0].pcolorfast(freqs[1], freqs[0], sdensity_matrix[..., 0, 0].real)
ax_sdensity[0, 0].set(ylabel=r'$\omega$', title=r'$Re [C_{00}]$', ylim=[-50, 50])

ax_sdensity[0, 1].pcolorfast(freqs[1], freqs[0], sdensity_matrix[..., 0, 1].real, cmap='RdBu_r')
ax_sdensity[0, 1].set(title=r'$Re [C_{01}]$', ylim=[-50, 50])

ax_sdensity[1, 0].pcolorfast(freqs[1], freqs[0], sdensity_matrix[..., 0, 0].imag, cmap='RdBu_r')
ax_sdensity[1, 0].set(xlabel=r'$k$', ylabel=r'$\omega$', title=r'$Im [C_{00}]$', ylim=[-50, 50])

ax_sdensity[1, 1].pcolorfast(freqs[1], freqs[0], sdensity_matrix[..., 0, 1].imag, cmap='RdBu_r')
ax_sdensity[1, 1].set(xlabel=r'$k$', title=r'$Im [C_{01}]$', ylim=[-50, 50])

fig_sdensity.suptitle(r'$(C^{-1}_{k, -\omega} - C^{-1}_{k, \omega})C_{k, \omega}$')
# plt.tight_layout()

plt.show()
