import numpy as np
import matplotlib.pyplot as plt
import freqent.freqentn as fen
import dynamicstructurefactor.sqw as sqw
from itertools import product
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/frequencySpaceDissipation/tests/freqentn_tests/'
plt.close('all')

# Set up parameters
xmax = 6 * np.pi  # total distance in physical units
ymax = 6 * np.pi
tmax = 100
nx = 250  # total number of pixels across
ny = 250
nt = 100
dx = xmax / nx  # sampling spacing
dy = ymax / ny
dt = tmax / nt

xArr = np.linspace(-xmax / 2, xmax / 2, nx)
yArr = np.linspace(-ymax / 2, ymax / 2, ny)
tArr = np.linspace(0, tmax, nt)

# Set up grid in real space, remembering to multiply by the
# sampling periods in time and space
tt, xx, yy = np.meshgrid(tArr, xArr, yArr, indexing='ij')

# Spatial and temporal frequency (in radians/length or time)
lambda0 = np.pi / 6
k0 = 2 * np.pi / lambda0
T0 = 5
w0 = 2 * np.pi / T0

lambda1 = np.pi / 6
k1 = 2 * np.pi / lambda1
T1 = 5
w1 = 2 * np.pi / T1

# Center offset
x0 = 0 * dx
y0 = 0 * dy
x1 = 0 * dx
y1 = 0 * dy

# phase difference
phi = 1 * np.pi / 2

# Function and its power spectrum
r0 = ((xx - x0)**2 + (yy - y0)**2)**0.5
r1 = ((xx - x1)**2 + (yy - y1)**2)**0.5
r0t = np.cos(k0 * r0 - w0 * tt)
r1t = np.cos(k1 * r1 - w1 * tt + phi)

data = np.zeros((2, *r0t.shape))
data[0] = r0t
data[1] = r1t

c, freqs = fen.corr_matrix(data, sample_spacing=[dt, dx, dy])
c = fen._nd_gauss_smooth(c, stddev=[1, 2, 2])

idx_array = list(product(np.arange(2), repeat=2))

figReal, axReal = plt.subplots(2, 2, sharex=True, sharey=True)
figImag, axImag = plt.subplots(2, 2, sharex=True, sharey=True)

for idx in idx_array:
    aziAvg_real, kr_real = sqw.azimuthal_average_3D(c[..., idx[0], idx[1]].real,
                                                    dx=2 * np.pi / xmax)
    aziAvg_imag, kr_imag = sqw.azimuthal_average_3D(c[..., idx[0], idx[1]].imag,
                                                    dx=2 * np.pi / xmax)
    axReal[idx[0], idx[1]].pcolormesh(kr_real, freqs[0], aziAvg_real, vmin=-1, vmax=15)
    axImag[idx[0], idx[1]].pcolormesh(kr_imag, freqs[0], aziAvg_imag, vmin=-0.3, vmax=0.3)

axReal[1, 0].set(xlabel=r'$k$ (rad/um)', ylabel=r'$\omega$ (rad/s)')
axReal[0, 0].set(ylabel=r'$\omega$ (rad/s)')
axReal[1, 1].set(xlabel=r'$k$ (rad/um)')
figReal.suptitle(r'$\Re[\langle r_i(\mathbf{{k}}, \omega) r_j^*(\mathbf{{k}}, \omega) \rangle]$')
# figReal.savefig(os.path.join(savepath, 'sphericalWaveCSD_Real_smoothed_sigma1.pdf'), format='pdf')

axImag[1, 0].set(xlabel=r'$k$ (rad/um)', ylabel=r'$\omega$ (rad/s)')
axImag[0, 0].set(ylabel=r'$\omega$ (rad/s)')
axImag[1, 1].set(xlabel=r'$k$ (rad/um)')
figImag.suptitle(r'$\Im[\langle r_i(\mathbf{{k}}, \omega) r_j^*(\mathbf{{k}}, \omega) \rangle]$')
# figImag.savefig(os.path.join(savepath, 'sphericalWaveCSD_Imag_smoothed_sigma1.pdf'), format='pdf')

plt.show()
