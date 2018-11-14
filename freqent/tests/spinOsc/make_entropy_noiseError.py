import numpy as np
import matplotlib.pyplot as plt
import freqent.freqent as fe
import scipy as sp

gamma = 2e-8
alpha = 10
k = 2
dt = 1e-3
nsteps = int(1e5)
kT = 4e-9
D = kT / gamma

# get lag time vector
maxTau = dt * nsteps
tau = np.linspace(-maxTau, maxTau, 2 * nsteps + 1)

omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(2 * nsteps + 1, d=dt))
c_thry_fft = np.zeros((int(2 * (nsteps) + 1), 2, 2), dtype=complex)

c_thry_fft_prefactor = 2 * D / (((k + 1j * omega)**2 + alpha**2) * (((k - 1j * omega)**2 + alpha**2)))
c_thry_fft[:, 0, 0] = c_thry_fft_prefactor * (alpha**2 + k**2 + omega**2)
c_thry_fft[:, 1, 1] = c_thry_fft_prefactor * (alpha**2 + k**2 + omega**2)
c_thry_fft[:, 0, 1] = c_thry_fft_prefactor * 2j * alpha * omega
c_thry_fft[:, 1, 0] = -c_thry_fft_prefactor * 2j * alpha * omega


scaleArray = np.logspace(-10, -1, 50)
nRepeats = 3
sArray = np.zeros(len(scaleArray) * nRepeats, dtype=complex)
distArray = np.zeros(len(scaleArray) * nRepeats)
# simulate noise as gaussian distributed noise. Take into account that
# diagonal elements are symmetric and off-diagonal elements are antisymmetric
# Noise is also biggest around zero, gets smaller at ends. multiply by Gaussian
gauss = sp.stats.norm(loc=0, scale=5)
for ind, scale in enumerate(scaleArray):
    for ii in range(nRepeats):
        noise_sym_xx = np.random.normal(loc=0, scale=scale, size=int((c_thry_fft.shape[0] + 1) / 2))
        noise_sym_xx = np.concatenate((np.flipud(noise_sym_xx)[:-1], noise_sym_xx))

        noise_sym_yy = np.random.normal(loc=0, scale=scale, size=int((c_thry_fft.shape[0] + 1) / 2))
        noise_sym_yy = np.concatenate((np.flipud(noise_sym_yy)[:-1], noise_sym_yy))

        noise_asym_xy = np.random.normal(loc=0, scale=scale, size=int((c_thry_fft.shape[0] - 1) / 2))
        noise_asym_xy = np.concatenate((-np.flipud(noise_asym_xy), np.zeros(1), noise_asym_xy))

        noise_asym_yx = np.random.normal(loc=0, scale=scale, size=int((c_thry_fft.shape[0] - 1) / 2))
        noise_asym_yx = np.concatenate((-np.flipud(noise_asym_yx), np.zeros(1), noise_asym_yx))

        # for now only add noise to real parts of diagonal elements and imaginary part
        # of off-diagonal elements. The others are supposed to be zero. Change later.
        noise = np.zeros(c_thry_fft.shape, dtype=complex)
        noise[:, 0, 0] += noise_sym_xx
        noise[:, 1, 1] += noise_sym_yy
        noise[:, 0, 1] += noise_asym_xy * 1j
        noise[:, 1, 0] += noise_asym_yx * 1j

        c_thry_fft_noisy = c_thry_fft + noise

        distArray[ind * nRepeats + ii] = np.sum(abs(noise))
        sArray[ind * nRepeats + ii] = fe.entropy(c_thry_fft_noisy, sample_spacing=dt)

fig, ax = plt.subplots()
sThry = 2 * alpha**2 / k
ax.loglog(np.repeat(scaleArray, nRepeats) / np.max(c_thry_fft), np.abs(-sArray - sThry) * dt, 'k.')
ax.set_xlabel(r'$max(C) / \sqrt{\langle \xi^2\rangle} \ [SNR]$')
ax.set_ylabel(r'$\vert \dot{S} - \dot{S}_{thry}\vert$')

plt.tight_layout()
plt.show()
