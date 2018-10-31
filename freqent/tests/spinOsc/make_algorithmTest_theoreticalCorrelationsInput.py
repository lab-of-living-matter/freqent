import numpy as np
import matplotlib.pyplot as plt
import freqent.freqent as fe
import matplotlib as mpl
import os
mpl.rcParams['pdf.fonttype'] = 42
savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/frequencySpaceDissipation/tests/spinOsc/'

# Simulation environment parameters
gamma = 2e-8  # drag on 1um particle in water in kg/s
dt = 1e-3  # time step of simulation in seconds
nsteps = 1e6  # number of simulation steps
kT = 4e-9  # thermal energy in kg um^2 / s^2
r0 = np.random.randn(2)  # starting xy position in um

D = kT / gamma

# correlation lag times
maxTau = dt * nsteps
tau = np.linspace(-maxTau, maxTau, int(2 * nsteps + 1))
# correlation frequencies
omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(tau), d=dt))

# vary alpha
alphaArray = np.linspace(0, 10, 20)
k_fixed = 2
c_thry_fft = np.zeros((len(tau), 2, 2), dtype=complex)
c_thry = np.zeros((len(tau), 2, 2), dtype=complex)

sArray_alphaVary_fftDirect = np.zeros(len(alphaArray), dtype=complex)
sArray_alphaVary_fftNumeric = np.zeros(len(alphaArray), dtype=complex)

for ind, alpha in enumerate(alphaArray):

    fft_prefactor = (2 * D) / (((k_fixed + omega * 1j)**2 + alpha**2) * ((k_fixed - omega * 1j)**2 + alpha**2))
    c_thry_fft[:, 0, 0] = fft_prefactor * (k_fixed**2 + alpha**2 + omega**2)
    c_thry_fft[:, 1, 1] = fft_prefactor * (k_fixed**2 + alpha**2 + omega**2)
    c_thry_fft[:, 0, 1] = fft_prefactor * alpha * omega * 2j
    c_thry_fft[:, 1, 0] = fft_prefactor * alpha * omega * -2j

    real_prefactor = (D / k_fixed) * np.exp(-k_fixed * np.abs(tau))
    c_thry[:, 0, 0] = real_prefactor * np.cos(alpha * tau)
    c_thry[:, 1, 1] = real_prefactor * np.cos(alpha * tau)
    c_thry[:, 0, 1] = real_prefactor * -np.sin(alpha * tau)
    c_thry[:, 1, 0] = real_prefactor * np.sin(alpha * tau)
    c_thry_fftNumeric = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(c_thry, axes=0), axis=0), axes=0) * dt

    sArray_alphaVary_fftDirect[ind] = fe.entropy(c_thry_fft, sample_spacing=dt)
    sArray_alphaVary_fftNumeric[ind] = fe.entropy(c_thry_fftNumeric, sample_spacing=dt)

# vary k
alpha_fixed = 3
kArray = np.linspace(0.1, 10, 20)
c_thry_fft = np.zeros((len(tau), 2, 2), dtype=complex)
c_thry = np.zeros((len(tau), 2, 2), dtype=complex)

sArray_kVary_fftDirect = np.zeros(len(kArray), dtype=complex)
sArray_kVary_fftNumeric = np.zeros(len(kArray), dtype=complex)

for ind, k in enumerate(kArray):
    fft_prefactor = (2 * D) / (((k + omega * 1j)**2 + alpha_fixed**2) * ((k - omega * 1j)**2 + alpha_fixed**2))
    c_thry_fft[:, 0, 0] = fft_prefactor * (k**2 + alpha_fixed**2 + omega**2)
    c_thry_fft[:, 1, 1] = fft_prefactor * (k**2 + alpha_fixed**2 + omega**2)
    c_thry_fft[:, 0, 1] = fft_prefactor * alpha_fixed * omega * 2j
    c_thry_fft[:, 1, 0] = fft_prefactor * alpha_fixed * omega * -2j

    real_prefactor = (D / k) * np.exp(-k * np.abs(tau))
    c_thry[:, 0, 0] = real_prefactor * np.cos(alpha_fixed * tau)
    c_thry[:, 1, 1] = real_prefactor * np.cos(alpha_fixed * tau)
    c_thry[:, 0, 1] = real_prefactor * -np.sin(alpha_fixed * tau)
    c_thry[:, 1, 0] = real_prefactor * np.sin(alpha_fixed * tau)
    c_thry_fftNumeric = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(c_thry, axes=0), axis=0), axes=0) * dt

    sArray_kVary_fftDirect[ind] = fe.entropy(c_thry_fft, sample_spacing=dt)
    sArray_kVary_fftNumeric[ind] = fe.entropy(c_thry_fftNumeric, sample_spacing=dt)

fig, ax = plt.subplots(2, 2)
# first plot theoretical results
kArray_fineSpacing = np.linspace(kArray.min(), kArray.max(), 100)
ax[0, 0].plot(kArray_fineSpacing, 2 * alpha_fixed**2 / kArray_fineSpacing, 'k', label='theory')
ax[0, 0].plot(kArray, -sArray_kVary_fftDirect.real, 'r.', alpha=0.5, label='fft direct')
ax[0, 0].set_ylabel(r'$\langle dS/dt \rangle$')
ax[0, 0].set_title(r'$\alpha / \gamma=${a:0.2f}'.format(a=alpha_fixed))

ax[1, 0].plot(kArray_fineSpacing, 2 * alpha_fixed**2 / kArray_fineSpacing, 'k', label='theory')
ax[1, 0].plot(kArray, -sArray_kVary_fftNumeric.real, 'b.', alpha=0.5, label='fft numeric')
ax[1, 0].set_xlabel(r'$k / \gamma$')
ax[1, 0].set_ylabel(r'$\langle dS/dt \rangle$')


alphaArray_fineSpacing = np.linspace(alphaArray.min(), alphaArray.max(), 100)
ax[0, 1].plot(alphaArray_fineSpacing, 2 * alphaArray_fineSpacing**2 / k_fixed, 'k', label='theory')
ax[0, 1].plot(alphaArray, -sArray_alphaVary_fftDirect.real, 'r.', alpha=0.5, label='theoretical F.T.')
ax[0, 1].set_ylabel(r'$\langle dS/dt \rangle$')
ax[0, 1].set_title(r'$k / \gamma=${k:0.2f}'.format(k=k_fixed))
ax[0, 1].legend()

ax[1, 1].plot(alphaArray_fineSpacing, 2 * alphaArray_fineSpacing**2 / k_fixed, 'k', label='theory')
ax[1, 1].plot(alphaArray, -sArray_alphaVary_fftNumeric.real, 'b.', alpha=0.5, label='numerical F.T.')
ax[1, 1].set_xlabel(r'$\alpha / \gamma$')
ax[1, 1].set_ylabel(r'$\langle dS/dt \rangle$')

ax[1, 1].legend()

plt.tight_layout()

plt.savefig(os.path.join(savepath, 'algorithmTest_theoreticalCorrelationsInput.pdf'), format='pdf')
plt.show()
