import numpy as np
import matplotlib.pyplot as plt
import freqent.freqent as fe
import matplotlib as mpl
import os
import argparse
from datetime import datetime
import csv
mpl.rcParams['pdf.fonttype'] = 42


parser = argparse.ArgumentParser(description=('Test entropy calculation algorithm '
                                              'by plugging in theoretical correlation '
                                              'matrices and seeing what comes out.'))
parser.add_argument('--save', type=bool, default=False,
                    help='Boolean of whether to save outputs')
parser.add_argument('--savepath', type=str, default='',
                    help='Path to save outputs if save')
parser.add_argument('--filename', type=str, default='',
                    help='Name of image file to save at savepath')
parser.add_argument('--gamma', type=float, default=2e-8,
                    help='drag on 1um particle in water in kg/s')
parser.add_argument('--dt', type=float, default=1e-3,
                    help='time step of simulation in seconds')
parser.add_argument('--nsteps', type=int, default=int(1e4),
                    help='number of simulation steps')
parser.add_argument('--kT', type=float, default=4e-9,
                    help='thermal energy in kg um^2 / s^2')
parser.add_argument('-kFixed', '--kFixed_multiple', type=float, default=2,
                    help='Fixed spring constant of harmonic potential in units of gamma '
                         'when varying alpha')
parser.add_argument('-aFixed', '--alphaFixed_multiple', type=float, default=2,
                    help='Fixed rotational force strength in units of gamma when '
                         'varying k')
parser.add_argument('--kRange', type=float, nargs=3, default=[0.1, 10, 11],
                    help='how to vary k, in form [k_min, k_max, n]')
parser.add_argument('--alphaRange', type=float, nargs=3, default=[0, 10, 11],
                    help='how to vary k, in form [k_min, k_max, n]')

args = parser.parse_args()

g = args.gamma
D = args.kT / g

# correlation lag times
maxTau = args.dt * args.nsteps
tau = np.linspace(-maxTau, maxTau, int(2 * args.nsteps + 1))
# correlation frequencies
omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(tau), d=args.dt))

# vary alpha
alphaArray = np.linspace(*args.alphaRange) * g
k_fixed = args.kFixed_multiple * g
c_thry_fft = np.zeros((len(tau), 2, 2), dtype=complex)
c_thry = np.zeros((len(tau), 2, 2), dtype=complex)

sArray_alphaVary_fftDirect = np.zeros(len(alphaArray), dtype=complex)
sArray_alphaVary_fftNumeric = np.zeros(len(alphaArray), dtype=complex)

for ind, alpha in enumerate(alphaArray):
    c0_omega = 2 * D / ((k_fixed / g)**2 + omega**2)
    c1_omega = -4 * D * omega / ((k_fixed / g)**2 + omega**2)**2 * 1j
    c2_omega = 2 * D / ((k_fixed / g)**2 + omega**2)**2

    c_thry_fft[:, 0, 0] = c0_omega + (alpha / g)**2 * c2_omega
    c_thry_fft[:, 1, 1] = c_thry_fft[:, 0, 0]
    c_thry_fft[:, 0, 1] = -(alpha / g) * c1_omega
    c_thry_fft[:, 1, 0] = -c_thry_fft[:, 0, 1]

    c0 = (D / (k_fixed / g)) * np.exp(-(k_fixed / g) * np.abs(tau))
    c1 = ((D / (k_fixed / g)**2) * (k_fixed / g) * tau *
          np.exp(-(k_fixed / g) * np.abs(tau)))
    c2 = ((D / (2 * (k_fixed / g)**3)) * ((k_fixed / g) * np.abs(tau) + 1) *
          np.exp(-(k_fixed / g) * np.abs(tau)))

    c_thry[:, 0, 0] = c0 + (alpha / g)**2 * c2
    c_thry[:, 1, 1] = c_thry[:, 0, 0]
    c_thry[:, 0, 1] = -(alpha / g) * c1
    c_thry[:, 1, 0] = -c_thry[:, 0, 1]
    c_thry_fftNumeric = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(c_thry, axes=0),
                                                   axis=0), axes=0) * args.dt

    sArray_alphaVary_fftDirect[ind] = fe.entropy(c_thry_fft,
                                                 sample_spacing=args.dt)
    sArray_alphaVary_fftNumeric[ind] = fe.entropy(c_thry_fftNumeric,
                                                  sample_spacing=args.dt)

# vary k
alpha_fixed = args.alphaFixed_multiple * g
kArray = np.linspace(*args.kRange) * g
c_thry_fft = np.zeros((len(tau), 2, 2), dtype=complex)
c_thry = np.zeros((len(tau), 2, 2), dtype=complex)

sArray_kVary_fftDirect = np.zeros(len(kArray), dtype=complex)
sArray_kVary_fftNumeric = np.zeros(len(kArray), dtype=complex)

for ind, k in enumerate(kArray):
    c0_omega = 2 * D / ((k / g)**2 + omega**2)
    c1_omega = -4 * D * omega / ((k / g)**2 + omega**2)**2 * 1j
    c2_omega = 2 * D / ((k / g)**2 + omega**2)**2

    c_thry_fft[:, 0, 0] = c0_omega + (alpha_fixed / g)**2 * c2_omega
    c_thry_fft[:, 1, 1] = c_thry_fft[:, 0, 0]
    c_thry_fft[:, 0, 1] = -(alpha_fixed / g) * c1_omega
    c_thry_fft[:, 1, 0] = -c_thry_fft[:, 0, 1]

    c0 = (D / (k / g)) * np.exp(-(k / g) * np.abs(tau))
    c1 = (D / (k / g)**2) * (k / g) * tau * np.exp(-(k / g) * np.abs(tau))
    c2 = ((D / (2 * (k / g)**3)) * ((k / g) * np.abs(tau) + 1) *
          np.exp(-(k / g) * np.abs(tau)))

    c_thry[:, 0, 0] = c0 + (alpha_fixed / g)**2 * c2
    c_thry[:, 1, 1] = c_thry[:, 0, 0]
    c_thry[:, 0, 1] = -(alpha_fixed / g) * c1
    c_thry[:, 1, 0] = -c_thry[:, 0, 1]
    c_thry_fftNumeric = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(c_thry, axes=0),
                                                   axis=0), axes=0) * args.dt

    sArray_kVary_fftDirect[ind] = fe.entropy(c_thry_fft, sample_spacing=args.dt)
    sArray_kVary_fftNumeric[ind] = fe.entropy(c_thry_fftNumeric, sample_spacing=args.dt)


fig, ax = plt.subplots(2, 2)
# first plot theoretical results
kArray_fineSpacing = np.linspace(kArray.min(), kArray.max(), 100)
ax[0, 0].plot(kArray_fineSpacing / g,
              2 * (alpha_fixed / g)**2 / (kArray_fineSpacing / g),
              'k', label='theory')
ax[0, 0].plot(kArray / g, -sArray_kVary_fftDirect.real,
              'r.', alpha=0.5, label='fft direct')
ax[0, 0].set_ylabel(r'$\langle dS/dt \rangle$')
ax[0, 0].set_title(r'$\alpha / \gamma=${a:0.2f}'.format(a=alpha_fixed / g))

ax[1, 0].plot(kArray_fineSpacing / g,
              2 * (alpha_fixed / g)**2 / (kArray_fineSpacing / g),
              'k', label='theory')
ax[1, 0].plot(kArray / g, -sArray_kVary_fftNumeric.real,
              'b.', alpha=0.5, label='fft numeric')
ax[1, 0].set_xlabel(r'$k / \gamma$')
ax[1, 0].set_ylabel(r'$\langle dS/dt \rangle$')

alphaArray_fineSpacing = np.linspace(alphaArray.min(), alphaArray.max(), 100)
ax[0, 1].plot(alphaArray_fineSpacing / g,
              2 * (alphaArray_fineSpacing / g)**2 / (k_fixed / g),
              'k', label='theory')
ax[0, 1].plot(alphaArray / g, -sArray_alphaVary_fftDirect.real,
              'r.', alpha=0.5, label='theoretical F.T.')
ax[0, 1].set_ylabel(r'$\langle dS/dt \rangle$')
ax[0, 1].set_title(r'$k / \gamma=${k:0.2f}'.format(k=k_fixed / g))
ax[0, 1].legend()

ax[1, 1].plot(alphaArray_fineSpacing / g,
              2 * (alphaArray_fineSpacing / g)**2 / (k_fixed / g),
              'k', label='theory')
ax[1, 1].plot(alphaArray / g, -sArray_alphaVary_fftNumeric.real,
              'b.', alpha=0.5, label='numerical F.T.')
ax[1, 1].set_xlabel(r'$\alpha / \gamma$')
ax[1, 1].set_ylabel(r'$\langle dS/dt \rangle$')
ax[1, 1].legend()

plt.tight_layout()

if args.save:
    argDict = vars(args)
    argDict['datetime'] = datetime.now()

    with open(os.path.join(args.savepath, args.filename + '_params.csv'), 'w') as csv_file:
        w = csv.DictWriter(csv_file, argDict.keys())
        w.writeheader()
        w.writerow(argDict)

    fig.savefig(os.path.join(args.savepath, args.filename + '.pdf'), format='pdf')

plt.show()
