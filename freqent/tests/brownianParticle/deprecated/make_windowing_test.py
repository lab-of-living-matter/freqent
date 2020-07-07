import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from spinOscSimulation import spinOscLangevin
import freqent.freqent as fe
import matplotlib as mpl
import os
import argparse
import multiprocessing
from datetime import datetime
import csv
mpl.rcParams['pdf.fonttype'] = 42
plt.close('all')


parser = argparse.ArgumentParser(description=('Perform simulations of Brownian particles'
                                              ' in a harmonic potential plus a rotating'
                                              ' force.'))
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
parser.add_argument('--nsteps', type=int, default=int(1e5),
                    help='number of simulation steps')
parser.add_argument('--kT', type=float, default=4e-9,
                    help='thermal energy in kg um^2 / s^2')
parser.add_argument('--r0', type=float, nargs=2, default=np.random.rand(2) - 0.5,
                    help='starting xy position in um')
parser.add_argument('-k', '--k_multiple', type=float, default=2,
                    help='Spring constant of harmonic potential in units of gamma')
parser.add_argument('-a', '--alpha_multiple', type=float, default=2,
                    help='Rotational force strength in units of gamma')
parser.add_argument('--norm', '-norm', type=str, default='biased',
                    help=('Normalization of correlation function to use. Options are '
                          '"biased", "unbiased", and "none"'))
parser.add_argument('--window', '-window', type=str, default='boxcar',
                    help='Windowing function to use on trajectory data')


args = parser.parse_args()
# create object
r = spinOscLangevin(dt=args.dt, nsteps=args.nsteps, kT=args.kT, gamma=args.gamma,
                    r0=args.r0)
r.runSimulation(k=args.k_multiple * r.gamma, alpha=args.alpha_multiple * r.gamma)

window = signal.get_window(args.window, Nx=len(r.t), fftbins=False)
window /= sum(window**2)**0.5  # normalize so sum(window**2) = 1

# c, omega = fe.corr_matrix(r.pos,
#                           sample_spacing=r.dt,
#                           mode='full',
#                           method='auto',
#                           norm=args.norm,
#                           return_fft=True)

# c_windowed, omega = fe.corr_matrix(r.pos * np.array([window, ] * 2),
#                                    sample_spacing=r.dt,
#                                    mode='full',
#                                    method='auto',
#                                    norm=args.norm,
#                                    return_fft=True)

# c_thry_fft = np.zeros((int(2 * (args.nsteps) + 1), 2, 2), dtype=complex)
# c_thry_fft_prefactor = 2 * r.D / (((args.k_multiple + 1j * omega)**2 + args.alpha_multiple**2) *
#                                   ((args.k_multiple - 1j * omega)**2 + args.alpha_multiple**2))
# c_thry_fft[:, 0, 0] = c_thry_fft_prefactor * (args.alpha_multiple**2 + args.k_multiple**2 + omega**2)
# c_thry_fft[:, 1, 1] = c_thry_fft_prefactor * (args.alpha_multiple**2 + args.k_multiple**2 + omega**2)
# c_thry_fft[:, 0, 1] = c_thry_fft_prefactor * 2j * args.alpha_multiple * omega
# c_thry_fft[:, 1, 0] = -c_thry_fft_prefactor * 2j * args.alpha_multiple * omega

# fig_traj, ax_traj = plt.subplots()
# ax_traj.plot(r.t, r.pos[0], label='x')
# ax_traj.plot(r.t, r.pos[1], label='y')
# ax_traj.legend()
# ax_traj.set(xlabel='t (s)', ylabel=r'position ($\mu$m)')

# fig_cfft, ax_cfft = plt.subplots(2, 2, sharex=True)
# ax_cfft[0, 0].plot(omega, c[:, 0, 0].real, alpha=0.4, label='raw')
# ax_cfft[0, 1].plot(omega, c[:, 0, 1].imag, alpha=0.4, label='raw')
# ax_cfft[1, 0].plot(omega, c[:, 1, 0].imag, alpha=0.4, label='raw')
# ax_cfft[1, 1].plot(omega, c[:, 1, 1].real, alpha=0.4, label='raw')

# ax_cfft[0, 0].plot(omega, c_windowed[:, 0, 0].real, alpha=0.4, label=args.window)
# ax_cfft[0, 1].plot(omega, c_windowed[:, 0, 1].imag, alpha=0.4, label=args.window)
# ax_cfft[1, 0].plot(omega, c_windowed[:, 1, 0].imag, alpha=0.4, label=args.window)
# ax_cfft[1, 1].plot(omega, c_windowed[:, 1, 1].real, alpha=0.4, label=args.window)

# ax_cfft[0, 0].plot(omega, c_thry_fft[:, 0, 0].real, 'k', label='thry')
# ax_cfft[0, 1].plot(omega, c_thry_fft[:, 0, 1].imag, 'k', label='thry')
# ax_cfft[1, 0].plot(omega, c_thry_fft[:, 1, 0].imag, 'k', label='thry')
# ax_cfft[1, 1].plot(omega, c_thry_fft[:, 1, 1].real, 'k', label='thry')

# ax_cfft[0, 0].set_yscale('log')
# # ax_cfft[0, 1].set_yscale('symlog', linthreshy=c_thry_fft[:, 0, 1].max() / 10)
# # ax_cfft[1, 0].set_yscale('symlog', linthreshy=c_thry_fft[:, 1, 0].max() / 10)
# ax_cfft[1, 1].set_yscale('log')
# # ax_cfft.legend()

# plt.show()
