import numpy as np
import matplotlib.pyplot as plt
from spinOscSimulation import spinOscLangevin
import freqent.freqent as fe
import matplotlib as mpl
import os
import argparse
import multiprocessing
from datetime import datetime
import csv
from astropy.convolution import Gaussian1DKernel, convolve
# import scipy.signal as signal

mpl.rcParams['pdf.fonttype'] = 42
plt.close('all')

parser = argparse.ArgumentParser(description=('Perform simulations of Brownian particles'
                                              ' in a harmonic potential plus a rotating'
                                              ' force. See how the error in entropy production'
                                              ' scales with size of Welch segments'))
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
parser.add_argument('--nsim', type=int, default=50,
                    help='number of simulations to run')
parser.add_argument('-k', '--k_multiple', type=float, default=2,
                    help='Spring constant of harmonic potential in units of gamma')
parser.add_argument('-a', '--alpha_multiple', type=float, default=2,
                    help='Rotational force strength in units of gamma')
parser.add_argument('--seed_type', type=str, default='time',
                    help=('a string to decide what seed to use when generating '
                          'trajectories. use "time" to use current microsecond or '
                          '"input" to directly input the seed'))
parser.add_argument('--seed_input', '-seed', type=float, default=None,
                    help='if seed_type=input, what the seed explicitly is')
parser.add_argument('--window', '-window', type=str, default='boxcar',
                    help='Window function to multiple data by')
parser.add_argument('--noverlap', default=None,
                    help='Number of overlap between Welch segments. Defaults to nperseg/2')
parser.add_argument('--nfft', default=None,
                    help='Number of points in fourier transform. Defaults to nperseg')
parser.add_argument('--detrend', default=False,
                    help=('Whether or how ot detrend the data. If False, no detrending.'
                          ' If "constant", remove mean from each Welch segment. If'
                          ' "linear", perform least-squares fit and subtract line.'))
parser.add_argument('--padded', default=False,
                    help='Pad the data with zeros to be integer number of Welch segments')
parser.add_argument('--nperseg_array', nargs=3, default=[None, None, None],
                    help=('Array of number of points in each Welch segment. Becomes input '
                          'to np.logspace(start, stop, num, base=2). Defaults to '
                          'np.logspace(256, nsteps, 5, base=2)'))

args = parser.parse_args()
# create object
r = spinOscLangevin(dt=args.dt, nsteps=args.nsteps, kT=args.kT, gamma=args.gamma,
                    r0=args.r0)

# forcing parameters
# equilibriationFrames = int(nsteps/2);
k = args.k_multiple * r.gamma
alpha = args.alpha_multiple * r.gamma

# create array of npersegs. Make ints, and add on a None to see what happens
# when no averaging is done.
nperseg_array = np.logspace(float(args.nperseg_array[0]),
                            float(args.nperseg_array[1]),
                            num=int(args.nperseg_array[2]),
                            base=2).astype(int)
nperseg_array = list(nperseg_array)

colors = plt.cm.cool(np.linspace(0, 1, len(nperseg_array)))


def get_corr_mat_fft(nperseg):
    '''
    helper function to pass to multiprocessing pool
    '''
    # np.random.seed(seed)
    r.reset()
    r.runSimulation(k=k, alpha=alpha)
    c_fft, omega = fe.corr_matrix(r.pos,
                                  sample_spacing=r.dt,
                                  window=args.window,
                                  nperseg=nperseg,
                                  noverlap=args.noverlap,
                                  nfft=2 * nperseg,
                                  detrend=args.detrend,
                                  padded=args.padded,
                                  return_fft=True)
    return c_fft, omega


if str(args.seed_type) == 'time':
    seed = datetime.now().microsecond
elif str(args.seed_type) == 'input':
    seed = args.seed_input
else:
    ValueError('Expected seed_type = {''time'', ''input''}, received {0}.'.format(int(args.seed)))
np.random.seed(seed)

if int(args.nperseg_array[2]) < 5:
    nProcesses = int(args.nperseg_array[2])
elif int(args.nperseg_array[2]) >= 5:
    nProcesses = 5

with multiprocessing.Pool(processes=nProcesses) as pool:
    result = pool.map(get_corr_mat_fft, nperseg_array)

sArray = np.zeros(len(result), dtype=complex)

fig, ax = plt.subplots(2, 2, sharex=True)
for ii in range(len(result)):
    c, omega = result[ii]
    T = len(omega) / (2 * omega.max())
    sArray[ii] = fe.entropy(c, T=T)
    ax[0, 0].plot(omega, c[:, 0, 0].real, alpha=0.4, color=colors[ii, :])
    ax[0, 1].plot(omega, c[:, 0, 1].imag, alpha=0.4, color=colors[ii, :])
    ax[1, 0].plot(omega, c[:, 1, 0].imag, alpha=0.4, color=colors[ii, :])
    ax[1, 1].plot(omega, c[:, 1, 1].real, alpha=0.4, color=colors[ii, :])

# use the last c and omega to plot the theoretical correlation functions
c_thry_fft = np.zeros(c.shape, dtype=complex)

c_thry_fft_prefactor = 2 * r.D / (((args.k_multiple + 1j * omega)**2 + args.alpha_multiple**2) *
                                  ((args.k_multiple - 1j * omega)**2 + args.alpha_multiple**2))
c_thry_fft[:, 0, 0] = c_thry_fft_prefactor * (args.alpha_multiple**2 + args.k_multiple**2 + omega**2)
c_thry_fft[:, 1, 1] = c_thry_fft_prefactor * (args.alpha_multiple**2 + args.k_multiple**2 + omega**2)
c_thry_fft[:, 0, 1] = c_thry_fft_prefactor * 2j * args.alpha_multiple * omega
c_thry_fft[:, 1, 0] = -c_thry_fft_prefactor * 2j * args.alpha_multiple * omega

ax[0, 0].plot(omega, c_thry_fft[:, 0, 0].real, 'k')
ax[0, 1].plot(omega, c_thry_fft[:, 0, 1].imag, 'k')
ax[1, 0].plot(omega, c_thry_fft[:, 1, 0].imag, 'k')
ax[1, 1].plot(omega, c_thry_fft[:, 1, 1].real, 'k')

ax[0, 0].set(xlim=[-15, 15], ylabel=r'$\Re \left[ \mathcal{F}\lbrace C_{xx} \rbrace \right]$')
ax[0, 1].set(xlim=[-15, 15], ylabel=r'$\Im \left[ \mathcal{F}\lbrace C_{xy} \rbrace \right]$')
ax[1, 0].set(xlim=[-15, 15], xlabel=r'frequency, $\omega$ [rad/s]', ylabel=r'$\Im \left[ \mathcal{F}\lbrace C_{yx} \rbrace \right]$')
ax[1, 1].set(xlim=[-15, 15], xlabel=r'frequency, $\omega$ [rad/s]', ylabel=r'$\Re \left[ \mathcal{F}\lbrace C_{yy} \rbrace \right]$')

plt.tight_layout()

sThry = 2 * args.alpha_multiple**2 / args.k_multiple

# fig_ent, ax_ent = plt.subplots(1, 2, figsize=[8, 4])
# ax_ent[0].plot(np.concatenate((np.zeros(1), scales)) * dw, sArray.real, 'ko')
# ax_ent[0].plot([0, scales.max() * dw], [sThry, sThry],
#                '--k', label='$S^{{thry}} = {0}$'.format(sThry))
# ax_ent[0].set(xlabel=r'$\sigma_{kernel}$',
#               ylabel=r'$d{S}/dt$')

# ax_ent[1].loglog(np.concatenate((np.zeros(1), scales)) * dw, abs(sThry - sArray.real), 'ko')
# ax_ent[1].set(xlabel=r'$\sigma_{kernel}$',
#               ylabel=r'$\vert \dot{S}_{thry} - \dot{S} \vert$')
# plt.tight_layout()

# if args.save:
#     argDict = vars(args)
#     argDict['datetime'] = datetime.now()
#     argDict['seeds'] = seeds

#     with open(os.path.join(args.savepath, args.filename + '_params.csv'), 'w') as csv_file:
#         w = csv.DictWriter(csv_file, argDict.keys())
#         w.writeheader()
#         w.writerow(argDict)

#     fig.savefig(os.path.join(args.savepath, args.filename + '_corrFuncs.pdf'), format='pdf')
#     fig_ent.savefig(os.path.join(args.savepath, args.filename + '_entropy.pdf'), format='pdf')

plt.show()
