import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--files', '-f', type=str, nargs='+',
                    help='files to perform TUR test on')
parser.add_argument('--dbin', '-d', type=float, default=1,
                    help='size of bins when calculating density and fluxes')
parser.add_argument('--tau', '-t', type=float, default=None,
                    help='length of time to divide each trajectory into when calculating flux statistics')
args = parser.parse_args()


def F(x, y, rates, chemostats, V):
    '''
    calculate deterministic equation of motion at location (x, y)
    '''
    k1plus, k1minus, k2plus, k2minus, k3plus, k3minus = rates
    A, B, C = chemostats
    fx = (k1plus * A - k1minus * x -
          k2plus * B * x / V + k2minus * y * C / V +
          k3plus * x * (x - 1) * y / V**2 - k3minus * x * (x - 1) * (x - 2) / V**2)
    fy = (k2plus * B * x / V - k2minus * y * C / V -
          k3plus * x * (x - 1) * y / V**2 + k3minus * x * (x - 1) * (x - 2) / V**2)

    return np.array([fx, fy])


def chunk(arr, n):
    '''
    divide arr into n-sized chunks, return as list of arrays
    '''
    n = max(1, n)
    return [arr[ii:ii + n] for ii in range(0, len(arr), n)]


fig, ax = plt.subplots()
for file in args.files:
    with h5py.File(os.path.join(file, 'data.hdf5')) as d:
        t = d['data']['t_points'][:]
        dt = np.diff(t)[0]

        # get parameters from simulation
        rates = d['params']['rates'][:]
        chemostats = np.array([d['params']['A'][()],
                               d['params']['B'][()],
                               d['params']['C'][()]])
        V = d['params']['V'][()]
        mu = np.log((chemostats[1] * rates[2] * rates[4]) /
                    (chemostats[2] * rates[3] * rates[5]))
        t_ss = t > 100  # get times in steady state
        n_traj = d['params']['nSim'][()]
        j_total = np.zeros((n_traj, len(np.where(t_ss)[0]) - 2))

        edges = [np.arange(d['data']['trajs'][:, 0, t_ss].min() - 5.5,
                           d['data']['trajs'][:, 0, t_ss].max() + 5.5,
                           args.dbin),
                 np.arange(d['data']['trajs'][:, 1, t_ss].min() - 5.5,
                           d['data']['trajs'][:, 1, t_ss].max() + 5.5,
                           args.dbin)]
        tau_length = len(np.where(t < args.tau)[0])
        j_mean = np.zeros(n_traj)
        j_var = np.zeros(n_traj)
        epr = d['data']['epr'][()]
        epr_blind = d['data']['epr_blind'][()]

        for traj_ind, traj in enumerate(d['data']['trajs'][..., t_ss]):
            # data comes out as in shape (2, nt), rotate to (nt, 2)
            traj = traj.T

            prob_map, edges = np.histogramdd(traj, bins=edges)
            prob_map /= prob_map.sum()
            bin_centers = [e[:-1] + args.dbin / 2 for e in edges]

            for t_ind, (prior_state, current_state, next_state) in enumerate(zip(traj[:-2],
                                                                                 traj[1:-1],
                                                                                 traj[2:])):
                flux = (next_state - prior_state) / (2 * dt)
                force = F(current_state[0], current_state[1], rates, chemostats, V)
                current_bin_index = [np.digitize(s, e) - 1 for s, e in zip(current_state, edges)]
                j_total[traj_ind, t_ind] = np.dot(flux, force) * prob_map[current_bin_index[0],
                                                                          current_bin_index[1]]
            j_chunks = chunk(j_total[traj_ind], tau_length)
            j_mean[traj_ind] = np.mean([jj[-1] for jj in [np.cumsum(j) for j in j_chunks]])
            j_var[traj_ind] = np.var([jj[-1] for jj in [np.cumsum(j) for j in j_chunks]])

        epr_tur = np.mean(j_mean)**2 / (args.tau * np.mean(j_var))

    ax.plot(mu, epr, 'o', color='C0')
    ax.plot(mu, epr_blind, 'o', color='C1')
    ax.plot(mu, epr_tur, 'o', color='k')

