import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing
import csv
from brusselator_gillespie import brusselatorStochSim
mpl.rcParams['pdf.fonttype'] = 42

rates = [0.5, 0.25, 1, 0.25, 1, 0.25]
V = 100
A = V
B = V
C = V * 16
t_points = np.linspace(0, 100, 1001)
nSim = 10
filename = '/media/daniel/storage11/Dropbox/LLM_Danny/frequencySpaceDissipation/tests/brusselator/stochastic_simulations/nq_10sim'


def get_traj(seed):
    '''
    function to pass to multiprocessing pool
    '''
    np.random.seed(seed)
    [X0, Y0] = np.random.rand(2) * 10 * V
    bz = brusselatorStochSim([X0, Y0, A, B, C], rates, V, t_points, seed)
    bz.runSimulation()

    return bz.population[:, :2]

seeds = np.zeros(nSim)

for ii in range(nSim):
    seeds[ii] = datetime.now().microsecond

with multiprocessing.Pool(processes=10) as pool:
    result = pool.map(get_traj, seeds.astype(int))

trajs = np.asarray(result)

fig, ax = plt.subplots()
for ii in range(nSim):
    ax.plot(trajs[ii, :, 0], trajs[ii, :, 1], 'k', alpha=0.2)

ax.set(xlabel='X', ylabel='Y')
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
plt.tight_layout()

params = {'rates': rates,
          'V': V,
          'A': A,
          'B': B,
          'C': C,
          't_points': t_points,
          'seeds': seeds}

# with open(filename + '_params.csv', 'w') as csv_file:
#     w = csv.DictWriter(csv_file, params.keys())
#     w.writeheader()
#     w.writerow(params)

# fig.savefig(filename + '.pdf', format='pdf')
