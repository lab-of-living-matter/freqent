import numpy as np
from brussfield_gillespie import *
import multiprocessing
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def runSim(seed):
    bf = brusselator1DFieldStochSim(XY0, ABC, rates, t, D, n, v, seed)
    bf.runSimulation()

    return bf.reactionTypeTracker


n = 25
v = 1
rates = [0, 0, 0, 0, 0, 0]
ABC = [0, 0, 0]
D = [10, 10]
t = np.linspace(0, 100, 100001)
XY0 = np.ones((2, n)) * 10
nsim = 40

seeds = (np.random.rand(nsim) * 1000).astype(int)

print('Running simulations...')
with multiprocessing.Pool(processes=8) as pool:
    result = pool.map(runSim, seeds)
print('Done...')

prob_diffusion = np.asarray(result).T[:4]

df = pd.DataFrame({'prob': np.ravel(prob_diffusion),
                   'species': ['X'] * (2 * nsim) + ['Y'] * (2 * nsim),
                   'dir': ['right'] * nsim + ['left'] * nsim + ['right'] * nsim + ['left'] * nsim})

sns.violinplot(x='species', y='prob', hue='dir', data=df, split=True, inner='stick')

plt.show()
