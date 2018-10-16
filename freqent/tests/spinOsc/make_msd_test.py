'''
check to make sure that the simulation gives valid diffusive behavior.
We run the simulation with the potential turned off and measure the mean
square displacement. The theoretical value expected is:
<(r(t+\tau) - r(t))^2> = 4 D \tau
'''

import numpy as np
import matplotlib.pyplot as plt
from spinOscSimulation import spinOscLangevin
import freqent.freqent as fe
import matplotlib as mpl
import os
import pandas as pd
import trackpy as tp
mpl.rcParams['pdf.fonttype'] = 42

savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/frequencySpaceDissipation/tests/spinOsc/'

# Simulation environment parameters
gamma = 2e-8  # drag on 1um particle in water in kg/s
dt = 1e-3  # time step of simulation in seconds
nsteps = 1e4  # number of simulation steps
kT = 4e-9  # thermal energy in kg um^2 / s^2
r0 = np.random.rand(2) - 0.5  # starting xy position in um

# create object
r = spinOscLangevin(dt=dt, nsteps=nsteps, kT=kT, gamma=gamma, r0=r0)

# Calculate mean square displacement using trackpy's built-in function
mpp = 1  # microns/pixel, i.e. how many um is "1" in r.pos?
fps = int(1 / r.dt)  # frames per second
max_lagtime = 1000  # max number of frames to calculate lagtime
all_data = pd.DataFrame(columns=['t', 'x', 'y', 'frame', 'particle'])
nsim = 50
for ii in range(nsim):
    txt = 'simulation {n} of {N}'.format(n=ii, N=nsim)
    print(txt, end='\r')
    r.reset()
    r.runSimulation(k=0, alpha=0)
    # First have to put data into a DataFrame
    data = pd.DataFrame({'t': r.t,
                         'x': r.pos[0],
                         'y': r.pos[1],
                         'frame': np.arange(len(r.t)),
                         'particle': ii * np.ones(len(r.t))})
    all_data = all_data.append(data)

# msd for each simulation
msdInd = tp.imsd(all_data, mpp, fps, max_lagtime=max_lagtime)

# ensemble mean across all particles
msd = tp.emsd(all_data, mpp, fps, max_lagtime=max_lagtime)

# theoretical answer is 4Dt
thry = 4 * r.D * r.t[:max_lagtime]

fig, ax = plt.subplots()
ax.loglog(msdInd.index, msdInd, 'k-', alpha=0.2)
ax.loglog(msd.index, msd, 'r-', linewidth=3, label='ensemble mean')
ax.loglog(r.t[:max_lagtime], thry, 'c-', label=r'$4 D \tau$')
ax.set(xlabel=r'lag time, $\tau$ [s]',
       ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu m^2$]')
ax.set_aspect('equal')
plt.legend()

fig.savefig(os.path.join(savepath, 'msd_test.pdf'), format='pdf')

plt.show()
