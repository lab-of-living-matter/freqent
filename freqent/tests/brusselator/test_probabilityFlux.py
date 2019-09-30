import numpy as np
from probabilityFlux import *
import matplotlib.pyplot as plt
import os

os.chdir('/home/daniel/llm/freqent/freqent/tests/brusselator/')
# d = h5py.File('/mnt/llmStorage203/Danny/brusselatorSims/reactionsOnly/190904/alpha10.56_nSim50/data.hdf5')
period = 1
t = np.linspace(0, 100, 10001)
dt = np.diff(t)[0]
omega = 2 * np.pi / period
x = np.cos(omega * t) + np.random.randn(len(t)) * 0.1
y = np.sin(omega * t) + np.random.randn(len(t)) * 0.1
data = np.stack((x, y), axis=1)
bins = [np.linspace(-1.5, 1.5, 51), np.linspace(-1.5, 1.5, 51)]
# bins = 50

prob_map, flux_field, edges = probabilityFlux(data, dt=dt, bins=bins)

dx, dy = [np.diff(edges[0])[0], np.diff(edges[1])[0]]

plt.close('all')
fig, ax = plt.subplots()
ax.pcolormesh(edges[0][:-1], edges[1][:-1], prob_map.T)

ax.quiver(edges[0][:-1] + dx / 2, edges[1][:-1] + dy / 2, -flux_field[1], -flux_field[0], color='w')

plt.show()
