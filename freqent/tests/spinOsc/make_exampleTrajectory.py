from spinOscSimulation import spinOscLangevin
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import os
mpl.rcParams['pdf.fonttype'] = 42
plt.close('all')

savepath = '/media/daniel/storage11/Dropbox/LLM_Danny/freqent/spinOsc/'
seed = 101028
np.random.seed(seed)
dt = 1e-3
r0 = np.random.randn(2)
nsteps = 1e5
alpha = 2

r = spinOscLangevin(dt=dt, r0=r0, nsteps=nsteps)
r.runSimulation(alpha=alpha)

# Get force field for plotting
xmax = 3.5
ymax = 3.5
xx, yy = np.meshgrid(np.linspace(-xmax, xmax, 10), np.linspace(-ymax, ymax, 10))
fx = -xx - alpha * yy
fy = -yy + alpha * xx

# plot results
fig, ax = plt.subplots()

# plot pdf of occupation
nbins = 50
d = ax.hist2d(r.pos[0], r.pos[1],
              bins=[np.linspace(-xmax, xmax, nbins), np.linspace(-ymax, ymax, nbins)],
              cmap='Blues')

prob = d[0] / d[0].sum()
cmap_prob = mpl.cm.get_cmap('Blues')
normalize_prob = mpl.colors.Normalize(vmin=prob.min(), vmax=prob.max())
colors_prob = [cmap_prob(normalize_prob(value)) for value in prob.ravel()]

# plot force field
ax.quiver(xx, yy, fx, fy, color='r', alpha=1)

# plot subset of total trajectory
t_range = np.arange(int(nsteps / 3 - nsteps * 0.008),
                    int(nsteps / 3 + nsteps * 0.008))

# cmap_time = mpl.cm.get_cmap('Greys')
# normalize_time = mpl.colors.Normalize(vmin=min(t_range), vmax=max(t_range))
# colors_time = [cmap_time(normalize_time(value)) for value in t_range]

# for tInd, t in enumerate(t_range):
#     ax.plot(r.pos[0, t:t + 2], r.pos[1, t:t + 2], color=colors_time[tInd])

ax.plot(r.pos[0, t_range], r.pos[1, t_range], color='k', alpha=0.7)
ax.plot(r.pos[0, t_range.max()],
        r.pos[1, t_range.max()],
        'o', markersize=10, markeredgecolor='k', color=(0.9, 0.9, 0.9))

# Optionally add a colorbar
# cax, _ = mpl.colorbar.make_axes(ax)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap_time, norm=normalize_time)
# cbar.ax.set_title(r'$t$')

cax2, _ = mpl.colorbar.make_axes(ax)
cbar2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap_prob, norm=normalize_prob)
cbar2.ax.set_title(r'$p(x)$')

ax.set(xlim=[-xmax, xmax], ylim=[-ymax, ymax], xticks=[-2, 0, 2], yticks=[-2, 0, 2])
ax.tick_params(axis='both', bottom=False, left=False)
ax.set_aspect('equal')

fig.savefig(os.path.join(savepath, datetime.now().strftime('%y%m%d') + '_exampleTrajectory_alpha2.pdf'), format='pdf')

plt.show()
