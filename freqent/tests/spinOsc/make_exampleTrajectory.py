from spinOscSimulation import spinOscLangevin
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42
plt.close('all')

seed = 101028
np.random.seed(seed)
dt = 1e-3
r0 = np.random.randn(2)
nsteps = 1e5
k_multiple = 1
alpha_multiple = 3

r = spinOscLangevin(dt=dt, r0=r0, nsteps=nsteps)
r.runSimulation(k=k_multiple * r.gamma, alpha=alpha_multiple * r.gamma)

# Get force field for plotting
xmax = 1.5
ymax = 1.5
xx, yy = np.meshgrid(np.linspace(-xmax, xmax, 10), np.linspace(-ymax, ymax, 10))
fx = -k_multiple * r.gamma * xx - alpha_multiple * r.gamma * yy
fy = -k_multiple * r.gamma * yy + alpha_multiple * r.gamma * xx

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

# cmap_time = mpl.cm.get_cmap('gray_r')
# normalize_time = mpl.colors.Normalize(vmin=min(t_range), vmax=max(t_range))
# colors_time = [cmap_time(normalize_time(value)) for value in t_range]

# for tInd, t in enumerate(t_range):
#     ax.plot(r.pos[0, t:t + 2], r.pos[1, t:t + 2], color=colors_time[tInd])

ax.plot(r.pos[0, t_range], r.pos[1, t_range], color=(0.2, 0.2, 0.2))
ax.plot(r.pos[0, t_range.max()],
        r.pos[1, t_range.max()],
        'wo', markersize=10, markeredgecolor='k')

# Optionally add a colorbar
# cax, _ = mpl.colorbar.make_axes(ax)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap_time, norm=normalize_time)
# cbar.ax.set_title(r'$t$')

cax2, _ = mpl.colorbar.make_axes(ax)
cbar2 = mpl.colorbar.ColorbarBase(cax2, cmap=cmap_prob, norm=normalize_prob)
cbar2.ax.set_title(r'$p(x)$')

ax.set(xlim=[-xmax, xmax], ylim=[-ymax, ymax], xticks=[-1, 0, 1], yticks=[-1, 0, 1])
ax.set_aspect('equal')

plt.show()
