import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2

rates = [1, 0.5, 2, 0.5, 2, 0.5]
V = 100
a = 100 / V
c = 400 / V
b = np.unique(np.logspace(0, 11, 135, base=2).astype(int)) / V

alpha = b * rates[2] * rates[4] / (c * rates[3] * rates[5])

xss = a * rates[0] / rates[1]
yss = (rates[2] * b * xss + rates[5] * xss**3) / (rates[3] * c + rates[4] * xss**2)

j11 = -(rates[1] + b * rates[2]) + 2 * xss * yss * rates[4] - 3 * rates[5] * xss**2
j12 = (c * rates[3] + xss**2 * rates[4])
j21 = b * rates[2] - 2 * xss * yss * rates[4] + 3 * rates[5] * xss**2
j22 = -(c * rates[3] + xss**2 * rates[4])

tr = j11 + j22
det = j11 * j22 - j12 * j21

lambda_plus = (tr / 2) + np.sqrt((tr / 2)**2 - det + 0j)
lambda_minus = (tr / 2) - np.sqrt((tr / 2)**2 - det + 0j)

normalize = mpl.colors.Normalize(vmin=alpha[0], vmax=alpha[-1])
cmap1 = mpl.cm.get_cmap('Greys')
colors1 = [cmap1(normalize(value)) for value in alpha]
cmap2 = mpl.cm.get_cmap('Blues')
colors2 = [cmap2(normalize(value)) for value in alpha]


fig, ax = plt.subplots(figsize=(20, 5))

a1 = ax.scatter(lambda_plus.real, lambda_plus.imag, s=70, c=alpha, cmap='Reds_r', edgecolors='k', alpha=0.7, label=r'$\lambda_+$')
a2 = ax.scatter(lambda_minus.real, lambda_minus.imag, s=70, c=alpha, cmap='Blues_r', edgecolors='k', alpha=0.7, label=r'$\lambda_-$')


# ax.scatter(lambda_plus.real[np.logical_and(alpha > 1, alpha < 30)], lambda_plus.imag[np.logical_and(alpha > 1, alpha < 30)], c='r')
# ax.scatter(lambda_minus.real[np.logical_and(alpha > 1, alpha < 30)], lambda_minus.imag[np.logical_and(alpha > 1, alpha < 30)], c='b')

# ax.scatter(lambda_plus.real[np.logical_and(alpha > 30, alpha < 45)], lambda_plus.imag[np.logical_and(alpha > 30, alpha < 45)], c='g')
# ax.scatter(lambda_minus.real[np.logical_and(alpha > 30, alpha < 45)], lambda_minus.imag[np.logical_and(alpha > 30, alpha < 45)], c='pink')

ax.axhline(lw=2, color='k')
ax.axvline(lw=2, color='k')
ax.tick_params(which='both', direction='in')
ax.set_aspect('equal')
ax.set(xlabel=r'$\Re [\lambda_\pm]}$', ylabel=r'$\Im[\lambda_\pm]$')
fig.colorbar(a1, ax=ax, shrink=0.6)

# cax, _ = mpl.colorbar.make_axes(ax)
# cbar = mpl.colorbar.ColorbarBase(a, cmap=cmap1, norm=normalize)
# cbar.ax.set_title(r'$\alpha$')
# fig.colorbar()

plt.tight_layout()
plt.show()
