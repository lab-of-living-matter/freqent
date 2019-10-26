import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.close('all')
# mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['font.size'] = 12
# mpl.rcParams['axes.linewidth'] = 2
# mpl.rcParams['xtick.major.width'] = 2
# mpl.rcParams['xtick.minor.width'] = 2
# mpl.rcParams['ytick.major.width'] = 2
# mpl.rcParams['ytick.minor.width'] = 2

k1plus, k1minus, k2plus, k2minus, k3plus, k3minus = [1, 0.5, 2, 0.5, 2, 0.5]
V = 100
const = 100
# mu = np.linspace(0, 20, 101)
mu = np.concatenate((np.arange(-2, 8, 0.1), np.arange(8, 9.01, 0.05)))
# mu = np.concatenate((-np.flip(mu[1:]), mu))
a = 100 / V
B = np.sqrt(k2minus * k3minus / (k2plus * k3plus)) * np.exp(mu / 2) * const
C = (const**2 / B)
b = B / V
c = C / V

# c = 400 / V
# b = np.unique(np.logspace(0, 11, 135, base=2).astype(int)) / V
# b = np.linspace(2, 2048, 2049) / V

alphas = b * k2plus * k3plus / (c * k2minus * k3minus)

xss = a * k1plus / k1minus
yss = (k2plus * b * xss + k3minus * xss**3) / (k2minus * c + k3plus * xss**2)

j11 = -(k1minus + b * k2plus) + 2 * xss * yss * k3plus - 3 * k3minus * xss**2
j12 = (c * k2minus + xss**2 * k3plus)
j21 = b * k2plus - 2 * xss * yss * k3plus + 3 * k3minus * xss**2
j22 = -(c * k2minus + xss**2 * k3plus)

tr = j11 + j22
det = j11 * j22 - j12 * j21

lambda_plus = (tr / 2) + np.sqrt((tr / 2)**2 - det + 0j)
lambda_minus = (tr / 2) - np.sqrt((tr / 2)**2 - det + 0j)

normalize = mpl.colors.Normalize(vmin=mu[0], vmax=mu[-1])
cmap1 = mpl.cm.get_cmap('Greys')
colors1 = [cmap1(normalize(value)) for value in mu]
cmap2 = mpl.cm.get_cmap('Blues')
colors2 = [cmap2(normalize(value)) for value in mu]


fig, ax = plt.subplots(figsize=(20, 5))

a1 = ax.scatter(lambda_plus.real, lambda_plus.imag, s=70, c=mu, cmap='Reds_r', edgecolors='k', alpha=0.5, label=r'$\lambda_+$')
a2 = ax.scatter(lambda_minus.real, lambda_minus.imag, s=70, c=mu, cmap='Blues_r', edgecolors='k', alpha=0.5, label=r'$\lambda_-$')


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
