import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.close('all')
rates = [1, 0.5, 2, 0.5, 2, 0.5]
V = 100
a = 100 / V
c = 400 / V
b = np.unique(np.logspace(0, 11, 501, base=2).astype(int)) / V

alpha = b * rates[2] * rates[4] / (c * rates[3] * rates[5])

xss = a * rates[0] / rates[1]
yss = (rates[2] * b * xss + rates[5] * xss**3) / (rates[3] * c + rates[4] * xss**2)

j11 = -(rates[1] + b * rates[2]) + 2 * xss * yss * rates[4] - 3 * rates[5] * xss**2
j12 = (c * rates[3] + xss * rates[4])
j21 = b * rates[2] - 2 * xss * yss * rates[4] + 3 * rates[5] * xss**2
j22 = -(c * rates[3] + xss * rates[4])

tr = j11 + j22
det = j11 * j22 - j12 * j21

lambda_plus = (tr / 2) + np.sqrt((tr / 2)**2 - det + 0j)
lambda_minus = (tr / 2) - np.sqrt((tr / 2)**2 - det + 0j)

normalize = mpl.colors.Normalize(vmin=alpha[0], vmax=alpha[-1])
cmap1 = mpl.cm.get_cmap('Reds')
colors1 = [cmap1(normalize(value)) for value in alpha]
cmap2 = mpl.cm.get_cmap('Blues')
colors2 = [cmap2(normalize(value)) for value in alpha]


fig, ax = plt.subplots()
# ax.spines['bottom'].set_position('zero')
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_position('zero')
# ax.spines['left'].set_linewidth(2)
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# for ind, (p, m) in enumerate(zip(lambda_plus, lambda_minus)):
ax.scatter(lambda_plus.real, lambda_plus.imag, c=alpha, cmap='Reds')
ax.scatter(lambda_minus.real, lambda_minus.imag, c=alpha, cmap='Blues')
# ax.grid(True)
ax.axhline(lw=2, color='k')
ax.axvline(lw=2, color='k')
ax.tick_params(which='both', direction='in')
ax.set_aspect('equal')
ax.set(xlabel=r'$\Re [\lambda]}$', ylabel=r'$\Im[\lambda]$')
# cax, _ = mpl.colorbar.make_axes(ax)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap1, norm=normalize)
# cbar.ax.set_title(r'$\alpha$')
plt.tight_layout()
plt.show()
