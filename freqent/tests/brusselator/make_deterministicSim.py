'''
Non-dimensionalized Brusselator model:

dx/dt = k1plus * a - k1minus * x + k2minus * y - k2plus * b * x + k3plus * x**2 * y - k3minus * x**3
dy/dt =  -k2minus * y + k2plus * b * x - k3plus * x**2 * y + k3minus * x**3

k1plus, k1minus, k2plus, k2minus, k3plus, k3minus > 0 are parameters
a, b, c > 0 are chemostat concentrations, (b,c) are determined by thermodynamic driving
x, y > 0 are non-dimensionalized concentrations

Fixed point at (a, b/a)
Critical point at bc = 1 + a^2
attractor if b < bc (orbits fall into fixed point)
repeller if b > bc (enters limit cycle)
'''

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

plt.close('all')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.width'] = 2

parser = argparse.ArgumentParser()
parser.add_argument('--deltaMu', '-mu', type=float, default=0.0,
                    help='driving force')
args = parser.parse_args()

mu = args.deltaMu
t = np.linspace(0, 100, 2001)
k1plus, k1minus, k2plus, k2minus, k3plus, k3minus = [1, 0.5, 2, 0.5, 2, 0.5]
a = 1
b = np.sqrt(k2minus * k3minus / (k2plus * k3plus)) * np.exp(mu / 2)
c = (k2minus * k3minus) / (b * k2plus * k3plus)
xss = a * k1plus / k1minus
yss = (k2plus * b * xss + k3minus * xss**3) / (k2minus * c + k3plus * xss**2)

def brusselator(r, t, a, b, c, rates):
    '''
    r = [x, y]
    '''
    x, y = r
    k1plus, k1minus, k2plus, k2minus, k3plus, k3minus = rates
    drdt = [k1plus * a - k1minus * x + k2minus * y - k2plus * b * x + k3plus * x**2 * y - k3minus * x**3,
            -k2minus * y + k2plus * b * x - k3plus * x**2 * y + k3minus * x**3]
    return drdt

fig, ax = plt.subplots()

for r0 in np.random.rand(100, 2) * 1.5 * np.array(xss, yss):
    sol = odeint(brusselator, r0, t, args=(a, b, c, [k1plus, k1minus, k2plus, k2minus, k3plus, k3minus]))
    ax.plot(sol[len(t) // 2:, 0], sol[len(t) // 2:, 1], color='k', alpha=0.1)
    ax.plot(xss, yss, marker='X', markersize=10, markeredgecolor='k', color='r')

# for bInd, b in enumerate(bArray):
#     r0 = np.random.rand(2) * bc
#     sol = odeint(brusselator, r0, t, args=(a, b))
#     ax.plot(sol[:, 0], sol[:, 1], color='C{0}'.format(bInd))
#     ax.plot(a, b / a, marker='X', markersize=10, markeredgecolor='k', color='C{0}'.format(bInd))

# Optionally add a colorbar
# cax, _ = mpl.colorbar.make_axes(ax)
# cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
# cbar.ax.set_title(r'$b$')

legend_elements = [mpl.lines.Line2D([0], [0], markersize=10, marker='X',
                                    linestyle='None', markeredgecolor='k',
                                    color='r', label=r'fixed point'),
                   mpl.lines.Line2D([0], [0], color='k', label='trajectory')]

ax.legend(handles=legend_elements, loc='best')
ax.set(xlabel=r'$[X]$', ylabel=r'$[Y]$')#, title=r'Brusselator, $a$={a}, $b_c$={bc}, $b=${b}'.format(a=a, bc=bc, b=b))
ax.set_aspect(np.diff(ax.set_xlim())[0] / np.diff(ax.set_ylim())[0])
ax.tick_params(which='both', direction='in')
plt.show()
