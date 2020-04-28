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
t = np.linspace(0, 1000, 20001)
k1plus, k1minus, k2plus, k2minus, k3plus, k3minus = [1, 0.5, 2, 0.5, 2, 0.5]
a = 1
b = np.exp(mu / 2) / (k2plus * k3plus)
c = np.exp(-mu / 2) / (k2minus * k3minus)
xss = a * k1plus / k1minus
yss = (k2plus * b * xss + k3minus * xss**3) / (k2minus * c + k3plus * xss**2)

def brusselator(r, t, a, b, c, rates):
    '''
    r = [x, y]
    '''
    x, y = r
    k1plus, k1minus, k2plus, k2minus, k3plus, k3minus = rates
    drdt = [k1plus * a - k1minus * x + k2minus * y * c - k2plus * b * x + k3plus * x**2 * y - k3minus * x**3,
            -k2minus * y * c + k2plus * b * x - k3plus * x**2 * y + k3minus * x**3]
    return drdt

fig, ax = plt.subplots()

for r0 in np.array([xss, yss]) + np.random.randn(10, 2) * 0.05:
    # numerically solve the differntial equations
    sol = odeint(brusselator, r0, t, args=(a, b, c, [k1plus, k1minus, k2plus, k2minus, k3plus, k3minus]))
    ax.plot(sol[len(t)//2:, 0], sol[len(t)//2:, 1], color='k', alpha=0.1)
ax.plot(xss, yss, marker='X', markersize=10, markeredgecolor='k', color='r')

# Calculate nullclines and streamlines
x_max, y_max = sol.max(axis=0)
x_min, y_min = sol.min(axis=0)

x_range = np.linspace(-5, x_max * 2, 100)
xx, yy = np.meshgrid(np.linspace(-5, 2 * x_max, 25), np.linspace(0, 2 * y_max, 25))
y_xdot0 = (-k1plus * a + k1minus * x_range + k2plus * b * x_range + k3minus * x_range**3) / (c * k2minus + x_range**2 * k3plus)
y_ydot0 = (k3minus * x_range**3 + k2plus * b * x_range) / (c * k2minus + x_range**2 * k3plus)
xdot = k1plus * a - k1minus * xx - k2plus * b * xx + k2minus * yy * c + k3plus * xx**2 * yy - k3minus * xx**3
ydot = -(- k2plus * b * xx + k2minus * yy * c + k3plus * xx**2 * yy - k3minus * xx**3)

# ax.plot(x_range, y_xdot0, lw=2, color='C0', label=r'$\dot{x} = 0$')
# ax.plot(x_range, y_ydot0, lw=2, color='C1', label=r'$\dot{y} = 0$')
# ax.streamplot(xx, yy, xdot, ydot, color=np.sqrt(xdot**2 + ydot**2), cmap='Reds_r', density=1)

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
ax.set(xlabel=r'$x$', ylabel=r'$y$')#, ylim=[0, y_max * 1.25], xlim=[-5, x_max * 1.25])
# ax.set_aspect('equal')
ax.tick_params(which='both', direction='in')
plt.show()
