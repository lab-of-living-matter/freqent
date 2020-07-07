import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import os

mpl.rcParams['pdf.fonttype'] = 42

x = np.arange(0.01, 5, 0.01)
xx, yy = np.meshgrid(np.linspace(0.01, 5, 101), np.linspace(0, 15, 101))

k1plus, k1minus, k2plus, k2minus, k3plus, k3minus = [1, 0.5, 2, 0.5, 2, 0.5]
V = 100
const = 100
a = 100 / V
mu = np.arange(4, 6.6, 0.25)
B = np.sqrt(k2minus * k3minus / (k2plus * k3plus)) * np.exp(mu / 2) * const
C = const**2 / B
b = B / V
c = C / V
# b = np.unique(np.logspace(0, 11, 135, base=2).astype(int)) / V
# alphas = b * k2plus * k3plus / (c * k2minus * k3minus)
today = datetime.now().strftime('%y%m%d')
savefolder = '/Users/Danny/Dropbox/LLM_Danny/freqent/brusselator'

# inds = [0, 23, 35, 82, 83, 91, 95]
plt.close('all')

for ind, (B, C) in enumerate(zip(b, c)):
    y_xdot0 = (-k1plus * a + k1minus * x + k2plus * B * x + k3minus * x**3) / (C * k2minus + x**2 * k3plus)
    y_ydot0 = (k3minus * x**3 + k2plus * B * x) / (C * k2minus + x**2 * k3plus)
    xdot = k1plus * a - k1minus * xx - k2plus * B * xx + k2minus * yy * C + k3plus * xx**2 * yy - k3minus * xx**3
    ydot = -(- k2plus * B * xx + k2minus * yy * C + k3plus * xx**2 * yy - k3minus * xx**3)
    fig, ax = plt.subplots()
    ax.plot(x, y_xdot0, lw=2, color='C0', label=r'$\dot{x} = 0$')
    ax.plot(x, y_ydot0, lw=2, color='C1', label=r'$\dot{y} = 0$')
    ax.streamplot(xx, yy, xdot, ydot, color=np.sqrt(xdot**2 + ydot**2), cmap='Greys_r', density=1)
    # ax.plot([xss, v[inds[ind]][0, 0]], [yss[inds[ind]], v[inds[ind]][1,0]], 'r')
    # ax.plot([xss, v[inds[ind]][0, 1]], [yss[inds[ind]], v[inds[ind]][1,1]], 'r')
    ax.set(title=r'$\mu = {mu:0.2f}$'.format(mu=mu[ind]), ylim=[0, 15])
    ax.legend([r'$\dot{x}=0$', r'$\dot{y}=0$', 'streamlines'], loc='upper right', framealpha=1)
    ax.tick_params(which='both', direction='in')
    plt.tight_layout()
    fig.savefig(os.path.join(savefolder, today + '_nullclines', 'nullclines_streamlines_mu{m}.pdf'.format(m=mu[ind])), format='pdf')

plt.show()
