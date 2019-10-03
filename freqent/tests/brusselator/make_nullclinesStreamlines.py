import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42

x = np.linspace(0.01, 5, 501)
xx, yy = np.meshgrid(np.linspace(0.01, 5, 101), np.linspace(-0.5, 8, 101))

k1plus, k1minus, k2plus, k2minus, k3plus, k3minus = [1, 0.5, 2, 0.5, 2, 0.5]
V = 100
a = 100 / V
c = 400 / V
b = np.unique(np.logspace(0, 11, 135, base=2).astype(int)) / V
alphas = b * k2plus * k3plus / (c * k2minus * k3minus)


inds = [0, 23, 35, 82, 83, 91, 95]
plt.close('all')
for ind, B in enumerate(b[(inds)]):
    y_xdot0 = (-k1plus * a + k1minus * x + k2plus * B * x + k3minus * x**3) / (c * k2minus + x**2 * k3plus)
    y_ydot0 = (k3minus * x**3 + k2plus * B * x) / (c * k2minus + x**2 * k3plus)
    xdot = k1plus * a - k1minus * xx - k2plus * B * xx + k2minus * yy * c + k3plus * xx**2 * yy - k3minus * xx**3
    ydot = -(- k2plus * B * xx + k2minus * yy * c + k3plus * xx**2 * yy - k3minus * xx**3)
    fig, ax = plt.subplots()
    ax.plot(x, y_xdot0, lw=2, label=r'$\dot{x} = 0$')
    ax.plot(x, y_ydot0, lw=2, label=r'$\dot{y} = 0$')
    ax.streamplot(xx, yy, xdot, ydot, color=np.sqrt(xdot**2 + ydot**2), cmap='Greys')
    # ax.plot([xss, v[inds[ind]][0, 0]], [yss[inds[ind]], v[inds[ind]][1,0]], 'r')
    # ax.plot([xss, v[inds[ind]][0, 1]], [yss[inds[ind]], v[inds[ind]][1,1]], 'r')
    ax.set(title=r'$\alpha = {a}$'.format(a=alphas[inds[ind]]))
    ax.legend([r'$\dot{x}=0$', r'$\dot{y}=0$', 'streamlines'], loc='upper right', framealpha=1)
    ax.tick_params(which='both', direction='in')
    # fig.savefig('/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brusselator/191002_nullclines_streamlines_alpha{a}.pdf'.format(a=alphas[inds[ind]]), format='pdf')

plt.show()
