inds = [0, 23, 35, 82, 83, 91, 95]
plt.close('all')
for ind, B in enumerate(b[(inds)]):
    y_xdot0 = (-rates[0] * a + rates[1] * x + rates[2] * B * x + rates[5] * x**3) / (c *rates[3] + x**2 * rates[4])
    y_ydot0 = (rates[5] * x**3 + rates[2] * B * x) / (c * rates[3] + x**2 * rates[4])
    xdot = k1plus * a - k1minus * xx - k2plus * B * xx + k2minus * yy * c + k3plus * xx**2 * yy - k3minus * xx**3
    ydot = -(- k2plus * B * xx + k2minus * yy * c + k3plus * xx**2 * yy - k3minus * xx**3)
    fig, ax = plt.subplots()
    ax.plot(x, y_xdot0, lw=2, label=r'$\dot{x} = 0$')
    ax.plot(x, y_ydot0, lw=2, label=r'$\dot{y} = 0$')
    ax.streamplot(xx, yy, xdot, ydot, color='k')
    #ax.plot([xss, v[inds[ind]][0, 0]], [yss[inds[ind]], v[inds[ind]][1,0]], 'r')
    #ax.plot([xss, v[inds[ind]][0, 1]], [yss[inds[ind]], v[inds[ind]][1,1]], 'r')
    ax.set(title=r'$\alpha = {a}$'.format(a=alphas[inds[ind]]))
    ax.legend([r'$\dot{x}=0$', r'$\dot{y}=0$', 'streamlines'], loc='upper right', framealpha=1)
    ax.tick_params(which='both', direction='in')
    fig.savefig('/media/daniel/storage11/Dropbox/LLM_Danny/freqent/brusselator/191002_nullclines_streamlines_alpha{a}.pdf'.format(a=alphas[inds[ind]]), format='pdf')
