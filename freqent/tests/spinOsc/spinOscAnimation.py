'''
Animation of a Brownian particle in a 2D force-field, described by the Langevin
Equation:

dr/dt = F + xi

with

F = -k*r + alpha * curl(z,r)

The k term represents a harmonic potential
the alpha term represents a rotational, non-conservative force for the potential
xi is Gaussian white noise with strength sqrt(2*gamma^2*D), D is the diffusion constant
which is given by the Einstein-relation D = kB*T/gamma
'''


import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
import matplotlib.animation as animation

gamma = 2e-8  # drag on particle in kg / s
kT = 4e-9  # energy in kg * um^2 / s^2
# kT = 4e-21  # energy in Joules at room temp
D = kT / gamma
dt = 1e-3  # time step

k = 2 * gamma  # spring constant of harmonic potential
alpha = 1 * gamma  # strength of rotational force
zhat = np.array([0, 0, 1])
nframes = int(2e3)


def update(p, r):
    '''
    Update r[p] to r[q] and move forward in time
    '''
    q = (p + 1)
    springForce = - k * r[p]
    rotationalForce = alpha * np.cross(zhat, np.append(r[p], 0))[:2]
    noise = np.sqrt(2 * gamma**2 * D / dt) * np.random.randn(2)

    r[q] = r[p] + (springForce + rotationalForce + noise) * dt / gamma
    return r


# Initialize random data. Row is time point, columns are x and y
xmax = 1
ymax = 1
r = (np.random.rand(nframes, 2) - 0.5) * [xmax, ymax]

fig, ax = plt.subplots()
ax.set_title(r'$\tau_\alpha$ = {a}, $\tau_k$ = {k}'.format(a=gamma / alpha, k=gamma / k))
ax.set_xlim(-xmax, xmax)
ax.set_ylim(-ymax, ymax)
ax.set_aspect('equal')
fig.tight_layout()
plt.axis('off')
xx, yy = np.meshgrid(np.linspace(-xmax, xmax, 10), np.linspace(-ymax, ymax, 10))
fx = -k * xx - alpha * yy
fy = -k * yy + alpha * xx

ax.quiver(xx, yy, fx, fy, color='r', alpha=0.5)
line, = ax.plot([], [], 'k', alpha=0.9)


def init():
    line.set_data(r[0, 0], r[0, 1])
    return line,


def animate(i, r, line):
    '''
    Animate the simulation
    '''

    r = update(i, r)
    # line.set_data(r[i, 0], r[i, 1])
    if i < 200:
        line.set_data(r[:i, 0], r[:i, 1])
    else:
        line.set_data(r[i - 200:i - 1, 0], r[i - 200:i - 1, 1])

    return line,


anim = animation.FuncAnimation(fig, animate, frames=nframes - 1, blit=True, fargs=(r, line), interval=5)

plt.show()

# save the animation
# anim.save(filename='spinOsc.mp4', fps=100)
