'''
Simulation of coupled Gaussian fields with noise

da/dt = -k * a + c * nabla^2 a + alpha * b + xi_a
db/dt = -k * b + c * nabla^2 b - alpha * a + xi_b

a and b are functions of space and time
xi_j is zero mean Gaussian noise with correlator

< xi_j (x, t) xi_k(x', t') > = 2D delta_jk delta(x-x') delta(t-t')
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


class gaussianFields1D():
    '''
    Simulation of 1D coupled Gaussian fields with noise

    da/dt = -k * a + c * nabla^2 a + alpha * b + xi_a
    db/dt = -k * b + c * nabla^2 b - alpha * a + xi_b

    a and b are functions of space and time
    xi_j is zero mean Gaussian noise with correlator

    < xi_j (x, t) xi_k(x', t') > = 2D delta_jk delta(x-x') delta(t-t')
    '''

    def __init__(self, dt, dx, ic, D, nsteps):

        self.dt = dt  # time step in seconds
        self.dx = dx  # lattice spacing
        self.npts = np.asarray(ic).shape[-1]  # number of lattice sites
        self.nsteps = int(nsteps)  # total number of time steps to take
        # self.k = np.asarray(k)  # spring strength as a 2-tuple, one for each field
        # self.c = np.asarray(c)  # diffusivity as a 2-tuple, one for each field
        # self.alpha = alpha  # interaction strength of the fields
        self.D = D  # noise strength

        # discretization of laplacian
        self.laplacian = (-2 * np.eye(self.npts, k=0) +
                          np.eye(self.npts, k=1) +
                          np.eye(self.npts, k=-1) +
                          np.eye(self.npts, k=(self.npts - 1)) +
                          np.eye(self.npts, k=-(self.npts - 1))).astype(int)

        # data
        self.t = np.linspace(0, self.nsteps * self.dt, self.nsteps + 1)
        self.L = np.linspace(0, self.npts * self.dx, self.npts)
        self.pos = np.zeros((2, self.nsteps + 1, self.npts))
        self.pos[:, 0, :] = np.asarray(ic)

    def reset(self):
        self.__init__(self.dt, self.dx, self.pos[:, 0, :], self.D, self.nsteps)

    def deterministicForce(self, pos, k, c, alpha):
        # spring = np.diag(-k * np.ones(self.npts))  # spring force along diagonal

        spring = np.stack((-k[0] * pos[0], -k[1] * pos[1]))
        diffusion = np.stack((np.matmul(c[0] * self.laplacian / (self.dx**2), pos[0]),
                              np.matmul(c[1] * self.laplacian / (self.dx**2), pos[1])))
        interaction = np.stack((alpha * pos[1], -alpha * pos[0]))

        return spring + diffusion + interaction

    def noise(self):
        '''
        Gaussian white noise
        '''
        return np.sqrt(2 * self.D / self.dt) * np.random.randn(2, self.npts)

    def runSimulation(self, k, c, alpha):
        # self.reset()
        for index, time in enumerate(self.t[1:]):
            pos_old = self.pos[:, index, :]
            pos_new = pos_old + (self.deterministicForce(pos_old, k, c, alpha) + self.noise()) * self.dt
            self.pos[:, index + 1, :] = pos_new

    def plotTrajectory(self, savepath=None):
        plt.ion()
        fig, ax = plt.subplots(1, 2, sharey=True)
        vmin = self.pos.min()
        vmax = self.pos.max()

        cmap = mpl.cm.get_cmap('RdBu')
        normalize = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = [cmap(normalize(value)) for value in self.pos]

        ax[0].pcolormesh(self.L, self.t, self.pos[0].T, cmap='RdBu', vmin=vmin, vmax=vmax)
        ax[0].set(xlabel='space [a.u.]', title=r'$\phi(x, t)$', ylabel='time [s]')
        ax[1].pcolormesh(self.L, self.t, self.pos[1].T, cmap='RdBu', vmin=vmin, vmax=vmax)
        ax[1].set(xlabel='space', title=r'$\psi(x, t)$')

        cax, _ = mpl.colorbar.make_axes(ax)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
        # cbar.ax.set_title(r'$$')

        plt.show()
