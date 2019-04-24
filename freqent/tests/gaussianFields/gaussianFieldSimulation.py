'''
Simulation of coupled Gaussian fields with noise

da/dt = -D * (r  - nabla^2) a + alpha * b + xi_a
db/dt = -D * (r  - nabla^2) b - alpha * a + xi_b

a and b are functions of space and time
xi_j is zero mean Gaussian noise with correlator

< xi_j (x, t) xi_k(x', t') > = 2 D delta_jk delta(x-x') delta(t-t')
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


class gaussianFields1D():
    '''
    Simulation of 1D coupled Gaussian fields with noise

    da/dt = -D * (r  - nabla^2) a + alpha * b + xi_a
    db/dt = -D * (r  - nabla^2) b - alpha * a + xi_b

    a and b are functions of space and time
    xi_j is zero mean Gaussian noise with correlator

    < xi_j (x, t) xi_k(x', t') > = 2 D kT delta_jk delta(x-x') delta(t-t')

    In everything below, kT = 1. We non-dimensionalize the system as

    da'/dt' = -(1  - nabla'^2) a' + alpha' * b' + tau/L^(1/2) xi_a
    db'/dt' = -(1  - nabla'^2) b' - alpha' * a' + tau/L^(1/2) xi_b

    where the length scale L = 1 / sqrt(r) and time scale T = 1/(D*r)
    are chosen to nondimensionalize the fields a and b (which have units
    of L^0.5) and rescale space and time. alpha' = T alpha is a dimensionless
    number describing how far the system is driven from equilibrium.

    Parameters
    ----------
    dt : scalar
        time step in units of T = 1/(D*r)
    dx : scalar
        spatial lattice spacing in units of L = 1/sqrt(r)
    ic : array-like
        array with initial conditions for the fields. Shape 2xN,
        where N is the total number of lattice sites
    D : scalar
        mobility and noise strength (must be both to obey
        fluctuation-dissipation in equilibrium)
    nsteps : int
        number of time steps to take in simulation

    Returns
    -------
    f : class object
        class object with the following properties:
            dt : time step
            dx : lattice spacing
            npts : number of lattice sites
            nsteps : number of time steps
            D : noise strength / mobility
            t : array of all time points
            L : array of all lattice points
            pos : 2 x (nsteps + 1) x npts array of values of
                  fields after simulation

        and the following methods:
            reset() : reset pos to zeros
            deterministicForce(pos, r, c, alpha) : calculate deterministic
                                                   forces given coefficients
            noise() : thermal noise
            runSimulation(k, c, alpha) : run a simulation given coefficients
            plotTrajectory(savepath=None) : plot trajectory


    '''

    def __init__(self, dt, dx, ic, nsteps):
        self.dt = dt  # time step in simulation time units
        self.dx = dx  # lattice spacing in simulation length units
        self.npts = np.asarray(ic).shape[-1]  # number of lattice sites
        self.nsteps = int(nsteps)  # total number of time steps to take
        # self.D = D  # noise strength

        # discretization of laplacian
        self.laplacian = (-2 * np.eye(self.npts, k=0) +
                          np.eye(self.npts, k=1) +
                          np.eye(self.npts, k=-1) +
                          np.eye(self.npts, k=(self.npts - 1)) +
                          np.eye(self.npts, k=-(self.npts - 1))).astype(int) / (self.dx**2)

        # data
        self.t = np.linspace(0, self.nsteps * self.dt, self.nsteps + 1)
        self.L = np.linspace(0, self.npts * self.dx, self.npts)
        self.pos = np.zeros((2, self.nsteps + 1, self.npts))
        self.pos[:, 0, :] = np.asarray(ic)

    def reset(self):
        self.__init__(self.dt, self.dx, self.pos[:, 0, :], self.D, self.nsteps)

    def deterministicForce(self, pos, alpha):
        spring = np.stack((pos[0], pos[1]))
        diffusion = np.stack((np.matmul(self.laplacian, pos[0]),
                              np.matmul(self.laplacian, pos[1])))
        interaction = np.stack((alpha * pos[1], -alpha * pos[0]))

        return (-spring + diffusion) + interaction

    def noise(self):
        '''
        Gaussian white noise
        '''
        return np.sqrt(2 / self.dt) * np.random.randn(2, self.npts)

    def runSimulation(self, alpha):
        '''
        Run a simulation of the interacting fields
        '''
        # self.reset()
        for index, time in enumerate(self.t[1:]):
            pos_old = self.pos[:, index, :]
            pos_new = pos_old + (self.deterministicForce(pos_old, alpha) + self.noise()) * self.dt
            self.pos[:, index + 1, :] = pos_new

    def plotTrajectory(self, savepath=None, tmin_frac=0, tmax_frac=1, delta=1):
        '''
        plot the fields in time range [tmin_frac, tmax_frac] * T in steps of delta
        '''
        T = self.dt * self.nsteps
        tInd = np.logical_and(self.t > tmin_frac * T, self.t < tmax_frac * T)

        fig, ax = plt.subplots(1, 2, sharey=True)
        vmin = self.pos[:, tInd, :].min()
        vmax = self.pos[:, tInd, :].max()

        cmap = mpl.cm.get_cmap('coolwarm')
        normalize = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = [cmap(normalize(value)) for value in self.pos[:, tInd, :]]

        ax[0].pcolorfast(self.L, self.t[tInd[::delta]], self.pos[0, tInd[::delta], :],
                         cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax[1].pcolorfast(self.L, self.t[tInd[::delta]], self.pos[1, tInd[::delta], :],
                         cmap='coolwarm', vmin=vmin, vmax=vmax)

        ax[0].set(xlabel=r'$x \ [1/\sqrt{r}]$', title=r'$\phi(x, t)$', ylabel=r'$t \ [1/Dr]$')
        ax[1].set(xlabel=r'$x \ [1/\sqrt{r}]$', title=r'$\psi(x, t)$')

        cax, _ = mpl.colorbar.make_axes(ax)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
        # cbar.ax.set_title(r'$$')

        if savepath:
            fig.savefig(os.path.join(savepath, 'traj.pdf'), format='pdf')

        plt.show()
