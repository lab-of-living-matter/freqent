'''
Simulation of a driven Brownian particle on a ring, described by the Langevin
Equation:

gamma * dx/dt = F + xi

xi is Gaussian white noise with strength sqrt(2*gamma^2*D), D is the diffusion constant
which is given by the Einstein-relation D = kB*T/gamma
'''

import numpy as np


class particleOnARing():
    '''
    class for solving the 1-D overdamped Langevin equation.
    Models a drift-diffusion process

    gamma * dx = F * dt + xi * dt

    gamma is a drag coefficient
    F / gamma = constant drift velocity
    xi is zero mean Gaussian white noise
        <xi> = 0
        <xi_i (t) xi_j (t')> = 2*D*gamma^2 * delta_ij * delta(t - t')
            D is diffusion constant, D = kB*T/gamma

    For simulations, we non-dimensionalize the equations of motion with
    time scale gamma/k and length scale sqrt(D * gamma / k).
    '''

    def __init__(self, dt, x0, nsteps=1e6):

        self.dt = dt  # time step in simulation time
        self.nsteps = int(nsteps)  # total number of steps
        # self.gamma = gamma  # drag on particle in kg / s
        # self.kT = kT  # in kg * um^2 / s^2
        self.x0 = x0

        # derived quantities
        # self.D = self.kT / self.gamma

        # data
        self.t = np.linspace(0, self.nsteps * self.dt, self.nsteps + 1)
        self.pos = np.zeros(self.nsteps + 1)
        self.pos[0] = x0

    def reset(self):
        self.__init__(self.dt, self.r0, self.nsteps)

    def deterministicForce(self, v):

        return v

    def noise(self, D):
        '''
        Gaussian white noise
        '''
        return np.sqrt(2 * D / self.dt) * np.random.randn()

    def runSimulation(self, v, D):
        for index, time in enumerate(self.t[1:]):
            pos_old = self.pos[index]
            pos_new = pos_old + (self.deterministicForce(v) + self.noise(D)) * self.dt
            self.pos[index + 1] = np.mod(pos_new + 0.5, 1) - 0.5
