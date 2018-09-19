'''
Simulation of a Brownian particle in a 2D force-field, described by the Langevin
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


class spinOscLangevin():
    '''
    class for solving the 2D overdamped langevin equation

    gamma * dr = F * dt + xi * dt

    gamma is a drag coefficient
    F(r) = -k * (x,y) + alpha * (-y, x)
        k is spring constant
        alpha is strength of rotational force
    xi is zero mean Gaussian white noise
        <xi> = 0
        <xi_i (t) xi_j (t')> = 2*D*gamma^2 * delta_ij * delta(t - t')
            D is diffusion constant, D = kB*T/gamma

    '''

    def __init__(self, dt, tfinal=10, kT=4e-9, gamma=2e-8, r0=np.zeros(2)):

        self.dt = dt  # time step in seconds
        self.tfinal = tfinal  # final time in seconds
        self.nsteps = int(self.tfinal / self.dt)  # total number of steps
        self.gamma = gamma  # drag on particle in kg / s
        self.kT = kT  # in kg * um^2 / s^2

        # derived quantities
        self.D = self.kT / self.gamma

        # data
        self.t = np.linspace(0, self.tfinal, self.nsteps + 1)
        self.pos = np.zeros((2, self.nsteps + 1))
        self.pos[:, 0] = r0

    def reset(self):
        self.__init__(self.dt, self.tfinal, self.alpha,
                      self.k, self.kT, self.gamma,
                      self.r0)

    def springForce(self, r, k):
        '''
        harmonic force
        '''
        return -k * r

    def rotationalForce(self, r, alpha):
        '''
        rotational force
        '''
        zhat = np.array([0, 0, 1])
        return alpha * np.cross(zhat, np.append(r, 0))[:2]

    def noise(self):
        '''
        Gaussian white noise
        '''
        return np.sqrt(2 * self.D * self.gamma**2 / self.dt) * np.random.randn(2)

    def runSimulation(self, k, alpha):
        for index, time in enumerate(self.t[1:]):
            pos_old = self.pos[:, index]
            pos_new = pos_old + (self.springForce(pos_old, k) + self.rotationalForce(pos_old, alpha) + self.noise()) * self.dt / self.gamma
            self.pos[:, index + 1] = pos_new


# def run(dt=1e-3, tfinal=1, alpha=1, k=1, kT=4e-9, gamma=2e-8, r0=np.random.rand(2)):
#     t = 0
#     while t<tfinal
