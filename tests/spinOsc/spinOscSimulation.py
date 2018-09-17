'''
Simulation of a Brownian particle in a 2D force-field, described by the Langevin
Equation:

dr/dt = F + xi

with

F = -k*r + alpha * curl(z,r)

The k term represents a harmonic potential
the alpha term represents a rotational, non-conservative force for the potential
xi is Gaussian white noise with strength sqrt(2*gamma*D), D is the diffusion constant
which is given by the Einstein-relation D = kB*T/gamma
'''


import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl


# Parameters
gamma = 1  # drag on particle
kT = 0.1  # energy in g * um^2 / s^2
D = kT / gamma
dt = 1e-2  # time step
k = 1  # spring constant of harmonic potential
alpha = 1.5  # strength of rotational force
nframes = int(2e3)


class Langevin():
    '''
    class for solving the 2D overdamped langevin equation

    gamma * dr = F * dt + xi * dt

    gamma is a drag coefficient
    F(r) = -k * (x,y) + alpha * (-y, x)
        k is spring constant
        alpha is strength of rotational force
    xi is zero mean Gaussian white noise
        <xi> = 0
        <xi_i (t) xi_j (t')> = 2*D*gamma * delta_ij * delta(t - t')
            D is diffusion constant, D = kB*T/gamma

    '''

    def __init__(self, dt, tfinal=10, alpha=1, k=1,
                 kT=4e-9, gamma=2e-8, r0=np.zeros(2)):
        self.dt = dt  # time step in seconds
        self.tfinal = tfinal  # final time in seconds
        self.nsteps = self.tfinal / self.dt  # total number of steps
        self.alpha = alpha  # strength of rotational force in Newtons /
        self.k = k  # strength of harmonic potential in
        self.gamma = gamma  # drag on particle in kg / s
        self.kT = kT  # in kg * microns^2 / seconds^2

        # derived quantities
        self.D = self.kT / self.gamma

        # data
        self.t = np.linspace(0, self.tfinal, self.nsteps + 1)
        self.pos = r0 * np.ones((self.nsteps + 1, 2))

    def reset(self):
        self.__init__(self.dt, self.tfinal, self.alpha,
                      self.k, self.kT, self.gamma,
                      self.r0)

    def springForce(self, r):
        '''
        harmonic force
        '''
        return -self.k * r

    def rotationalForce(self, r):
        '''
        rotational force
        '''
        zhat = np.array([0, 0, 1])
        return self.alpha * np.cross(zhat, np.append(r, 0))[:2]

    def noise(self):
        '''
        Gaussian white noise
        '''
        return np.sqrt(2 * self.gamma * self.D / self.dt) * np.random.randn(2)

    def runSimulation(self):
        for index, time in enumerate(self.t[1:]):
            pos_old = self.pos[index]
            pos_new = pos_old + (self.springForce(pos_old) + self.rotationalForce(pos_old) + self.noise()) * self.dt
            self.pos[index + 1] = pos_new
