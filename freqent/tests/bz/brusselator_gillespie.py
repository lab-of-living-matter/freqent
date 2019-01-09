'''
Brusselator given by

           k0
        A <==> X
           k1

           k2
    B + X <==> C + Y
           k3

           k4
2 * X + Y <==> 3 * X
           k5

'''

import numpy as np
import numba


class brusselatorStochSim():
    '''
    Class for evolving a Brusselator using Gillespie's algorithm.

               k0
            A <==> X
               k1

               k2
        B + X <==> C + Y
               k3

               k4
    2 * X + Y <==> 3 * X
               k5

    All chemical quantities are given as total number of molecules,
    not concentrations.
    '''
    def __init__(self, population_init, rates, V, t_points):
        self.rates = rates  # reaction rates given in docstring
        self.V = V  # volume of reaction spaces
        self.t_points = t_points  # time points to output simulation results
        self.pop0 = population_init  # store initial point, [X, Y, A, B, C]

        # find and store steady state
        x_ss = population_init[2] * rates[0] / rates[1]
        y_ss = self.V * x_ss * ((rates[2] * (population_init[3] / V) + rates[5] * x_ss**2) /
                                (rates[3] * (population_init[4] / V) + rates[4] * x_ss**2))
        self.ss = np.array([x_ss, y_ss])

        # find equilibrium point (only important if detailed balance met)
        self.eq = np.array([population_init[2] * rates[0] / rates[1],
                            population_init[2] * rates[0] * rates[5] / (rates[1] * rates[4])])

        # update rule for brusselator reactions given above
        # this assumes A, B, and C remain unchanged
        self.update = np.array([[1, 0, 0, 0, 0],
                                [-1, 0, 0, 0, 0],
                                [-1, 1, 0, 0, 0],
                                [1, -1, 0, 0, 0],
                                [1, -1, 0, 0, 0],
                                [-1, 1, 0, 0, 0]])

        # preallocate space for evolved population
        self.population = np.empty((len(self.t_points), len(population_init)), dtype=int)
        self.population[0, :] = self.pop0

    def reset(self):
        self.__init__(self.pop0, self.rates, self.V, self.t_points)

    def propensities_brusselator(self, population):
        '''
        Calculates propensities for the reversible brusselator to be used
        in the Gillespie algorithm

        Parameters
        ----------
        population : array-like
            array with current number of species in the mixture.
            Given in order [X, Y, A, B, C]

        Returns
        -------
        a : array
            array with propensities measured given the current population
        '''
        X, Y, A, B, C = population
        k0, k1, k2, k3, k4, k5 = self.rates

        a = np.array([k0 * A,
                      k1 * X,
                      k2 * B * X / self.V,
                      k3 * C * Y / self.V,
                      k4 * X * (X - 1) * Y / self.V**2,
                      k5 * X * (X - 1) * (X - 2) / self.V**2])
        return a

    def sample_discrete(self, probs, r):
        '''
        Randomly sample an index with probability given by probs (assumed normalized)
        '''
        n = 1
        p_sum = probs[0]
        while p_sum < r:
            p_sum += probs[n]
            n += 1
        return n - 1

    def gillespie_draw(self, population):

        # get propensities
        props = self.propensities_brusselator(population)
        props_sum = props.sum()
        probs = props / props_sum

        # draw two random numbers
        r1 = np.random.rand()
        # r2 = np.random.rand()  # or just one?

        # pick time of next reaction
        dt = -np.log(r1) / props_sum

        # pick next reaction
        reaction = self.sample_discrete(probs, r1)

        return reaction, dt

    def gillespie_simulator(self):
        '''
        Main loop to run gillespie algorithm.
        Produce output at specified time points.
        '''

        # set time and time index
        t = 0.0
        i = 0
        i_time = 1
        pop = np.asarray(self.pop0).copy()

        while i < len(self.t_points):
            while t < self.t_points[i_time]:
                reaction, dt = self.gillespie_draw(pop)

                # update population
                pop_prev = pop.copy()
                pop += self.update[reaction, :]

                # increment time
                t += dt

            # update index
            i = np.searchsorted(self.t_points > t, True)

            # update population
            self.population[i_time:min(i, len(self.t_points))] = pop_prev

            # increment index
            i_time = i

    def plot_traj(self):
        '''

        '''


# update = np.array([[1, 0, 0, 0, 0],
#                    [-1, 0, 0, 0, 0],
#                    [-1, 1, 0, 0, 0],
#                    [1, -1, 0, 0, 0],
#                    [1, -1, 0, 0, 0],
#                    [-1, 1, 0, 0, 0]])
# rates = [0.5, 0.25, 1, 0.25, 1, 0.25]
# V = 100
# A = V
# B = V
# C = B * rates[2] * rates[4] / (rates[3] * rates[5])
# X0 = 10
# Y0 = 10
# pop_init = [X0, Y0, A, B, C]
# t_points = np.linspace(0, 100, 501)

# pop = gillespie_simulator(pop_init, rates, V, propensities_brusselator,
#                           update, t_points)
