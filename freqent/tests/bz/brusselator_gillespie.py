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
from datetime import datetime
from scipy import sparse
# import numba


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
    def __init__(self, population_init, rates, V, t_points, seed=None):
        self.rates = rates  # reaction rates given in docstring
        self.V = V  # volume of reaction spaces
        self.t_points = t_points  # time points to output simulation results
        self.pop0 = population_init  # store initial point, [X, Y, A, B, C]
        self.epr = np.zeros(len(t_points))  # store entropy production rate
        # self.occupancy = sparse.dok_matrix((self.V * 20, self.V * 20))
        self.occupancy = np.zeros((self.V * 15, self.V * 15))

        if seed is None:
            self.seed = datetime.now().microsecond
        else:
            self.seed = seed

        # find equilibrium point (only important if detailed balance met)
        X, Y, A, B, C = self.pop0
        k0, k1, k2, k3, k4, k5 = self.rates

        self.eq = np.array([A * k0 / k1, A * k0 * k5 / (k1 * k4)])

        # update rule for brusselator reactions given above
        # this assumes A, B, and C remain unchanged
        self.update = np.array([[1, 0, 0, 0, 0],    # A -> X, k0
                                [-1, 0, 0, 0, 0],   # X -> A, k1
                                [-1, 1, 0, 0, 0],   # B + X -> Y + C, k2
                                [1, -1, 0, 0, 0],   # Y + C -> B + X, k3
                                [1, -1, 0, 0, 0],   # 2X + Y -> 3X, k4
                                [-1, 1, 0, 0, 0]])  # 3X -> 2X + Y, k5

        # update rule for brusselator reactions given above
        # this assumes A, B, and C change
        # self.update = np.array([[1, 0, -1, 0, 0],   # A -> X, k0
        #                         [-1, 0, 1, 0, 0],   # X -> A, k1
        #                         [-1, 1, 0, -1, 1],  # B + X -> Y + C, k2
        #                         [1, -1, 0, 1, -1],  # Y + C -> B + X, k3
        #                         [1, -1, 0, 0, 0],   # 2X + Y -> 3X, k4
        #                         [-1, 1, 0, 0, 0]])  # 3X -> 2X + Y, k5

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
        r2 = np.random.rand()  # or just one?

        # pick time of next reaction
        dt = -np.log(r1) / props_sum

        # pick next reaction
        reaction = self.sample_discrete(probs, r2)

        return reaction, dt

    def runSimulation(self):
        '''
        Main loop to run gillespie algorithm.
        save output at specified time points.
        '''
        # set time and time index
        n = 0  # use to track how many reactions occur
        t = 0.0
        i = 0
        i_time = 1
        pop = np.asarray(self.pop0).copy()

        # set seed
        np.random.seed(self.seed)

        # current = np.zeros(self.update.shape[0])

        while i < len(self.t_points):
            # current = np.zeros(self.update.shape[0])

            while t < self.t_points[i_time]:
                n += 1
                reaction, dt = self.gillespie_draw(pop)

                # # add to current
                # current[reaction] += 1 / dt

                # update population
                pop_prev = pop.copy()
                pop += self.update[reaction, :]

                # track population. Keep Y in rows, X in columns
                self.occupancy[pop_prev[1], pop_prev[0]] += dt

                # increment time
                t += dt
                txt = 't = {time:.3f}'.format(time=t)
                print(txt, end='\r')

            # update index
            i = np.searchsorted(self.t_points > t, True)

            # update population
            self.population[i_time:min(i, len(self.t_points))] = pop_prev

            # update entropy production rate at this point
            # self.epr[i_time:min(i, len(self.t_points))] = (np.sum((current[::2] - current[1::2]) *
            #                                                np.log(current[::2] / current[1::2])))

            # increment index
            i_time = i

        self.occupancy /= self.t_points.max()
