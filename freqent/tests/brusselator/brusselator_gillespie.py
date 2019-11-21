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
import pdb
from numba import jit


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

    Parameters
    ----------
    population_init : array-like
        2 element array with initial population of X and Y
    rates : array-like
        6 element array with reaction rates in format
        [k1plus, k1minus, k2plus, k2minus, k3plus, k3minus,]
    V : scalar
        Reaction volume. sets size of stochastic effects
    t_points : array-like
        length N array of time points that simulation should output in.

    Returns
    -------
    f : class object
        class object with the following properties:
            population : Nx2 array of (X, Y) over time
            rates : reaction rates
            V : reaction volume
            t_points : time points, length N
            n : total number of simulation time steps
            reationTypeTracker : fraction of each reaction that occurs
            ep : entropy produced over time
            ep_blind : blinded entropy produced over time
            chemostat : values of (A, B, C) held constant

        and the following methods:
            reset() : reset to initial conditions
            runSimulation() : run a simulation

        All other methods only need to be run internally

    '''

    def __init__(self, population_init, rates, V, t_points, seed=None):
        self.rates = rates  # reaction rates given in docstring
        self.V = V  # volume of reaction space
        self.t_points = t_points  # time points to output simulation results
        self.n = 0  # number of simulation time steps taken
        self.reactionTypeTracker = np.zeros(6)  # track which reactions take place

        X, Y, A, B, C = population_init
        k0, k1, k2, k3, k4, k5 = self.rates

        self.pop0 = [X, Y]  # store initial point, [X, Y, A, B, C]
        self.ep = np.zeros(len(t_points))  # store entropy production time series
        self.ep_blind = np.zeros(len(t_points))  # store entropy production blind to specific reaction pathway
        # self.occupancy = np.zeros((self.V * 15, self.V * 15))

        if seed is None:
            self.seed = datetime.now().microsecond
        else:
            self.seed = seed

        # find equilibrium point (only important if detailed balance met)
        self.eq = np.array([A * k0 / k1, A * k0 * k5 / (k1 * k4)])

        # update rule for brusselator reactions given above
        # this assumes A, B, and C remain unchanged
        self.update = np.array([[1, 0],    # A -> X, k0
                                [-1, 0],   # X -> A, k1
                                [-1, 1],   # B + X -> Y + C, k2
                                [1, -1],   # Y + C -> B + X, k3
                                [1, -1],   # 2X + Y -> 3X, k4
                                [-1, 1]])  # 3X -> 2X + Y, k5

        # preallocate space for evolved population
        self.population = np.zeros((len(self.t_points), 2), dtype=int)
        self.population[0, :] = self.pop0
        self.chemostat = np.array([A, B, C])

        # # update rule for brusselator reactions given above
        # # this assumes A, B, and C change
        # self.update = np.array([[1, 0, -1, 0, 0],   # A -> X, k0
        #                         [-1, 0, 1, 0, 0],   # X -> A, k1
        #                         [-1, 1, 0, -1, 1],  # B + X -> Y + C, k2
        #                         [1, -1, 0, 1, -1],  # Y + C -> B + X, k3
        #                         [1, -1, 0, 0, 0],   # 2X + Y -> 3X, k4
        #                         [-1, 1, 0, 0, 0]])  # 3X -> 2X + Y, k5
        #
        # # preallocate space for evolved population
        # self.population = np.zeros((len(self.t_points), len(population_init)), dtype=int)
        # self.population[0, :] = self.pop0

    def reset(self):
        self.__init__(self.pop0, self.rates, self.V, self.t_points, self.seed)

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
        props : array
            array with propensities measured given the current population
        '''
        X, Y, A, B, C = population
        k0, k1, k2, k3, k4, k5 = self.rates

        props = np.array([k0 * A,
                          k1 * X,
                          k2 * B * X / self.V,
                          k3 * C * Y / self.V,
                          k4 * X * (X - 1) * Y / self.V**2,
                          k5 * X * (X - 1) * (X - 2) / self.V**2])
        return props

    @jit
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

    @jit
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

        return reaction, dt, props

    def runSimulation(self):
        '''
        Main loop to run gillespie algorithm.
        save output at specified time points.
        '''
        # set time and time index
        t = 0.0
        i = 0
        i_time = 1
        ep = 0
        ep_blind = 0

        # set seed
        np.random.seed(self.seed)

        # do first random draw
        pop = np.asarray(self.pop0).copy()
        reaction, dt, props = self.gillespie_draw(np.concatenate((pop, self.chemostat)))
        while i < len(self.t_points):
            # current = np.zeros(self.update.shape[0])
            while t < self.t_points[i_time]:
                self.n += 1
                self.reactionTypeTracker[reaction] += 1
                # update population
                pop_prev = pop.copy()
                pop += self.update[reaction, :]

                if (pop < 0).any():
                    pdb.set_trace()

                # track population. Keep Y in rows, X in columns
                # self.occupancy[pop_prev[1], pop_prev[0]] += dt

                # Calculate trajectory entropy. On the way, calculate next set of random draws
                # Do next Gillespie draw
                reaction_next, dt_next, props_next = self.gillespie_draw(np.concatenate((pop, self.chemostat)))

                # Find backwards reaction from what was just done
                # [0, 2, 4] <--> [1, 3, 5]
                # If reaction is an even number (or 0) add one, if an odd number, subtract one
                backward_reaction = reaction + (-1)**(reaction % 2)

                # add to entropy
                ep += np.log((props[reaction] / props.sum()) /
                             (props_next[backward_reaction] / props_next.sum()))

                # Note that reactions [2, 5] and [3, 4] each give the same dynamics of
                # (X, Y) -> (X - 1, Y + 1) and (X, Y) -> (X + 1, Y - 1), respectively
                if reaction in [0, 1]:
                    ep_blind += np.log((props[reaction] / props.sum()) /
                                       (props_next[backward_reaction] / props_next.sum()))
                elif reaction in [2, 5]:
                    ep_blind += np.log((props[[2, 5]].sum() / props.sum()) /
                                       (props_next[[3, 4]].sum() / props_next.sum()))
                elif reaction in [3, 4]:
                    ep_blind += np.log((props[[3, 4]].sum() / props.sum()) /
                                       (props_next[[2, 5]].sum() / props_next.sum()))

                # increment time
                t += dt

                # update reaction, dt, and propensities
                reaction, dt, props = reaction_next, dt_next, props_next

            # update index
            i = np.searchsorted(self.t_points > t, True)

            # update population
            self.population[i_time:min(i, len(self.t_points))] = pop_prev
            self.ep[i_time:min(i, len(self.t_points))] = ep
            self.ep_blind[i_time:min(i, len(self.t_points))] = ep_blind

            # increment index
            i_time = i

        # track percentage of each reaction done
        self.reactionTypeTracker /= self.n
        # self.occupancy /= self.t_points.max()
