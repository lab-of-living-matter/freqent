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
from numba import jit, jitclass, int32, float32


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


spec = [('rates', float32[:]),
        ('V', float32),
        ('t_points', float32[:]),
        ('D', float32[:]),
        ('h', float32),
        ('K', int32),
        ('XY0', int32[:, :]),
        ('ABC', int32[:]),
        ('ep', float32[:]),
        ('seed', int32),
        ('props', float32[:]),
        ('update', float32[:, :]),
        ('population', float32[:, :, :])]


# @jitclass(spec)
class brusselator1DFieldStochSim():
    '''
    Class for evolving a 1D reaction-diffusion equation
    The reactions are those of the Brusselator

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

    Diffusion is modeled as two chains of "chemical reactions" between
    K different subvolumes

         d        d        d        d
    X_1 <==> X_2 <==> X_3 <==> ... <==> X_K
         d        d        d        d

         d        d        d        d
    Y_1 <==> Y_2 <==> Y_3 <==> ... <==> Y_K
         d        d        d        d

    Where d = D / h^2, where h is the length of the subvolumes.
    '''

    def __init__(self, XY_init, ABC, rates, t_points,
                 D, n_subvolumes, v_subVolumes, seed=None):
        self.rates = rates  # reaction rates given in docstring
        # self.V = V  # volume of each reaction subvolume
        self.t_points = t_points  # time points to output simulation results
        self.D = np.asarray(D, dtype=np.float32)  # diffusion constant for X and Y, as 2-element array [D_X, D_Y]
        self.V = v_subVolumes  # length of subvolumes
        self.K = n_subvolumes  # number of subvolumes
        self.XY0 = XY_init  # 2 by K array of initial values for X and Y
        self.ABC = np.asarray(ABC, dtype=np.float32)  # number of chemostatted molecules IN EACH SUBVOLUME
        self.ep = np.zeros(len(t_points), dtype=np.float32)  # store entropy production time series
        self.n = 0  # keep track of how many reactions are being done
        self.reactionTypeTracker = np.zeros(10)  # track which reactions take place
        self.seed = seed

        # Store propensities for t=0, to be updated throughout simulation
        # Order given as (each line is K reactions)
        #     - X diffusing clockwise (1 -> 2 ->...-> K -> 1)
        #     - X diffusing counter-clockwise (K -> K-1 ->...-> 2 -> 1 -> K)
        #     - Y diffusing clockwise (1 -> 2 ->...-> K -> 1)
        #     - Y diffusing counter-clockwise (K -> K-1 ->...-> 2 -> 1 -> K)
        #     - k0 reaction for each cell
        #     - k1 reaction for each cell
        #     - k2 reaction for each cell
        #     - k3 reaction for each cell
        #     - k4 reaction for each cell
        #     - k5 reaction for each cell
        X, Y = XY_init
        k0, k1, k2, k3, k4, k5 = self.rates
        dx = self.D[0] / self.V**2  # diffusion rate of X molecule
        dy = self.D[1] / self.V**2  # diffusion rate of Y molecule
        A = np.repeat(self.ABC[0], self.K)
        B = np.repeat(self.ABC[1], self.K)
        C = np.repeat(self.ABC[2], self.K)

        x_cw = X * dx
        x_ccw = X * dx
        y_cw = Y * dy
        y_ccw = Y * dy
        k0_reaction = k0 * A
        k1_reaction = k1 * X
        k2_reaction = k2 * B * X / self.V
        k3_reaction = k3 * C * Y / self.V
        k4_reaction = k4 * X * (X - 1) * Y / self.V**2
        k5_reaction = k5 * X * (X - 1) * (X - 2) / self.V**2

        self.props = np.concatenate((x_cw,
                                     x_ccw,
                                     y_cw,
                                     y_ccw,
                                     k0_reaction,
                                     k1_reaction,
                                     k2_reaction,
                                     k3_reaction,
                                     k4_reaction,
                                     k5_reaction))

        # # find equilibrium point (only important if detailed balance met)
        # self.eq = np.array([A * k0 / k1, A * k0 * k5 / (k1 * k4)])

        # write update matrices. See Notability note from 1/22/2019 for details

        # create diagonal matrix to move X_i -> X_i+1
        # with periodic b.c., X_K -> X_1
        # start at X_1
        x_cw_update = (-np.eye(self.K, dtype=int) +
                       np.eye(self.K, k=1, dtype=int) +
                       np.eye(self.K, k=-(self.K - 1), dtype=int))
        # y clockwise is the same
        y_cw_update = x_cw_update

        # create anti-diagonal matrix to move X_i -> X_i-1
        # with periodic b.c., X_1 -> X_K
        # start at X_1
        x_ccw_update = (-np.eye(self.K, dtype=int) +
                        np.eye(self.K, k=-1, dtype=int) +
                        np.eye(self.K, k=(self.K - 1), dtype=int))
        # y counter-clockwise is the same
        y_ccw_update = x_ccw_update

        x_update = np.vstack((x_cw_update,  # i -> i+1 diffusion
                              x_ccw_update,  # i -> i-1 diffusion
                              np.zeros((self.K, self.K), dtype=int),  # x doesn't change
                              np.zeros((self.K, self.K), dtype=int),  # when y diffuses
                              np.eye(self.K, dtype=int),  # A -> X, k0
                              -np.eye(self.K, dtype=int),  # X -> A, k1
                              -np.eye(self.K, dtype=int),  # B + X -> Y + C, k2
                              np.eye(self.K, dtype=int),  # Y + C -> B + X, k3
                              np.eye(self.K, dtype=int),  # 2X + Y -> 3X, k4
                              -np.eye(self.K, dtype=int)))  # 3X -> 2X + Y, k5

        y_update = np.vstack((np.zeros((self.K, self.K), dtype=int),  # y doesn't change
                              np.zeros((self.K, self.K), dtype=int),  # when x diffuses
                              y_cw_update,  # i -> i+1 diffusion
                              y_ccw_update,  # i -> i-1 diffusion
                              np.zeros((self.K, self.K), dtype=int),  # A -> X, k0
                              np.zeros((self.K, self.K), dtype=int),  # X -> A, k1
                              np.eye(self.K, dtype=int),  # B + X -> Y + C, k2
                              -np.eye(self.K, dtype=int),  # Y + C -> B + X, k3
                              -np.eye(self.K, dtype=int),  # 2X + Y -> 3X, k4
                              np.eye(self.K, dtype=int)))  # 3X -> 2X + Y, k5

        # Build update matrix index by [reaction, species, compartment]
        # with size [10*K, 2, K]
        self.update = np.stack((x_update, y_update), axis=1)

        # preallocate space for evolved population
        # Dimensions are [time, chemical_species, subvolume]
        self.population = np.zeros((len(self.t_points), 2, self.K), dtype=int)
        self.population[0, :] = self.XY0

    def reset(self):
        self.__init__(self.XY0, self.ABC, self.rates, self.t_points,
                      self.D, self.K, self.V)

    # Functions to update the propensities with
    # XY is current population as 2xK array
    def reaction_update(self, XY, compartment, reactionType):
        if reactionType in [0, 1]:
            dx = self.D[0] / self.V**2
            return XY[0, compartment] * dx

        elif reactionType in [2, 3]:
            dy = self.D[1] / self.V**2
            return XY[1, compartment] * dy

        elif reactionType == 4:
            return self.rates[0] * self.ABC[0]

        elif reactionType == 5:
            X = XY[0, compartment]
            return self.rates[1] * X

        elif reactionType == 6:
            X = XY[0, compartment]
            return self.rates[2] * self.ABC[1] * X / self.V

        elif reactionType == 7:
            Y = XY[1, compartment]
            return self.rates[3] * self.ABC[2] * Y / self.V

        elif reactionType == 8:
            X = XY[0, compartment]
            Y = XY[1, compartment]
            return self.rates[4] * X * (X - 1) * Y / self.V**2

        elif reactionType == 9:
            X = XY[0, compartment]
            return self.rates[5] * X * (X - 1) * (X - 2) / self.V**2

        else:
            raise ValueError('reactionType should be between 0 and 9.\n'
                              'Entered reactionType is {0}'.format(reactionType))

    def update_propensities(self, XY, reaction):
        '''
        update the propensities of the 1D reaction diffusion system
        Anything that changes the quantity of X must change reactions
        0, 1, 5, 6, 8, 9

        Anything that changes the quantity of Y must change reactions
        2, 3, 7, 8

        Reaction 4 never changes (fixed by quantity of A)

        For order of reactions, see self.__init__

        Parameters
        ----------
        XY : array-like
            2xK array of current population used to update the
            brusselator propensities
        reaction : int
            Which reaction was chosen, integer from 0 to 10K - 1

        Returns
        -------
        self.props : array-like
            updated propensities array
        '''
        dependsOnX = [0, 1, 5, 6, 8, 9]
        dependsOnY = [2, 3, 7, 8]

        reactionType = reaction // self.K
        compartment = reaction % self.K

        if reactionType == 0:
            # X diffuses to the right
            # Take into account periodic boundary conditions
            rightCompartment = (compartment + 1) - ((compartment + 1) // self.K) * self.K
            toChange = np.sort([a * self.K + compartment for a in dependsOnX] +
                               [a * self.K + rightCompartment for a in dependsOnX])
        elif reactionType == 1:
            # X diffuses to the left
            # Take into account periodic boundary conditions
            leftCompartment = (compartment - 1) - ((compartment - 1) // self.K) * self.K
            toChange = np.sort([a * self.K + compartment for a in dependsOnX] +
                               [a * self.K + leftCompartment for a in dependsOnX])
        elif reactionType == 2:
            # Y diffuses to the right
            # Take into account periodic boundary conditions
            rightCompartment = (compartment + 1) - ((compartment + 1) // self.K) * self.K

            toChange = np.sort([a * self.K + compartment for a in dependsOnY] +
                               [a * self.K + rightCompartment for a in dependsOnY])
        elif reactionType == 3:
            # Y diffuses to the left
            # Take into account periodic boundary conditions
            leftCompartment = (compartment - 1) - ((compartment - 1) // self.K) * self.K
            toChange = np.sort([a * self.K + compartment for a in dependsOnY] +
                               [a * self.K + leftCompartment for a in dependsOnY])
        elif reactionType in [4, 5]:
            # A <=> X
            toChange = np.sort([a * self.K + compartment for a in dependsOnX])
        elif reactionType in [6, 7, 8, 9]:
            # B + X <=> C + Y and 2X + Y <=> 3X
            toChange = np.sort([a * self.K + compartment for a in dependsOnX + dependsOnY])
        else:
            raise ValueError('Reaction number should be between 0 and {nReact}.\n Current number is {m}'.format(nReact=10 * self.K - 1, m=reaction))

        for propInd in toChange:
            # Can't use compartment as second argument because diffusion also changes
            # quantities in neighboring compartments
            self.props[propInd] = self.reaction_update(XY, propInd % self.K, propInd // self.K)

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
    def gillespie_draw(self):

        # get propensities
        props_sum = self.props.sum()
        probs = self.props / props_sum

        # draw two random numbers
        r1 = np.random.rand()
        r2 = np.random.rand()  # or just one?

        # pick time of next reaction
        dt = -np.log(r1) / props_sum

        # pick next reaction
        reaction = self.sample_discrete(probs, r2)

        return reaction, dt, probs

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

        # set seed
        np.random.seed(self.seed)

        # do first random draw
        pop = np.asarray(self.XY0).copy()
        reaction, dt, probs = self.gillespie_draw()
        while i < len(self.t_points):
            # current = np.zeros(self.update.shape[0])
            while t < self.t_points[i_time]:
                self.n += 1  # add 1 to reactions taken
                # update population
                pop_prev = pop.copy()
                pop += self.update[reaction, ...]

                if (pop < 0).any():
                    pdb.set_trace()

                # update propensities
                self.update_propensities(pop, reaction)
                if np.isnan(self.props).sum():
                    pdb.set_trace()

                # track population. Keep Y in rows, X in columns
                # self.occupancy[pop_prev[1], pop_prev[0]] += dt

                # Calculate trajectory entropy. On the way, calculate next set of random draws
                # Do next Gillespie draw
                reaction_next, dt_next, probs_next = self.gillespie_draw()

                # Find backwards reaction from what was just done
                # First, get the reaction type
                reactionType = reaction // self.K
                # Then get the backward reaction given the reaction type
                if reactionType in [0, 2]:
                    # if diffuse to right, diffuse to left from next compartment over
                    # make sure to take care of boundary condition

                    # This expression was attained after a lot of trial and error...
                    # it returns [K+1, K+1, K+1, K+1, K+1, K+1, K+1, K+1, K+1, 1]
                    left_diffusion = 1 + self.K - ((reaction % self.K + 1) // self.K) * self.K

                    backward_reaction = reaction + left_diffusion
                elif reactionType in [1, 3]:
                    # if diffuse to left, diffuse to right from next compartment over
                    # make sure to take care of boundary condition

                    # This expression was attained after a lot of trial and error...
                    # it returns [-1, -K-1, -K-1, -K-1, -K-1, -K-1, -K-1, -K-1, -K-1, -K-1]
                    right_diffusion = -1 - self.K - ((reaction % self.K - 1) // self.K) * self.K

                    backward_reaction = reaction + right_diffusion
                elif reactionType in [4, 6, 8]:
                    # "forward" chemical reactions
                    backward_reaction = reaction + self.K
                elif reactionType in [5, 7, 9]:
                    # "backward" chemical reactions
                    backward_reaction = reaction - self.K

                # stop simulation if about to get infinite entropy
                if probs_next[backward_reaction] == 0:
                    pdb.set_trace()

                # add to entropy,
                ep += np.log(probs[reaction] / probs_next[backward_reaction])

                # Track which type of reaction just happened.
                self.reactionTypeTracker[reactionType] += 1

                # increment time
                t += dt

                # update reaction, dt, and propensities
                # reaction_prev, dt_prev, probs_prev = reaction, dt, probs
                reaction, dt, probs = reaction_next, dt_next, probs_next

            # update index
            # pdb.set_trace()
            # print(np.searchsorted(self.t_points > t, True))
            i = np.searchsorted(self.t_points > t, True)

            # update population
            self.population[i_time:min(i, len(self.t_points))] = pop_prev
            self.ep[i_time:min(i, len(self.t_points))] = ep
            self.reactionTypeTracker /= self.n

            # increment index
            i_time = i

        # self.occupancy /= self.t_points.max()
