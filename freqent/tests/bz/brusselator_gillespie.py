'''
Brusselator given by

           k0
        A <==> X
           k1

           k2
    B + X <==> C + XY
           k3

           k4
2 * X + Y <==> 3 * X
           k5

'''

import numpy as np


def propensities_brusselator(population, rates, V):
    '''
    Calculates propensities for the reversible brusselator to be used
    in the Gillespie algorithm

    Parameters
    ----------
    population : array-like
        array with number of intermediate species in the mixture.
        Given in order [X, Y, A, B, C]
    params: array-like
        array with kinetic rate constants for reactions, ordered as at beginning
        of this file, and chemostated
    V : scalar
        volume of mixture
    '''
    X, Y, A, B, C = np.asarray(population)
    k0, k1, k2, k3, k4, k5 = np.asarray(rates)

    a = np.array([k0 * A,
                  k1 * X,
                  k2 * B * X / V,
                  k3 * C * Y / V,
                  k4 * X * (X - 1) * Y / V**2,
                  k5 * X * (X - 1) * (X - 2) / V**2])
    return a


def sample_discrete(probs, r):
    '''
    Randomly sample an index with probability given by probs (assumed normalized)
    '''
    n = 1
    p_sum = probs[0]
    while p_sum < r:
        p_sum += probs[n]
        n += 1
    return n - 1


def gillespie_draw(population, rates, V, propensity_func):

    # get propensities
    props = propensity_func(population, rates, V)
    props_sum = props.sum()
    probs = props / props_sum

    # draw time of next reaction
    dt = np.random.exponential(1 / props_sum)

    # pick next reaction
    r = np.random.rand()
    rxn = 1
    p_sum = probs[0]
    while p_sum < r:
        p_sum += probs[rxn]
        rxn += 1

    reaction = rxn - 1

    return reaction, dt


def gillespie_simulator(population_init, rates, V,
                        propensity_func, update, t_points):
    '''
    Main loop to run gillespie algorithm.
    Produce output at specified time points.

    Parameters
    ----------

    Returns
    -------

    '''
    # initialize array to output population
    pop_out = np.empty((len(population_init), len(t_points)), dtype=int)

    # set time and time index
    t = 0.0
    i = 0
    i_time = 1
    pop = population_init.copy()
    pop_out[0, :] = pop
    while i < len(t_points):
        while t < t_points[i_time]:
            reaction, dt = gillespie_draw(pop, rates, V, propensity_func)

            # update population
            pop_prev = pop.copy()
            pop += update[reaction, :]

            # increment time
            t += dt

        # update index
        i = np.searchsorted(t_points > t, True)

        # update population
        pop_out[i_time:min(i, len(t_points))] = pop_prev

        # increment index
        i_time = i

    return pop_out
