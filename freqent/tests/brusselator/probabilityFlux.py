import numpy as np
import pdb


def probabilityFlux(data, dt=1, bins=10):
    '''
    Compute the flux field of a time series data
    *** ONLY WORKS FOR 2D DATA RIGHT NOW ***

    Parameters
    ----------
    data : array
        NxD array of time series data with N time points in D
        dimensions
    dt : float, optional
        size of time step between data points. Defaults to 1
    bins : sequence or int, optional
        The bin specification:
            - A sequence of arrays describing the monotonically increasing bin edges along each dimension.
            - The number of bins for each dimension (nx, ny, … =bins)
            - The number of bins for all dimensions (nx=ny=…=bins).
        Defaults to 10

    Results
    -------
    prob_map : array
        histogram of probabilities for each state
    flux_field : array
        vector field of fluxes in shape D x [nbins]
    edges : array
        edges used to discretize space

    See also
    --------

    Example
    -------

    '''

    data = np.asarray(data)
    nt, ndim = data.shape
    T = nt * dt

    if ndim != 2:
        raise ValueError('This function only works for 2D data')

    prob_map, edges = np.histogramdd(data, bins=bins)
    nbins = [len(e) - 1 for e in edges]

    flux_field = np.zeros((ndim, *nbins))

    for tInd, (prior_state, current_state, next_state) in enumerate(zip(data[:-2], data[1:-1], data[2:])):
        # print('\r tInd={tInd}'.format(tInd=tInd))
        prior_bin_index = np.array([[np.digitize(s, e) - 1 for s, e in zip(prior_state, edges)]])
        current_bin_index = np.array([[np.digitize(s, e) - 1 for s, e in zip(current_state, edges)]])
        next_bin_index = np.array([[np.digitize(s, e) - 1 for s, e in zip(next_state, edges)]])

        # same_bin = prior_bin_index is current_bin_index

        # if not same_bin:
        traversed_inds_before = np.concatenate((prior_bin_index,
                                                bresenhamline(prior_bin_index,
                                                              current_bin_index,
                                                              max_iter=-1)))
        traversed_inds_after = np.concatenate((current_bin_index,
                                               bresenhamline(current_bin_index,
                                                             next_bin_index,
                                                             max_iter=-1)))

        all_traversed_index = np.concatenate((traversed_inds_before, traversed_inds_after[1:]))

        # flux_vecs = all_traversed_index[2:] - all_traversed_index[:-2]
        flux_vecs = (next_bin_index - prior_bin_index) / (2 * dt)
        # pdb.set_trace()
        flux_field[:, current_bin_index[0][0], current_bin_index[0][1]] += flux_vecs[0]

        # for ind, state in enumerate(all_traversed_index[1:-1]):
        #     # this only works for 2D data

        #     # add 1 flux vector to each state the path leaves
        #     flux_field[:, state[0], state[1]] += flux_vecs[ind]

            # add 1 flux vector to each state the path enters
            # flux_field[:, in_ind[0], in_ind[1]] += flux_vecs[ind]

    # flux_field /= T

    return prob_map, flux_field, edges


def bresenhamline(start, end, max_iter=5):
    """
    N-D Bresenham line algorithm
    Directly taken from the following url (accessed 19/09/28)
    http://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/

    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])


def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.

    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])

    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])

    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope


def _bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension)

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)
