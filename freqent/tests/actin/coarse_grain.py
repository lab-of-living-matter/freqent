import numpy as np
from itertools import product
import warnings
import argparse

def coarse_grain(im, winsize, overlap):
    '''
    Coarse grain a 2D array by getting average value in chunks of winsize that
    overlap by a given fraction

    Parameters
    ----------
    im: array-like
        image to be coarse grained in array-like structure
    winsize: int (odd)
        size of window to do averaging in pixels. Must be odd number
    overlap : float in [0, 1)
        fraction of overlap between adjacent windows in both directions. Overlap
        of 0 means no overlap, overlap of 1 means windows are on top of each other.
        Really shouldn't be less than 0.25

    Results
    -------
    cgim : array-like
        coarse-grained image

    See also
    --------

    Example
    -------

    '''
    winsize = int(winsize)

    if overlap >= 1 or overlap < 0:
        raise ValueError('overlap must be a float less than 1, greater than or equal to 0\n'
                         'Current value is {0}'.format(overlap))
    elif overlap < 0.25:
        raise warnings.warn('overlap = {0}, not recommended to not go below 0.25'.format(overlap),
                            RuntimeWarning)

    winspace = int(winsize - np.ceil(winsize * overlap))
    winrad = int(np.floor(winsize / 2))

    # get rows and columns of centers of sample windows
    sample_rows = np.arange(winrad, im.shape[0] - winrad, winspace)
    sample_cols = np.arange(winrad, im.shape[1] - winrad, winspace)

    # allocate space for coarse-grained image
    cgim = np.zeros((len(sample_rows), len(sample_cols)))

    idx_pairs = list(product(sample_rows, sample_cols))

    for idx in idx_pairs:
        cgim_row = np.where(sample_rows == idx[0])[0][0]
        cgim_col = np.where(sample_cols == idx[1])[0][0]
        cgim[cgim_row, cgim_col] = np.mean(im[idx[0] - winrad:idx[0] + winrad,
                                              idx[1] - winrad:idx[1] + winrad])

    return cgim
