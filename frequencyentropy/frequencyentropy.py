import numpy as np
import scipy.signal as signal
import warnings


def corrMatrix(data, sample_rate=1, mode='full', method='auto', return_fft=False):
    '''
    Takes time series data of multiple variables and returns a correlation matrix
    for every lag time

    Parameters
    ----------
    data : 2D array
        Data is an NxM array that gives length M time series data of N variables.
        e.g. data[0] returns time series for first variable.
    sample_rate : scalar
        Sample rate of data in seconds. Default = 1
    mode : str {'valid', 'same', 'full'}, optional
        Refer to the 'scipy.signal.convolve' docstring. Default is 'full'.
    method : str {'auto', 'direct', 'fft'}, optional
        Refer to the 'scipy.signal.convolve' docstring. Default is 'auto'.
    return_fft : bool (optional)
        Boolean asking whether to return the temporal fourier transform of the
        correlation matrix

    Returns
    -------
    c : 3D array
        an NxNx(2M-1) matrix that gives the NxN correlation matrix for
    '''

    data = np.asarray(data)
    if data.shape[0] > data.shape[1]:
        warnings.warn('Number of rows (variables) > number of columns (time points). '
                      'Make sure data has variables as rows.')

    nvars, npts = data.shape
    c = np.zeros((nvars, nvars, npts * 2 - 1))

    # get all pairs of indices
    idx_pairs = np.array(np.meshgrid(np.arange(nvars), np.arange(nvars))).T.reshape(-1, 2)

    for idx in idx_pairs:
        c[idx[0], idx[1], :] = _correlate_mean(data[idx[0]], data[idx[1]],
                                               mode, method, return_fft)

    return c


def _correlate_mean(x1, x2, mode='full', method='auto', return_fft=False):
    '''
    Calculate cross-correlation between two time series using fourier transform
    Wrapper around scipy.signal.correlate function that takes a mean rather than
    just summing up the signal

    Parameters
    ----------
    x, y : 1D array
        data to find cross-correlation between
    return_fft : bool (optional)
        boolean asking whether to return the temporal fourier transform of the
        correlation matrix

    Returns
    -------
    xcorr : 1D array
        cross correlation between
    '''

    N = max(len(x1), len(x2))
    n = N * np.ones(N) - np.arange(N)
    n = np.concatenate((np.flipud(n)[:-1], n))
    xcorr = signal.correlate(x1, x2, mode, method) / n

    if return_fft:
        return np.fft.fftshift(np.fft.fft(xcorr))
    else:
        return xcorr
