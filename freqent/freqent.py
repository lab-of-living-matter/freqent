import numpy as np
import scipy.signal as signal
import warnings
from itertools import product


def entropy(c_fft, sample_spacing=1):
    '''
    Calculate the entropy using the frequency space measure:

    dS/dt = (sum_n ((C^-1)^T_ij (f_n) - C^-1_ij(f_n)) C_ij(f_n)) / 2T

    where T is the total time of the signal, C_ij(w_n) is the (i,j)th component
    of the correlation matrix evaluated at the frequency f_n, where f_n = n/T,
    and n is in [-N/2, N/2], where N is the total number of points in the original
    signal

    Parameters
    ----------
    c_fft : 3D array
        an MxNxN matrix that gives an NxN correlation matrix as a function of
        M frequencies
    sample_spacing : float
        Sample spacing (inverse of sample rate) of data in seconds. Default = 1

    Returns
    -------
    s : float
        entropy production rate given correlation functions
    '''

    T = sample_spacing * (c_fft.shape[0] - 1)

    # get inverse of each NxN submatrix of c_fft. See the stackexchange:
    # https://stackoverflow.com/questions/41850712/compute-inverse-of-2d-arrays-along-the-third-axis-in-a-3d-array-without-loops
    c_fft_inv = np.linalg.inv(c_fft)
    s = np.sum((np.transpose(c_fft_inv, (0, 2, 1)) - c_fft_inv) * c_fft)

    s /= 2 * T

    return s


def corr_matrix(data, sample_spacing=1, mode='full', method='auto', return_fft=False):
    '''
    Takes time series data of multiple variables and returns a correlation matrix
    for every lag time

    Parameters
    ----------
    data : 2D array
        Data is an NxM array that gives length M time series data of N variables.
        e.g. data[n] returns time series for nth variable.
    sample_spacing : float
        Sample spacing (inverse of sample rate) of data in seconds. Default = 1
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
        an (2M-1)xNxN matrix that gives the NxN correlation matrix for the variables
        contained in the rows of data. Returns fft(c) is return_fft=True
    tau : array
        2M-1 length array of lag times for correlations. Returns frequencies if
        return_fft=True

    '''

    data = np.asarray(data)
    if data.shape[0] > data.shape[1]:
        warnings.warn('Number of rows (variables) > number of columns (time points). '
                      'Make sure data has variables as rows.')

    nvars, npts = data.shape
    c = np.zeros((npts * 2 - 1, nvars, nvars), dtype=complex)

    # get all pairs of indices
    idx_pairs = list(product(np.arange(nvars), repeat=2))

    for idx in idx_pairs:
        c[:, idx[0], idx[1]] = _correlate_mean(data[idx[0]], data[idx[1]], sample_spacing,
                                               mode, method, return_fft)

    if not return_fft:
        c = c.real
        maxTau = sample_spacing * (npts - 1)
        tau = np.linspace(-maxTau, maxTau, 2 * npts - 1)
        return c, tau
    else:
        freqs = np.fft.fftshift(np.fft.fftfreq(2 * npts - 1, d=sample_spacing))
        return c, freqs


def _correlate_mean(x1, x2, sample_spacing=1, mode='full', method='auto', return_fft=False):
    '''
    Calculate cross-correlation between two time series using fourier transform
    Wrapper around scipy.signal.correlate function that takes a mean rather than
    just summing up the signal

    Parameters
    ----------
    x1, x2 : 1D array
        data to find cross-correlation between
    mode : str {'valid', 'same', 'full'}, optional
        Refer to the 'scipy.signal.convolve' docstring. Default is 'full'.
    method : str {'auto', 'direct', 'fft'}, optional
        Refer to the 'scipy.signal.convolve' docstring. Default is 'auto'.
    return_fft : bool (optional)
        boolean asking whether to return the temporal fourier transform of the
        correlation matrix

    Returns
    -------
    xcorr : 1D array
        cross correlation between x1 and x2. Returns fft(xcorr) if return_fft=True

    See also
    --------
    scipy.signal.correlate
    '''

    N = max(len(x1), len(x2))
    n = N * np.ones(N) - np.arange(N)
    n = np.concatenate((np.flipud(n)[:-1], n))
    xcorr = signal.correlate(x1, x2, mode, method) / n

    if return_fft:
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(xcorr))) * sample_spacing
    else:
        return xcorr


