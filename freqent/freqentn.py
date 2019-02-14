import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
import warnings
from itertools import product
import pdb


def entropy(data, sample_spacing, window='boxcar', nperseg=None,
            noverlap=None, nfft=None, detrend='constant', smooth_corr=True,
            sigma=1, subtract_bias=True, many_traj=True):
    '''
    Calculate the entropy using the frequency space measure:

    dS/dt = (sum_(n,m) (C^-T(k_m, -f_n) - C^-T(k_m, f_n))_ij C_ij(k_m, f_n)) / 2T

    where T is the total time of the signal, C_ij(k_m, f_n) is the (i,j)th component
    of the correlation matrix evaluated at the temporal frequency f_n and spatial
    frequency k_m, where f_n = n/T and k_m = m/L (note that is a vector for every
    spatial dimension).and n is in [-N/2, N/2], where N is the total number of points
    in the original signal

    Parameters
    ----------
    data : array-like
        data is an array that gives spacetime data of N variables.
        e.g. data[n] returns data of nth variable. data[n] is k+1 dimensional.
        First dimension is time, last k dimensions are space
    sample_spacing : float or array-like
        Sampling interval of data. Can either be a sequence with each element
        referring to each dimension of data[n], or a constant if the spacing is
        the same for all dimensions.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `scipy.signal.get_window` to generate the window values,
        which are DFT-even by default. See `get_window` for a list of windows
        and required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Boxcar window.
    ****** NOT IMPLEMENTED ******
    nperseg : int, optional
        Length of each segment. Defaults to None, which takes nperseg=len(x)
        but if window is str or tuple, is set to 256, and if window is
        array_like, is set to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    *****************************
    nfft : int, optional
        Length of the FFT used in each dimensions, if a zero padded FFT
        is desired. If `None`, the FFT length is size of data in each
        dimension. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    smooth_corr : bool, optional
        option to smooth the correlation function or not
    sigma : int, optional
        if smooth_corr, standard deviation of gaussian kernel used to
        smooth corelation matrix
    subtract_bias : bool, optional
        option to subtract systematic bias from entropy estimate or not.
        Bias given by N(N-1) / (2 sqrt(pi)) * omega_max / (J * T_max * sigma)
    many_traj : bool, optional
        option to say whether input data has many trajectories. If so, each trajectory
        should be indexed by the first dimension of data

    Returns
    -------
    s : float
        entropy production rate given correlation functions
    '''

    if not sample_spacing:
        raise ValueError('sample_spacing must be given as a single value for all dimensions\n'
                         'or as a sequence with as many elements as the number of dimensions of the data')

    if many_traj:
        # number of replicates, number of variables, number of time and space points
        nrep, nvar, nt, *nspace = data.shape
        c_all = np.zeros((nrep, nt, *nspace, nvar, nvar), dtype=complex)

        for ii in range(nrep):
            c_all[ii, ...], freqs = corr_matrix(data[ii, ...],
                                                sample_spacing,
                                                window,
                                                nperseg,
                                                noverlap,
                                                nfft,
                                                detrend)
        c = c_fft_all.mean(axis=0)

    else:
        nrep = 1
        nvar, nt, *nspace = data.shape
        c, freqs = corr_matrix(data,
                               sample_spacing,
                               window,
                               nperseg,
                               noverlap,
                               nfft,
                               detrend)

    if type(sample_spacing) is int:
        sample_spacing = np.asarray([sample_spacing] * (len(nspace) + 1))
    elif len(sample_spacing) == len(nspace) + 1:
        sample_spacing = np.asarray(sample_spacing)

    TL = sample_spacing * np.array([nt, *nspace])  # find total time and length of simulation
    dk = 2 * np.pi / TL  # find spacing of all frequencies, temporal and spatial

    # smooth c if wanted
    if smooth_corr:

        if type(sigma) is int:
            sigma = [sigma] * (len(TL))
        elif len(sigma) == len(TL):
            sigma = np.asarray(sigma)
        else:
            raise ValueError('sigma is either a single value for all dimensions\n'
                             'or has as many elements as the number of dimensions of the data')

        c = _nd_gauss_smooth(c, sigma)

    # get inverse of each NxN submatrix of c.
    # Broadcasts to find inverse of square matrix in last two dimensions of matrix
    c_inv = np.linalg.inv(c)

    # transpose last two indices
    axes = list(range(c.ndim))
    axes[-2:] = [axes[-1], axes[-2]]
    c_inv_transpose = np.transpose(c_inv, axes=axes)

    # first axis is temporal frequency.
    # flip along that axis to get C^-T(k, -w)
    s = np.sum((np.flip(c_inv_transpose, axis=0) - c_inv_transpose) * c)

    s /= 2 * TL[0]

    # Calculate and subtract off bias if wanted
    if subtract_bias:
        bias = ((1 / nrep) * (nvar * (nvar - 1) / 2) *
                np.prod([((freqs[n].max() / sigma[n]) / (TL[n] * dk[n])) for n in range(len(TL))]))
        print(bias)
        s -= bias

    return s


def corr_matrix(data, sample_spacing=None, window='boxcar', nperseg=None,
                noverlap=None, nfft=None, detrend='constant'):
    '''
    Takes time series data of multiple fields and returns a correlation matrix
    for every lag interval.

    Each data set is k+1 dimensional, where k is the number of space dimensions.

    Parameters
    ----------
    data : array-like
        data is an array that gives spacetime data of N variables.
        e.g. data[n] returns data of nth variable. data[n] is k+1 dimensional.
        First dimension is time, last k dimensions are space
    sample_spacing : float or array-like, optional
        Sampling interval of data. Can either be a sequence with each element
        referring to each dimension of data[n], or a constant if the spacing is
        the same for all dimensions. Defaults to None, which inputs 1 for all
        dimensions.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `scipy.signal.get_window` to generate the window values,
        which are DFT-even by default. See `get_window` for a list of windows
        and required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Boxcar window.
    ****** NOT IMPLEMENTED ******
    nperseg : int, optional
        Length of each segment. Defaults to None, which takes nperseg=len(x)
        but if window is str or tuple, is set to 256, and if window is
        array_like, is set to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    *****************************
    nfft : int, optional
        Length of the FFT used in each dimensions, if a zero padded FFT
        is desired. If `None`, the FFT length is size of data in each
        dimension. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.

    Returns
    -------
    c : 3D array
        an MxNxN matrix that gives the NxN correlation matrix for the variables
        contained in the rows of data. Returns fft(c) is return_fft=True
    tau : array
        2M-1 length array of lag times for correlations. Returns frequencies if
        return_fft=True

    '''

    data = np.asarray(data)
    # get total number of variables and number of time points and space points
    nvar, *ntspace = data.shape

    ####################################################################
    # Not sure how to implement Welch's method robustly here
    # nperseg checks
    # if nperseg is not None:  # if specified by user
    #     nperseg = int(nperseg)
    #     if nperseg < 1:
    #         raise ValueError('nperseg must be a positive integer')
    # else:
    #     nperseg = npts
    #
    # noverlap checks
    # if noverlap is None:
    #     noverlap = nperseg // 2
    # else:
    #     noverlap = int(noverlap)
    # if noverlap >= nperseg:
    #     raise ValueError('noverlap must be less than nperseg.')
    ####################################################################

    # nfft checks
    if nfft is None:
        nfft = [*ntspace]
    else:
        if len(nfft) == 1:
            if type(nfft) is not int:
                raise ValueError('nfft must be integer')
            else:
                nfft = np.repeat(np.asarray(int(nfft)), len(ntspace))
        elif len(nfft) == space_dim + 1:
            if not all(type(n) is int for n in nfft):
                raise ValueError('nfft must be a list of integers')
            else:
                nfft = np.asarray(nfft).astype(int)
        else:
            raise ValueError('size of fft taken is either an integer for all dimensions or equal to the number of dimensions as the data')

    # sample_spacing checks
    if sample_spacing is None:
        sample_spacing = np.ones(len(ntspace))
    else:
        if len(sample_spacing) == 1:
            sample_spacing = np.repeat(np.asarray(sample_spacing), len(ntspace))
        elif len(sample_spacing) == len(ntspace):
            sample_spacing = np.asarray(sample_spacing)
        else:
            raise ValueError('sample_spacing is either a single value for all dimensions\n'
                             'or has as many elements as the number of dimensions of the data')

    # preallocate correlation matrix
    c = np.zeros((*nfft, nvar, nvar), dtype=complex)

    # get all pairs of indices
    idx_pairs = list(product(np.arange(nvar), repeat=2))

    for idx in idx_pairs:
        c[..., idx[0], idx[1]] = csdn(data[idx[0]], data[idx[1]],
                                      sample_spacing=sample_spacing,
                                      window=window,
                                      detrend=detrend,
                                      nfft=nfft)

    freqs = []

    for dim, Delta in enumerate(sample_spacing):
        freqs.append(2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ntspace[dim], sample_spacing[dim])))

    return c, freqs


def csdn(data1, data2, sample_spacing=None, window=None,
         detrend='constant', nfft=None):
    """
    Estimate cross spectral density of n-dimensional data.

    Parameters
    ----------
    data1, data2 : array_like
        N-dimensional input arrays, can be complex, must have same size
    sample_spacing : float or array-like, optional
        Sampling interval of data. Can either be a sequence with each element
        referring to each dimension of data[n], or a constant if the spacing is
        the same for all dimensions. Defaults to None, which inputs 1 for all
        dimensions.
    window : string, float, or tuple, optional
        Type of window to use, defaults to 'boxcar' with shape of data, i.e.
        data stays the same. See `scipy.signal.get_window` for all windows
        allowed
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    nfft : array_like, optional
        Length of the FFT used in each dimension, if a zero padded FFT is
        desired. If `None`, the FFT is taken over entire array. Defaults to `None`.
    Returns
    -------
    csd : array_like
        Cross spectral density data as nd numpy array

    See also
    --------
    scipy.signal.get_window
    """

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    same_data = data2 is data1

    # nspace = data1.shape[1:]
    # space_dim = len(nspace)

    if not same_data:
        if data1.shape != data2.shape:
            raise ValueError('Both inputs must have same size.')

    if window is None:
        window = 'boxcar'

    # Handle detrending and window functions
    if not detrend:
        def detrend_func(d, ax):
            return d
    elif not hasattr(detrend, '__call__'):
        def detrend_func(d, ax):
            return signal.signaltools.detrend(d, type=detrend, axis=ax)
    else:
        detrend_func = detrend

    # detrend the data in all dimensions
    for ax in range(data1.ndim):
        data1 = detrend_func(data1, ax)

    data1, win = _nd_window(data1, window)

    # Window squared and summed, generalized to n-dimensions
    # See Numerical Recipes, section 13.4.1
    wss = 1
    for arr in win:
        wss *= (arr**2).sum()

    # Set scaling
    scale = np.prod(np.asarray(sample_spacing)) / wss
    data1_fft = np.fft.fftn(data1, s=nfft)

    if not same_data:
        for ax in range(data2.ndim):
            data2 = detrend_func(data2, ax)
        data2, _ = _nd_window(data2, window)
        data2_fft = np.fft.fftn(data2, s=nfft)
        csd = data1_fft * np.conjugate(data2_fft)
    else:
        csd = data1_fft * np.conjugate(data1_fft)

    csd *= scale

    return np.fft.fftshift(csd)


def _nd_window(data, window):
    """
    Windows n-dimensional array. Done to mitigate boundary effects in the FFT.
    This is a helper function for csdn
    Adapted from: https://stackoverflow.com/questions/27345861/
                  extending-1d-function-across-3-dimensions-for-data-windowing

    Parameters
    ----------
    data : array_like
        nd input data to be windowed, modified in place.
    window : string, float, or tuple
        Type of window to create. Same as `scipy.signal.get_window()`

    Results
    -------
    data : array_like
        windowed version of input array, data
    win : list of arrays
        each element returns the window used on the corresponding dimension of
        `data`

    See also
    --------
    `scipy.signal.get_window()`
    `freqent.freqentn.csdn()`
    """
    win = []
    for axis, axis_size in enumerate(data.shape):
        # set up shape for numpy broadcasting
        filter_shape = [1, ] * data.ndim
        filter_shape[axis] = axis_size
        win.append(signal.get_window(window, axis_size).reshape(filter_shape))
        # scale the window intensities to maintain image intensity
        np.power(win[axis], (1.0 / data.ndim), out=win[axis])
        data *= win[axis]

    return data, win


def _nd_gauss_smooth(corr, stddev=1, mode='reflect'):
    '''
    Helper function that smooths a correlation matrix along its time axis with a Gaussian.
    To be used on the correlation functions out of corr_matrix.

    Parameters
    ----------
    corr: array-like
        correlation matrix array, output of freqentn.corr_matrix.
    stddev: scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    mode : str or sequence, optional
        The `mode` parameter determines how the input array is
        extended when the filter overlaps a border.

    Results
    -------
    smooth_corr : array-like
        smoothed correlation array, same size as input.

    See also
    --------
    freqentn.corr_matrix()
    scipy.ndimage.gaussian_filter()
    '''

    nvar = corr.shape[-1]
    smooth_corr = np.zeros(corr.shape, dtype=complex)
    idx_pairs = list(product(np.arange(nvar), repeat=2))

    for idx in idx_pairs:
        smooth_corr[..., idx[0], idx[1]].real = gaussian_filter(corr[..., idx[0], idx[1]].real,
                                                                sigma=stddev,
                                                                mode=mode)
        smooth_corr[..., idx[0], idx[1]].imag = gaussian_filter(corr[..., idx[0], idx[1]].imag,
                                                                sigma=stddev,
                                                                mode=mode)

    return smooth_corr
