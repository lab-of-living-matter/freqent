import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import warnings
from itertools import product
from astropy.convolution import Gaussian1DKernel, convolve


def entropy(data, sample_spacing=1, window='boxcar', nperseg=None,
            noverlap=None, nfft=None, detrend='constant', padded=False,
            smooth_corr=True, sigma=1, subtract_bias=True):
    '''
    Calculate the entropy using the frequency space measure:

    dS/dt = (sum_n ((C^-1)^T_ij (f_n) - C^-1_ij(f_n)) C_ij(f_n)) / 2T

    where T is the total time of the signal, C_ij(w_n) is the (i,j)th component
    of the correlation matrix evaluated at the frequency f_n, where f_n = n/T,
    and n is in [-N/2, N/2], where N is the total number of points in the original
    signal

    Parameters
    ----------
    data : 2D or 3D array
        If 2D, an NxM array that gives length M time series data of N variables.
        e.g. data[n] returns time series for nth variable.
        If 3D, an JxNxM array that gives length M time series of N variables for
        J different replicates. Each replicate's correlation matrix will be
        calculated and averaged together before calculating entropy
    sample_spacing : float, optional
        Sampling interval of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `scipy.signal.get_window` to generate the window values,
        which are DFT-even by default. See `get_window` for a list of windows
        and required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Boxcar window.
    nperseg : int, optional
        Length of each segment. Defaults to None, which takes nperseg=len(x)
        but if window is str or tuple, is set to 256, and if window is
        array_like, is set to the length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `False`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`.
    smooth_corr : bool, optional
        option to smooth the correlation function or not
    sigma : int, optional
        if smooth_corr, standard deviation of gaussian kernel used to
        smooth corelation matrix
    subtract_bias : bool, optional
        option to subtract systematic bias from entropy estimate or not.
        Bias given by N(N-1) / (2 sqrt(pi)) * omega_max / (J * T_max * sigma)

    Returns
    -------
    s : float
        entropy production rate given correlation functions
    '''

    if data.ndim == 3:
        # print('Assuming data dimensions are nReplicates, nVariables, nTimePoints.\n',
        #       'If not, you are about to get nonsense.')
        nRep, nVar, nTime = data.shape  # number of replicates, number of variables, number of time points
        c_fft_all = np.zeros((nRep, nTime, nVar, nVar), dtype=complex)
        for ii in range(nRep):
            c_fft_all[ii, ...], omega = corr_matrix(data[ii, ...],
                                                    sample_spacing,
                                                    window,
                                                    nperseg,
                                                    noverlap,
                                                    nfft,
                                                    detrend,
                                                    padded,
                                                    return_fft=True)
        c_fft = c_fft_all.mean(axis=0)

    elif data.ndim == 2:
        nRep = 1
        nVar, nTime = data.shape
        c_fft, omega = corr_matrix(data,
                                   sample_spacing,
                                   window,
                                   nperseg,
                                   noverlap,
                                   nfft,
                                   detrend,
                                   padded,
                                   return_fft=True)
    elif data.ndim not in [2, 3]:
        raise ValueError('Number of dimensions of data needs to be 2 or 3. \n',
                         'Currently is {0}'.format(data.ndim))

    T = sample_spacing * nTime  # find total time of simulation
    dw = 2 * np.pi / T  # find spacing of fourier frequencies

    # smooth c_fft if wanted
    if smooth_corr:
        c_fft = _gauss_smooth(c_fft, sigma)

    # get inverse of each NxN submatrix of c_fft. Broadcasts to find inverse of square
    # matrix in last two dimensions of matrix
    c_fft_inv = np.linalg.inv(c_fft)
    s = np.sum((c_fft_inv - np.transpose(c_fft_inv, (0, 2, 1))) * c_fft)

    s /= 2 * T

    # Calculate and subtract off bias if wanted
    if subtract_bias:
        bias = (np.pi**-0.5) * (nVar * (nVar - 1) / 2) * (omega.max() / (nRep * T * sigma * dw))
        s -= bias

    return s


def corr_matrix(data, sample_spacing=None, window='boxcar', nperseg=None,
                noverlap=None, nfft=None, detrend='constant',
                return_fft=True):
    '''
    Takes time series data of multiple fields and returns a correlation matrix
    for every lag interval.

    Each data set is k+1 dimensional, where k is the number of space dimensions.

    Parameters
    ----------
    data : array-like
        data is an array that gives spacetime data of N variables.
        e.g. data[n] returns data of nth variable. data[n] is k+1 dimensional.
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
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
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
    nvars = data.shape[0]  # total number of variables
    nspace = data.shape[1:-1]  # sequence of size of each spatial dimension
    space_dim = len(nspace)
    nt = data.shape[-1]  # number of time points

    if nvars > nt:
        warnings.warn('Number of variables > number of time points. '
                      'Make sure data has variables as rows.')

    ####################################################################
    # Not sure how to implement Welch's method robustly here
    # nperseg checks
    # if nperseg is not None:  # if specified by user
    #     nperseg = int(nperseg)
    #     if nperseg < 1:
    #         raise ValueError('nperseg must be a positive integer')
    # else:
    #     nperseg = npts

    # nfft checks
    if nfft is None:
        nfft = (*nspace, nt)
    else:
        if len(nfft) == 1:
            nfft = np.repeat(np.asarray(int(nfft)), space_dim + 1)
        elif len(nfft) == space_dim + 1:
            nfft = np.asarray(nfft).astype(int)
        else:
            raise ValueError('size of fft taken is either an integer for all dimensions or equal to the number of dimensions as the data')

    # noverlap checks
    # if noverlap is None:
    #     noverlap = nperseg // 2
    # else:
    #     noverlap = int(noverlap)
    # if noverlap >= nperseg:
    #     raise ValueError('noverlap must be less than nperseg.')
    ####################################################################

    # preallocate correlation matrix
    c = np.zeros((*nfft, nvars, nvars), dtype=complex)

    # get all pairs of indices
    idx_pairs = list(product(np.arange(nvars), repeat=2))

    for idx in idx_pairs:
        c[..., idx[0], idx[1]] = csdn(data[idx[0]], data[idx[1]], sample_spacing,
                                      window, nfft, detrend,
                                      padded, return_fft)

    if not return_fft:
        c = c.real
        maxTau = sample_spacing * (npts - 1)
        tau = np.linspace(-maxTau, maxTau, npts)
        return c, tau
    else:
        freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nfft, d=sample_spacing))
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
    psd : array_like
        Power spectrum of data as nd numpy array
    freqs : list
        list whose nth element is a numpy array of fourier coordinates of nth
        dimension of data

    See also
    --------
    scipy.signal.get_window
    """

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    same_data = data2 is data1

    nspace = data1.shape[:-1]
    space_dim = len(nspace)

    if not same_data:
        if data1.shape != data2.shape:
            raise ValueError('Both inputs must have same size.')

    if window is None:
        window = 'boxcar'

    if sample_spacing is None:
        sample_spacing = np.ones(data1.ndim)
    else:
        if len(sample_spacing) == 1:
            sample_spacing = np.repeat(np.asarray(sample_spacing), space_dim + 1)
        elif len(sample_spacing) == space_dim + 1:
            sample_spacing = np.asarray(sample_spacing)
        else:
            raise ValueError('sample_spacing is either an integer for all dimensions or equal to the number of dimensions as the data')

    # Handle detrending and window functions
    if not detrend:
        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):
        def detrend_func(d, ax):
            return signal.signaltools.detrend(d, type=detrend, axis=ax)
    else:
        detrend_func = detrend

    # detrend the data in all dimensions
    for ax in range(space_dim + 1):
        data1 = detrend_func(data1, ax)

    data1, win = _nd_window(data1, window)

    freqs = []

    # Window squared and summed, generalized to n-dimensions
    # See Numerical Recipes, section 13.4.1
    wss = 1
    for arr in win:
        wss *= (arr**2).sum()

    # Set scaling
    scale = sample_spacing / wss
    data1_fft = np.fft.fftn(data1, s=nfft)

    if not same_data:
        for ax in range(space_dim + 1):
            data2 = detrend_func(data2, ax)
        data2, _ = _nd_window(data2, window)
        data2_fft = np.fft.fftn(data2, s=nfft)
        csd = data1_fft * np.conjugate(data2_fft)
    else:
        csd = data1_fft * np.conjugate(data1_fft)

    csd *= scale

    for dim, Delta in enumerate(sample_spacing):
        freqs.append(np.linspace(-np.pi / Delta, np.pi / Delta, csd.shape[dim]))

    return np.fft.fftshift(csd), freqs


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


def _nd_gauss_smooth(corr, stddev=10, mode='reflect'):
    '''
    Helper function that smooths a correlation matrix along its time axis with a Gaussian.
    To be used on the correlation functions out of corr_matrix.

    Parameters
    ----------
    corr: array-like
        correlation matrix array, output of corr_matrix
    stddev: int
        standard deviation of gaussian used to smooth data, in units of the sampling
        spacing between points in data
    mode : str or sequence, optional
        The `mode` parameter determines how the input array is extended
        when the filter overlaps a border. See scipy.ndimage.gaussian_filter documentation

    Results
    -------
    smooth_corr : 2D array
        NxMxM array that contains data smoothed with window along axis

    See also
    --------
    freqent.corr_matrix()
    astropy.convolution.convolve()
    '''

    nvars = corr.shape[-1]
    smooth_corr = np.zeros(corr.shape, dtype=complex)
    idx_pairs = list(product(np.arange(nvars), repeat=2))

    for idx in idx_pairs:
        smooth_corr[..., idx[0], idx[1]].real = ndimage.gaussian_filter(corr[..., idx[0], idx[1]].real,
                                                                        sigma=stddev,
                                                                        mode=mode)
        smooth_corr[..., idx[0], idx[1]].imag = ndimage.gaussian_filter(corr[..., idx[0], idx[1]].imag,
                                                                        sigma=stddev,
                                                                        mode=mode)

    return smooth_corr


def _correlate_mean(x, y, sample_spacing=1.0, mode='full', method='auto', norm='biased', return_fft=False):
    '''
    DEPRECATED
    Calculate cross-correlation between two time series. Just a wrapper
    around scipy.signal.correlate function that takes a mean rather than
    just a sum. Helper function for corr_matrix.

    Parameters
    ----------
    x, y : 1D array
        data to find cross-correlation between
    mode : str {'valid', 'same', 'full'}, optional
        Refer to the 'scipy.signal.correlate' docstring. Default is 'full'.
    method : str {'auto', 'direct', 'fft'}, optional
        Refer to the 'scipy.signal.correlate' docstring. Default is 'auto'.
    norm : str {'unbiased', 'biased', 'none'}, optional
        Determine which normalization to use on correlation function. If 'unbiased',
        divide by number of points in sum for each lag time. If 'biased', divide by
        number of elements in time series. If 'none', don't normalize correlation.
        Default is 'biased'
    return_fft : bool (optional)
        boolean asking whether to return the temporal fourier transform of the
        correlation matrix

    Returns
    -------
    xcorr : 1D array
        cross correlation between x and y. Returns fft(xcorr) if return_fft=True

    See also
    --------
    scipy.signal.correlate
    '''

    N = max(len(x), len(y))
    xcorr = signal.correlate(x - x.mean(), y - y.mean(), mode, method)

    if norm in {'biased', 'Biased'}:
        xcorr /= N
    elif norm in {'unbiased', 'Unbiased'}:
        n = N * np.ones(N) - np.arange(N)
        n = np.concatenate((np.flipud(n)[:-1], n))
        xcorr /= n
    elif norm not in {'biased', 'Biased', 'unbiased', 'Unbiased', 'none', 'None', None}:
        raise ValueError('norm = {"biased", "unbiased", or "none"}. Given as {0}'.format(norm))

    if return_fft:
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(xcorr))) * sample_spacing
    else:
        return xcorr
