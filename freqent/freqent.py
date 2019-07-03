import numpy as np
import scipy.signal as signal
import warnings
from itertools import product
from scipy.ndimage import gaussian_filter
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

    if nfft is None:
        nfft = nTime

    T = sample_spacing * nfft  # find total time of simulation
    dw = 2 * np.pi / T  # find spacing of fourier frequencies

    # smooth c_fft if wanted
    if smooth_corr:
        c_fft = _gauss_smooth(c_fft, sigma)

    # get inverse of each NxN submatrix of c_fft. Broadcasts to find inverse of square
    # matrix in last two dimensions of matrix
    c_fft_inv = np.linalg.inv(c_fft)
    sdensity = np.sum(np.sum((c_fft_inv - np.transpose(c_fft_inv, (0, 2, 1))) * c_fft, axis=-1), axis=-1) / (2 * T)

    # return omega, sdensity
    s = np.sum(sdensity)

    # s /= (2 * T)n

    # Calculate and subtract off bias if wanted
    if subtract_bias and smooth_corr:
        bias = (np.pi**-0.5) * (nVar * (nVar - 1) / 2) * (omega.max() / (nRep * T * sigma * dw))
        # print(bias)
        s -= bias

    return s.real


def corr_matrix(data, sample_spacing=1, window='boxcar', nperseg=None,
                noverlap=None, nfft=None, detrend='constant', padded=False,
                return_fft=True):
    '''
    Takes time series data of multiple variables and returns a correlation matrix
    for every lag time

    Parameters
    ----------
    data : 2D array
        Data is an NxM array that gives length M time series data of N variables.
        e.g. data[n] returns time series for nth variable.
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
    nvars, npts = data.shape

    if nvars > npts:
        warnings.warn('Number of rows (variables) > number of columns (time points). '
                      'Make sure data has variables as rows.')

    # nperseg checks
    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')
    else:
        nperseg = npts

    # nfft checks
    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    # noverlap checks
    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')

    # preallocate correlation matrix
    c = np.zeros((nfft, nvars, nvars), dtype=complex)

    # get all pairs of indices
    idx_pairs = list(product(np.arange(nvars), repeat=2))

    for idx in idx_pairs:
        # c[:, idx[0], idx[1]] = _correlate_mean(data[idx[0]], data[idx[1]], sample_spacing,
        #                                        mode, method, norm, return_fft)
        c[:, idx[0], idx[1]] = csd(data[idx[0]], data[idx[1]], sample_spacing,
                                   window, nperseg, noverlap, nfft, detrend,
                                   padded, return_fft)

    if not return_fft:
        c = c.real
        maxTau = sample_spacing * (npts - 1)
        tau = np.linspace(-maxTau, maxTau, npts)
        return c, tau
    else:
        freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nfft, d=sample_spacing))
        return c, freqs


def csd(x, y, sample_spacing=1.0, window='boxcar', nperseg=None,
        noverlap=None, nfft=None, detrend='constant', padded=False,
        return_fft=True):
    '''
    Estimate the cross power spectral density using Welch's method.
    Basically just copying scipy.signal.csd with some default differences.

    Parameters
    ---------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memory as `x` (i.e. ``_spectral_helper(x,
        x, ...)``), the extra computations are spared.
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
    '''

    # make sure we have np.arrays and subtract mean
    x = np.asarray(x)
    y = np.asarray(y)
    same_data = y is x

    # Check if x and y are the same length, zero-pad if necessary
    if not same_data:
        if x.shape[0] != y.shape[0]:
            if x.shape[0] < y.shape[0]:
                pad_shape = list(x.shape)
                pad_shape[0] = y.shape[0] - x.shape[0]
                x = np.concatenate((x, np.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[0] = x.shape[0] - y.shape[0]
                y = np.concatenate((y, np.zeros(pad_shape)), -1)

    nstep = nperseg - noverlap

    # Handle detrending and window functions
    if not detrend:
        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):
        def detrend_func(d):
            return signal.signaltools.detrend(d, type=detrend, axis=-1)
    else:
        detrend_func = detrend

    win = signal.get_window(window, Nx=nperseg)
    scale = sample_spacing / (win * win).sum()

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        nadd = (-(x.shape[-1] - nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = np.concatenate((y, np.zeros(zeros_shape)), axis=-1)

    # break up array into segments, window each segment
    step = nperseg - noverlap
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    x_reshaped = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # detrend each segment
    x_reshaped = detrend_func(x_reshaped)

    x_reshaped = win * x_reshaped
    x_fft = np.fft.fft(x_reshaped, n=nfft)

    if not same_data:
        y_reshaped = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
        y_reshaped = detrend_func(y_reshaped)
        y_reshaped = win * y_reshaped
        y_fft = np.fft.fft(y_reshaped, n=nfft)
        csd = x_fft * np.conjugate(y_fft)
    else:
        csd = x_fft * np.conjugate(x_fft)

    csd *= scale

    csd = np.mean(csd, axis=0)  # take average over segments
    if not return_fft:
        # return the cross-covariance sequence
        ccvs = np.fft.fftshift(np.fft.ifft(csd)) / sample_spacing
        return ccvs
    else:
        return np.fft.fftshift(csd)


def _gauss_smooth(corr, stddev=10, mode='reflect'):
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
    freqent.corr_matrix()
    astropy.convolution.convolve()
    '''

    nvars = corr.shape[-1]
    smooth_corr = np.zeros(corr.shape, dtype=complex)
    idx_pairs = list(product(np.arange(nvars), repeat=2))

    for idx in idx_pairs:
        smooth_corr[..., idx[0], idx[1]].real = gaussian_filter(corr[..., idx[0], idx[1]].real,
                                                                sigma=stddev,
                                                                mode=mode)
        smooth_corr[..., idx[0], idx[1]].imag = gaussian_filter(corr[..., idx[0], idx[1]].imag,
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
