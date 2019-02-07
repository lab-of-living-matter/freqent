import numpy as np
import scipy.signal as signal
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


"""
Module to calculate dynamic structure factor for 2D movies

Created on Wed Jul 5 22:48:12 2017

Daniel S. Seara
www.github.com/dsseara
daniel.seara@yale.edu
"""
import numpy as np
import scipy.signal as signal
from astropy.convolution import convolve_fft
import warnings


def azimuthal_average(data, center=None, binsize=1, mask=None, weight=None,
                      dx=1.0):
    """
    Calculates the azimuthal average of a 2D array

    Parameters
    ----------
    data : array_like
        2D numpy array of numerical data
    center : array_like, optional
        1x2 numpy array the center of the image from which to measure the
        radial profile from, in units of array index. Default is center of
        data array
    binsize : scalar, optional
        radial width of each annulus over which to average,
        in units of array index
    mask : {array_like, 'circle', None}, optional
        Mask of data. Either a 2D array same size as data with 0s where you
        want to exclude data, "circle", which only includes data in the
        largest inscribable circle within the data, or None, which uses no mask.
        Defaults to None
    dx : float, optional
        Sampling spacing in data. To be used when returning radial coordinate.
        Defaults to 1.0

    Returns
    -------
    radialProfile : array_like
        Radially averaged 1D array from data array
    r : array_like
        Radial coordinate of radial_profile

    Based on radialProfile found at:
    http://www.astrobetter.com/wiki/tiki-index.php?page=python_radial_profiles

    To do
    -----
    1) Make "circle" option of mask accept non-square inputs
    """
    data = np.asarray(data)
    # Get all the indices in x and y direction
    [y, x] = np.indices(data.shape)

    # Define the center from which to measure the radius
    if not center:
        center = np.array([(x.max() - x.min()) / 2, (y.max() - y.min()) / 2])

    # Get distance from all points to center
    r = np.hypot(x - center[0], y - center[1])

    if mask is None:
        mask = np.ones(data.shape, dtype='bool')
    elif mask is 'circle':
        radius = (x.max() + 1) / 2
        mask = (r < radius)
    else:
        mask = np.asarray(mask)

    if weight is None:
        weight = np.ones(data.shape)

    # Get the bins according to binsize
    nbins = int(np.round((r * mask).max()) / binsize) + 1
    maxbin = nbins * binsize
    binEdges = np.linspace(0, maxbin, nbins + 1)

    binCenters = (binEdges[1] - binEdges[0]) / 2 + binEdges[:-1]

    # Number of data points in each bin. Cut out last point, always 0
    nBinnedData = np.histogram(r, binEdges, weights=mask * weight)[0][:-1]
    # Azimuthal average
    radialProfile = (np.histogram(r, binEdges,
                                  weights=data * mask * weight)[0][:-1] /
                     nBinnedData)

    return radialProfile, binCenters[:-1] * binsize * dx


def azimuthal_average_3D(data, tdim=0, center=None, binsize=1, mask=None,
                         weight=None, dx=1.0):
    """
    Takes 3D data and gets radial component of two dimensions

    Parameters
    ----------
    data : array_like
        3D numpy array, 1 dimension is time, the other two are spatial.
        User specifies which dimension in temporal
    tdim : scalar, optional
        specifies which dimension of data is temporal. Options are 0, 1, 2.
        Defaults to 0
    center : array_like or None, optional
        1x2 numpy array the center of the image from which to measure the
        radial profile from, in units of array index. If None, uses center
        of spatial slice of array. Defaults to None
    binsize : scalar, optional
        radial width of each annulus over which to average,
        in units of array index
    mask : {array_like, 'circle', None}, optional
        Mask of data. Either a 2D array same size as data with 0s where you
        want to exclude data, "circle", which only includes data in the
        largest inscribable circle within the data, or None, which uses no mask.
        Defaults to None
    dx : float, optional
        Sampling spacing in dimensions of data over which the averagin is done.
        To be used when returning radial coordinate. Defaults to 1.0

    Returns
    -------
    tr_profile : array_like
        2D, spatially radially averaged data over time.
        First dimesion is time, second is spatial
    r : array_like
        Radial coordinate of radial_profile

    See also: azimuthal_average

    TO DO
    -----
    1) Allow input of non-square data to average over
    """

    data = np.asarray(data)

    # Put temporal axis first
    data = np.rollaxis(data, tdim)

    for frame, spatial_data in enumerate(data):
        radial_profile, r = azimuthal_average(spatial_data,
                                              center,
                                              binsize,
                                              mask,
                                              weight,
                                              dx)
        if frame == 0:
            tr_profile = radial_profile
        else:
            tr_profile = np.vstack((tr_profile, radial_profile))

    return tr_profile, r


def csdn(data1, data2, sample_spacing=None, window=None,
         detrend='constant', nfft=None):
    """
    Estimate cross spectral density of n-dimensional data.

    Parameters
    ----------
    data1, data2 : array_like
        N-dimensional input arrays, can be complex, must have same size
    sample_spacing : array_like, optional
        Sampling frequency in each dimension of data. Defaults to 1 for all
        dimensions
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

    if data1.shape != data2.shape:
        raise ValueError('Both inputs must have same size.')

    if window is None:
        window = 'boxcar'

    if sample_spacing is None:
        fs = np.ones(data1.ndim)
    else:
        sample_spacing = np.asarray(sample_spacing)
        if sample_spacing.size != data1.ndim:
            raise ValueError('sample_spacing.size = {0}, need data1.ndim = {1} values'
                             .format(fs.size, data1.ndim))

    data1, win = _nd_window(data1, window)
    data2, _ = _nd_window(data2, window)
    freqs = []

    # Window squared and summed, generalized to nd
    # See Numerical Recipes, section 13.4.1
    wss = 1
    for arr in win:
        wss *= (arr**2).sum()

    # Set scaling
    scale = sample_spacing / wss
    data1_fft = np.fft.fftn(data1, s=nfft)
    data2_fft = np.fft.fftn(data2, s=nfft)

    csd = data1_fft * np.conjugate(data2_fft)

    csd *= scale

    for dim, Delta in enumerate(sample_spacing):
        freqs.append(np.linspace(-np.pi * Delta, np.pi * Delta, csd.shape[dim]))

    # if return_onesided:
    #     if np.iscomplexobj(data):
    #         return_onesided = False
    #         warnings.warn('Input data is complex, switching to '
    #                       'return_onesided=False')
    #     else:
    #         psd = _one_side(psd)
    #         for dim, array in enumerate(freqs):
    #             freqs[dim] = _one_side(array)

    return csd, freqs


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
    nperseg : array_like
        Length of each segment in each dimension. If `None`, uses whole
        dimension length. Defaults to `None`.

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
    `sqw.psdn()`
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
        c[:, idx[0], idx[1]] = _direct_csd(data[idx[0]], data[idx[1]], sample_spacing,
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


def _direct_csd(x, y, sample_spacing=1.0, window='boxcar', nperseg=None,
                noverlap=None, nfft=None, detrend='constant', padded=False,
                return_fft=True):
    '''
    Estimate the direct cross power spectral density using Welch's method.
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
    csd = np.mean(csd, axis=0)
    if not return_fft:
        # return the cross-covariance sequence
        ccvs = np.fft.fftshift(np.fft.ifft(csd)) / sample_spacing
        return ccvs
    else:
        return np.fft.fftshift(csd)


def _gauss_smooth(corr, stddev=10, axis=0):
    '''
    Helper function that smooths a correlation matrix along its time axis with a Gaussian.
    To be used on the correlation functions out of corr_matrix.

    Parameters
    ----------
    corr: 2D array
        MxNxN correlation matrix array. M is the number of time points,
        N is the number of variables
    stddev: int
        standard deviation of gaussian used to smooth data, in units of the sampling
        spacing between points in data
    axis: int, optional
        Axis along which to smooth the data

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
    gauss = Gaussian1DKernel(stddev)

    for idx in idx_pairs:
        smooth_corr[:, idx[0], idx[1]].real = convolve(corr[:, idx[0], idx[1]].real,
                                                       gauss, normalize_kernel=True)
        smooth_corr[:, idx[0], idx[1]].imag = convolve(corr[:, idx[0], idx[1]].imag,
                                                       gauss, normalize_kernel=True)

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
