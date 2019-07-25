import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
import warnings
from itertools import product
import pdb


def entropy(data, sample_spacing, window='boxcar', nperseg=None,
            noverlap=None, nfft=None, detrend='constant', smooth_corr=True,
            sigma=1, subtract_bias=True, many_traj=True, return_density=False,
            azimuthal_average=False):
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
    return_density : bool, optional
        option to return entropy production rate and its density (i.e. the
        quantity summed over to give the epr). Defaults to False
    azimuthal_average : bool, optional
        option to perform an azimuthal average over epr density before calculating
        epr. **Only works for 3D data (e.g. 2 spatial + 1 temporal dimension)**

    Returns
    -------
    s : float
        entropy production rate divided by system size
    s_density : array (optional)
        entropy production rate density divided by system size. Only returned if
        return_density=True
    freqs : list of arrays (optional)
        frequency bins of s_density. Only returned in return_density=True
    '''

    if not sample_spacing:
        raise ValueError('sample_spacing must be given as a single value for all dimensions\n'
                         'or as a sequence with as many elements as the number of dimensions of the data')

    if not many_traj:
        # code is written to run over number of replicates, so add extra singleton dimension
        # to data if there is only one replicate of the data available
        data = data[np.newaxis, :]

    # number of replicates, number of variables, number of time and space points
    nrep, nvar, *ntspace = data.shape

    # nfft checks
    if nfft is None:
        nfft = [*ntspace]
    else:
        if len(nfft) == 1:
            if type(nfft) is not int:
                raise ValueError('nfft must be integer')
            else:
                nfft = np.repeat(np.asarray(int(nfft)), len(ntspace))
        elif len(nfft) == len(nspace) + 1:
            if not all(type(n) is int for n in nfft):
                raise ValueError('nfft must be a list of integers')
        else:
            raise ValueError('size of fft taken is either an integer for all dimensions '
                             'or equal to the number of dimensions as the data')

    # c = np.zeros((*nfft, nvar, nvar), dtype=complex)
    for ii in range(nrep):
        c_temp, freqs = corr_matrix(data[ii, ...],
                                    sample_spacing,
                                    window,
                                    nperseg,
                                    noverlap,
                                    nfft,
                                    detrend,
                                    azimuthal_average)
        if ii == 0:
            c = c_temp
        else:
            c += c_temp
    c /= nrep

    '''
    Here is where I will check the size of nfft in each dimension and make sure the returned correlation function
    and frequency are odd in order to not mess with the flipping that happens below
    '''
    for ndim, n in enumerate(nfft):
        inds = [slice(None)] * len(nfft)  # get first elements in the appropriate dimension
        singletonInds = [slice(None)] * len(nfft)  # use this to expand selected slice for concatenation
        if n % 2 == 0:
            inds[ndim] = 0
            singletonInds[ndim] = np.newaxis
            c = np.concatenate((c, np.conj(c[tuple(inds)][tuple(singletonInds)])), axis=ndim)
            freqs[ndim] = np.concatenate((freqs[ndim], -freqs[ndim][0][np.newaxis]))

    # find spacing of all frequencies, temporal and spatial
    dk = np.array([np.diff(f)[0] for f in freqs])

    # find total time and length signal, including zero padding and azimuthal averaging
    TL = 2 * np.pi / dk
    # TL[-1] *= 2  # multiply length of signal by two to test azimuthal averaging

    # sigma checks
    if isinstance(sigma, (int, float)) or len(sigma) == 1:
        sigma = np.repeat(sigma, len(dk))
    elif len(sigma) == len(dk):
        sigma = np.asarray(sigma)
    else:
        raise ValueError('sigma is either a single value for all dimensions\n'
                         'or has as many elements as the number of dimensions of the data')

    # smooth c if wanted
    if smooth_corr:
        c = _nd_gauss_smooth(c, sigma)

    # get inverse of each NxN submatrix of c.
    # Broadcasts to find inverse of square matrix in last two dimensions of matrix
    c_inv = np.linalg.inv(c)

    # transpose last two indices
    axes = list(range(c.ndim))
    axes[-2:] = [axes[-1], axes[-2]]
    c_inv_transpose = np.transpose(c_inv, axes=axes)

    # first axis is temporal frequency, flip along that axis to get C^-T(k, -w)
    # Also sum over last two axes to sum over matrix indices, leaving only frequency
    # indices for integration
    sdensity = np.sum(np.sum((np.flip(c_inv_transpose, axis=0) - c_inv_transpose) * c, axis=-1),
                      axis=-1) / (2 * TL.prod())

    s = np.sum(sdensity)

    # Calculate and subtract off bias if wanted
    if subtract_bias:
        # print([((freqs[n].max() / sigma[n]) / (TL[n] * dk[n] * (np.pi)**0.5)) for n in range(len(TL))])
        bias = ((1 / nrep) * (nvar * (nvar - 1) / 2) *
                np.prod([((freqs[n].max() / sigma[n]) / (TL[n] * dk[n] * (np.pi)**0.5)) for n in range(len(TL))]))
        s -= bias

    if return_density:
        return s.real, sdensity.real, freqs
    else:
        return s.real


def corr_matrix(data, sample_spacing=None, window='boxcar', nperseg=None,
                noverlap=None, nfft=None, detrend='constant',
                azimuthal_average=False):
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
    azimuthal_average : bool, optional
        Compute azimuthal average of correlation functions? Defaults to False.
        **Only works for 2+1 dimensional data**

    Returns
    -------
    c : ND array
        an ND  matrix that gives the NxN correlation matrix for the variables
        contained in data. Returns fft(c) is return_fft=True
    tau : array
        2M-1 length array of lag times for correlations. Returns frequencies if
        return_fft=True

    '''

    data = np.asarray(data)
    # get total number of variables and number of time points and space points
    nvar, *ntspace = data.shape

    ''' Not sure how to implement Welch's method robustly here

    nperseg checks
    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')
    else:
        nperseg = npts

    noverlap checks
    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')

    '''

    # nfft checks
    if nfft is None:
        nfft = [*ntspace]
    else:
        if len(nfft) == 1:
            if type(nfft) is not int:
                raise ValueError('nfft must be integer')
            else:
                nfft = np.repeat(np.asarray(int(nfft)), len(ntspace))
        elif len(nfft) == len(ntspace):
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
    if azimuthal_average:
        c = np.zeros((*nfft[:-2], nfft[-1] // 2, nvar, nvar), dtype=complex)
    else:
        c = np.zeros((*nfft, nvar, nvar), dtype=complex)

    # get all pairs of indices
    idx_pairs = list(product(np.arange(nvar), repeat=2))

    for idx in idx_pairs:
        c[..., idx[0], idx[1]], freqs = csdn(data[idx[0]], data[idx[1]],
                                             sample_spacing=sample_spacing,
                                             window=window,
                                             detrend=detrend,
                                             nfft=nfft,
                                             azimuthal_average=azimuthal_average)

    return c, freqs


def csdn(data1, data2, sample_spacing=None, window=None,
         detrend='constant', nfft=None, azimuthal_average=False):
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
    azimuthal_average : bool, optional
        Compute azimuthal average of correlation functions? Defaults to False.
        **Only works for 2+1 dimensional data**

    Returns
    -------
    csd : array_like
        Cross spectral density data as nd numpy array
    freqs: list of arrays
        frequency vectors for each dimension of csd

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
        def detrend_func(d):
            return d
    elif detrend == 'constant':
        def detrend_func(d):
            return d - np.mean(d)
    elif detrend == 'linear':
        raise NotImplementedError('Multidimensional linear detrend '
                                  'not implemented')
    # elif not hasattr(detrend, '__call__'):
    #     def detrend_func(d, ax):
    #         return signal.signaltools.detrend(d, type=detrend, axis=ax)
    # else:
    #     detrend_func = detrend

    # detrend the data
    data1 = detrend_func(data1)

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
            data2 = detrend_func(data2)
        data2, _ = _nd_window(data2, window)
        data2_fft = np.fft.fftn(data2, s=nfft)
        csd = data1_fft * np.conjugate(data2_fft)
    else:
        csd = data1_fft * np.conjugate(data1_fft)

    csd *= scale

    csd = np.fft.fftshift(csd)

    # get fourier frequencies
    freqs = []
    for dim, Delta in enumerate(sample_spacing):
        freqs.append(2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nfft[dim], Delta)))

    if azimuthal_average:
        if csd.ndim != 3:
            raise ValueError('Input must be 2+1 dimensional to do azimuthal averaging')
        else:
            dk = np.diff(freqs[-1])[0]
            csd, fr = _azimuthal_average_3D(csd, tdim=0,
                                            center=None,
                                            binsize=1,
                                            mask='circle',
                                            weight=None,
                                            dx=dk)
            freqs = freqs[:-1]
            freqs[-1] = fr

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


def _azimuthal_average(data, center=None, binsize=1, mask=None, weight=None,
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
        max_rad = np.min([x.max(), y.max()])  # pick smaller dimension for non-square input
        radius = (max_rad + 1) / 2
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
                                  weights=data * mask * weight)[0][:-1] / nBinnedData)

    return radialProfile, binCenters[:-1] * binsize * dx


def _azimuthal_average_3D(data, tdim=0, center=None, binsize=1, mask=None,
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
        radial_profile, r = _azimuthal_average(spatial_data,
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
