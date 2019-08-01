import os
import numpy as np
from scipy.ndimage import gaussian_filter


def cedf(img, t_dim=0, img_sigma=1, gradient_sigma=5):
    '''
    Perform coherence enhanced diffusion filtering to extract scalar
    nematic order parameter. Begin with image and perform Gaussian blur
    with standard deviation img_sigma. Find gradient at each pixel,
    del(u(x, y)). Then, create structure tensor by taking outer product
    of gradient at each pixel. Smooth each component of the structure tensor
    in space with Gaussian with standard deviation gradient_sigma. The nematic
    director is the eigenvector associated with smallest eigenvalue of the structure
    tensor at that pixel. Construct the nematic tensor, Q, using

    Parameters
    ----------
    img : 3D array
        array to perform cedf on, size NxMxP
    t_dim : int, optional
        dimension in img that corresponds to time. Defaults to 0
    img_sigma : float, optional
        standard deviation of Gaussian to blur image with,
        in units of pixels. Defaults to 1
    gradient_sigma : float, optional
        standard deviation of Gaussian with which to blur
        each component of structure tensor. Should be larger
        than img_sigma. Defaults to 5.

    Results
    -------
    s : 3D array
        scalar nematic order parameter, size NxMxP

    See also
    --------

    Example
    -------

    '''
    if gradient_sigma <= img_sigma:
        raise ValueError('gradient_sigma should be greater than img_sigma')

    if img.ndim != 3:
        raise ValueError('input data should be 2+1 dimensional')

    # rearrange img to have t_dim in front
    img = np.moveaxis(cedf, t_dim, 0)
    s = np.zeros(img.shape)

    for ind, im in enumerate(img):
        s[ind] = cedf_2D(im, img_sigma, gradient_sigma)

    s = np.moveaxis(s, 0, t_dim)

    return s


def cedf_2D(img, img_sigma=1, gradient_sigma=5):
    '''
    Perform CEDF on 2D data

    Parameters
    ----------
    img : 2D array
        2D array of data to be analyzed, size NxM
    img_sigma : float, optional
        standard deviation of Gaussian to blur image with,
        in units of pixels. Defaults to 1
    gradient_sigma : float, optional
        standard deviation of Gaussian with which to blur
        each component of structure tensor. Should be larger
        than img_sigma. Defaults to 5.

    Results
    -------
    s : 2D array
        array of scalar order parameter, size NxM

    See also
    --------

    Example
    -------

    '''
    # smooth the image
    im = gaussian_filter(img, sigma=img_sigma)

    # construct smoothed structure tensor
    j = structure_tensor(im, sigma=gradient_sigma)

    # get molecular director at each pixel
    u = molecular_director(j)

    # construct nematic tensor
    Q = nematic_tensor(u)

    # Get scalar nematic order parameter and nematic director field
    s, n = scalar_director(Q)

    return s, n


def structure_tensor(im, sigma=5):
    '''
    construct structure tensor from image data

    Parameters
    ----------
    im : 2D array
        2D array of data to be analyzed, size NxM
    sigma : float, optional
        standard deviation of Gaussian with which to blur
        each component of structure tensor. Defaults to 5.

    Results
    -------
    j : 4D array
        structure tensor at each data point in im. Size NxMx2x2

    See also
    --------

    Example
    -------

    '''

    N, M = im.shape
    j = np.zeros((N, M, 2, 2))

    # take image gradient. Returns a 2xNxM array, where im_grad[0]
    # refers to gradient along rows (i.e. y direction). Flip
    # to put into cartesian coordinates
    im_grad = np.flip(np.asarray(np.gradient(im)), axis=0)

    for ind, g in enumerate(np.reshape(np.moveaxis(im_grad, 0, 2), (N * M, 2))):
        rowInd, colInd = np.unravel_index(ind, (N, M))
        j[rowInd, colInd, ...] = np.outer(g, g)

    idx_pairs = ((0, 0), (0, 1), (1, 0), (1, 1))

    for ind in idx_pairs:
        j[..., ind[0], ind[1]] = gaussian_filter(j[..., ind[0], ind[1]], sigma=sigma)

    return j


def molecular_director(j):
    '''
    calculate molecular director from structure tensor j.
    Each pixel of image has a 2x2 tensor assocaited with it.
    The molecular director is the eigenvector associated with
    the smallest eigenvalue of each 2x2 tensor

    Parameters
    ----------
    j: array_like
        array of structure tensor, size NxMx2x2

    Results
    -------
    u : array_like
        array of molecular director at each pixel of image,
        size NxMx2

    See also
    --------

    Example
    -------

    '''
    N, M, _, _ = j.shape
    u = np.zeros((N, M, 2))
    # get eigenvalues and eigenvectors of j
    evals, evecs = np.linalg.eig(j)

    # get evecs associated with smallest eval at each pixel
    # get all indices where the first eval is smaller
    u[(evals[..., 0] / evals[..., 1]) < 1] = evecs[(evals[..., 0] / evals[..., 1]) < 1, :, 0]

    # get all indices where the second eval is smaller
    u[(evals[..., 1] / evals[..., 0]) < 1] = evecs[(evals[..., 1] / evals[..., 0]) < 1, :, 1]

    return u


def nematic_tensor(u, radius=5):
    '''
    Calculate nematic tensor at each pixel given the molecular director field
    With a molecular field u(x,y), each component of the nematic tensor

    Parameters
    ----------
    u : array-like
        array of molecular director field, size NxMx2
    radius : float
        take an average over all directors within this radius of each pixel

    Results
    -------
    Q : array-like
        array of nematic tensor at each pixel, size NxMx2x2

    See also
    --------

    Example
    -------

    '''
    pass


def scalar_director(Q):
    pass
