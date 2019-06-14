#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usefule numpy based calculations
"""

import numpy        as     np
import scipy        as     sp

from   typing       import Tuple
from   numpy.linalg import norm


def normalize(vec: np.ndarray, axis=None) -> np.ndarray:
    """
    Description
    -----------
        normalize a vector/matrix

    Parameters
    ----------
    vec: np.ndarray
        input vector/matrix
    axis: int|None
        axis=None : normalize entire vector/matrix
        axis=0    : normalize by column
        axis=1    : normalize by row

    Returns
    -------
    np.ndarray:
        normalized vector/matrix
    """
    vec = np.array(vec, dtype=np.float64)
    if axis is None:
        return vec/norm(vec)
    else:
        return np.divide(vec,
                         np.tile(norm(vec, axis=axis),
                                 vec.shape[axis],
                                 ).reshape(vec.shape,
                                           order='F' if axis == 1 else 'C',
                                           )
                         )


def random_three_vector() -> np.ndarray:
    """
    Description
    -----------
        Generates a random 3D unit vector (direction) with a uniform spherical
        distribution Algo from
        http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    
    Parameters
    ----------

    Returns
    -------
    np.ndarray
        randomly oriented 3D vector
    """
    phi = np.random.uniform(0, np.pi*2)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def safe_dotprod(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Description
    -----------
        Perform dot product that is forced to be between -1.0 and 1.0.  
        Both vectors are normalized to prevent error.
    
    Parameters
    ----------
    vec1: np.ndarray
        input vector one (will be normalized)
    vec2: np.ndarray
        input vector two (will be normalized)
    
    Returns
    -------
    float
        dot product results bounded between -1.0 and 1.0
    
    Examples
    -------
    >>> safe_dotprod([1,1,0], [0,2,0])
    0.7071067811865475
    """
    return min(1.0, max(-1.0, np.dot(normalize(vec1), normalize(vec2))))


def ang_between(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Description
    -----------
        Calculate the angle (radians) bewteen vec1 and vec2
    
    Parameters
    ----------
    vec1: np.ndarray
        input vector one
    vec2: np.ndarray
        input vector two
    
    Returns
    -------
    float:
        angle in radians

    Examples
    --------
    >>> ang_between([1,0,0], [0,1,1])
    1.5707963267948966
    """
    return np.arccos(safe_dotprod(vec1, vec2))


def gauss1d(x: np.ndarray, *p) -> np.ndarray:
    """
    1D Gaussian function used for curve fitting.

    Parameters
    ----------
    x  :  np.ndarray
        1D array for curve fitting
    p  :  parameter lis t
        magnitude, center, std = p

    Returns
    -------
    1d Gaussian distribution evaluted at x with p
    """
    A, mu, sigma = p
    return A * np.exp(-(x - mu)**2 / (2. * sigma**2))


def discrete_cdf(data: np.ndarray, steps=None) -> tuple:
    """
    Calculate CDF of given data without discrete binning to avoid unnecessary
    skew of distribution.

    The default steps (None) will use the whole data. In other words, it is
    close to considering using bin_size=1 or bins=len(data).

    Parameters
    ----------
    data  :  np.ndarray
        1-D numpy array
    steps :  [ None | int ], optional
        Number of elements in the returning array

    Returns
    -------
    pltX  : np.ndarray
        Data along x (data) direction
    pltY  : np.ndarray
        Data along y (density) direction
    """
    x = np.sort(data)

    # check if list is empty
    if len(x) == 0:
        return [], []

    # subsamping if steps is specified and the number is smaller than the
    # total lenght of x
    if (steps is not None) and len(x) > steps:
        x = x[np.arange(0, len(x), int(np.ceil(len(x) / steps)))]

    # calculate the cumulative density
    xx = np.tile(x, (2, 1)).flatten(order='F')
    y = np.arange(len(x))
    yy = np.vstack((y, y + 1)).flatten(order='F') / float(y[-1])

    return xx, yy


def calc_affine_transform(pts_source: np.ndarray, 
                          pts_target: np.ndarray,
        ) -> np.ndarray:
    """
    Description
    -----------
    Use least square regression to calculate the 2D affine transformation
    matrix (3x3, rot&trans) based on given set of (marker) points.
                            pts_source -> pts_target

    Parameters
    ----------
    pts_source  :  np.2darray
        source points with dimension of (n, 2) where n is the number of
        marker points
    pts_target  :  np.2darray
        target points where
                F(pts_source) = pts_target
    Returns
    -------
    np.2darray
        A 3x3 2D affine transformation matrix
          | r_11  r_12  tx |    | x1 x2 ...xn |   | x1' x2' ...xn' |
          | r_21  r_22  ty | *  | y1 y2 ...yn | = | y1' y2' ...yn' |
          |  0     0     1 |    |  1  1 ... 1 |   |  1   1  ... 1  |
        where r_ij represents the rotation and t_k represents the translation
    
    Examples
    --------

    """
    # augment data with padding to include translation
    def pad(x): return np.hstack([x, np.ones((x.shape[0], 1))])

    # NOTE:
    #   scipy affine_transform performs as np.dot(m, vec),
    #   therefore we need to transpose the matrix here
    #   to get the correct rotation

    return scipy.linalg.lstsq(pad(pts_source), pad(pts_target))[0].T


def rescale_image(
    img: np.ndarray,
    dynamic_range: Tuple[float, float] = (0, 1),
    ) -> np.ndarray:
    """
    Description
    -----------
    Rescale the intensity of the given image between minval and maxval

    Parameters
    ----------
    img: np.ndarray
        input 2D image
    dynamic_range: (float, float)
        target dynamic range

    Returns
    -------
    np.ndarray:
        Rescaled 2D image
    """
    # rescale to 0-1
    img -= img.min()
    img /= img.max()

    # rescale to dynamic range
    return dynamic_range[0] + img*(dynamic_range[1] - dynamic_range[0])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
