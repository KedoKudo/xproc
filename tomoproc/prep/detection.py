#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Provide functions that operate on projections
"""

import numpy                as     np

from   typing               import Tuple
from   scipy.signal         import medfilt
from   scipy.signal         import medfilt2d
from   scipy.ndimage        import gaussian_filter
from   tomopy               import minus_log
from   tomopy               import find_center_pc
from   tomoproc.util.npmath import rescale_image
from   tomoproc.util.logger import logger_default
from   tomoproc.util.logger import log_exception


@log_exception(logger_default)s
def detect_sample_in_sinogram(
    sino: np.ndarray,
    kernel_size: int=3,
    sigma: int=50,
    minimum_distance_to_edge: int=5,
    ) -> Tuple[int, int]:
    """
    Description
    -----------
        Automatically detect the left and right edge of the sample region
        in a sinogram with median and gaussian filtering.

    Parameters
    ----------
    sino: np.ndarray
        Sinogram for evaluation
    kernel_size: int
        median filter (quick denoising) kernel size
    sigma: int
        gaussian filter kernel size
    minimum_distance_to_edge: int
        minimum amount of pixels to sinogram edge

    Returns
    -------
    (int, int)
        left and right edge of the sample region
    """
    # use median filter and gaussian filter to locate the sample region 
    # -- median filter is to counter impulse noise
    # -- gaussian filter is for estimating the sample location
    prf = np.gradient(
        np.sum(
            gaussian_filter(
                medfilt2d(sino, kernel_size=kernel_size), 
                sigma=sigma,
                ),
            axis=0,
            )
        )
    return  (
        max(prf.argmin(), minimum_distance_to_edge), 
        min(prf.argmax(), sino.shape[1]-minimum_distance_to_edge),
        )


@log_exception(logger_default)
def detect_corrupted_proj(
    projs: np.ndarray,
    omegas: np.ndarray,
    threshold: float=0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Corrupted frames/projections will add a forgy layer (artifact) of random 
    noise to the final reconstruction.  These corrupted frames can be detected
    through 180 degree pair-wise checking.

    Parameters
    ----------
    projs: np.ndarray
        tomography image stack [axis_omega, axis_imgrow, axis_imgcol]
    omegas: np.ndarray
        angular position vector
    threshold: float
        Threshold for picking out the outliers

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
           idx_BAD     idx_GOOD
        Return the indices of BAD frames and GOOD frames/projections
    """
    # assume equal step, find the index range equals to 180 degree
    dn = int(np.pi/(omegas[1] - omegas[0]))

    # get the cnts from each 180 pairs
    cnts = [
        tomopy.find_center_pc(
            rescale_image(minus_log(projs[n_img,:,:])), 
            rescale_image(minus_log(projs[n_img+dn,:,:])), 
            rotc_guess=projs.shape[2]/2,
            )   for nimg in range(dn)
    ]

    # 180 -> 360
    cnts = np.array(cnts + cnts)

    # locate outlier
    diff = np.absolute(cnts - medfilt(cnts))/cnts

    return np.where(diff>threshold)[0], np.where(diff<=threshold)[0]


def detect_slit_corner():
    pass


if __name__ == "__main__":
    projs = np.random.random((360,60,60))
    thetas = np.linspace(0, np.pi*2, 360)
    detect_bad_proj(projs, thetas)
