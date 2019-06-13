#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provide functions that operate on projections
"""

from   typing               import Tuple
import numpy                as     np
from   scipy.signal         import medfilt2d
from   scipy.ndimage        import gaussian_filter
from   tomopy               import minus_log
from   tomopy               import find_center_pc
from   tomoproc.util.logger import logger_default
from   tomoproc.util.logger import log_exception


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


def detect_corrupted_proj(
    projs: np.ndarray,
    thetas: np.ndarray,
    threshold: float=0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    The corrupted (bad) frames 
    Parameters
    ----------

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Return the indices of bad frames and good frames
    """
    pass


if __name__ == "__main__":
    projs = np.random.random((360,60,60))
    thetas = np.linspace(0, np.pi*2, 360)
    detect_bad_proj(projs, thetas)
