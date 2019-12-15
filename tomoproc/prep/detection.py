#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Provide functions that operate on projections
"""

import tomopy
import numpy                     as     np
import concurrent.futures        as     cf

from   typing                    import Tuple
from   scipy.signal              import medfilt
from   scipy.signal              import medfilt2d
from   scipy.ndimage             import gaussian_filter
from   scipy.ndimage             import gaussian_filter1d
from   skimage                   import exposure
from   tomopy                    import minus_log
from   tomopy                    import find_center_pc
from   tomoproc.util.npmath      import rescale_image
from   tomoproc.util.peakfitting import fit_sigmoid
from   tomoproc.util.npmath      import rescale_image
from   tomoproc.util.npmath      import binded_minus_log


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
    omegas: np.ndarray,
    threshold: float=0.8,
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
    # use the faster version instead
    with cf.ProcessPoolExecutor() as e:
        _jobs = [
            e.submit(
                tomopy.find_center_pc,
                rescale_image(binded_minus_log(projs[nimg,:,:])), 
                rescale_image(binded_minus_log(projs[nimg+dn,:,:])), 
            )
            for nimg in range(dn)
        ]

    cnts = [me.result() for me in _jobs]
    # cnts = [
    #     tomopy.find_center_pc(
    #         rescale_image(minus_log(projs[nimg,:,:])), 
    #         rescale_image(minus_log(projs[nimg+dn,:,:])), 
    #         rotc_guess=projs.shape[2]/2,
    #         )   for nimg in range(dn)
    # ]

    # 180 -> 360
    cnts = np.array(cnts + cnts)

    # locate outlier
    diff = np.absolute(cnts - medfilt(cnts))/cnts

    return np.where(diff>threshold)[0], np.where(diff<=threshold)[0]


def guess_slit_box(img: np.ndarray, boost: bool=True) -> dict:
    """
    Description
    -----------
    Auto detect/guess the four blades position (in pixels) for given image

    Parameters
    ----------
    img: np.ndarray
        2D tomography image with slit box

    Returns
    -------
    dict:
        dictionary contains the approximated position (in pixel) for the four
        slit blades
    
    NOTE
    ----
    For images without any slit blades, a random (probably useless) region
    will be returned.

    Relative fast:
    tested on MacBookPro13,3
    395 ms ± 14 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """    
    if boost:
        # Contrast stretching
        pl, ph = np.percentile(img, (2, 98))
        img = exposure.rescale_intensity(img, in_range=(pl, ph))
        
        # equilize hist
        img = exposure.equalize_adapthist(img)
    
    # map to log to reveal transition box
    img = np.log(medfilt2d(img.astype(float))+1)
    
    # get row and col profile gradient
    pdot_col = np.gradient(gaussian_filter1d(np.average(img, axis=0), sigma=11))
    pdot_row = np.gradient(gaussian_filter1d(np.average(img, axis=1), sigma=11))

    return {
        'left':  np.argmax(pdot_col),
        'right': np.argmin(pdot_col),
        'top':   np.argmax(pdot_row),
        'bot':   np.argmin(pdot_row),
    }


def detect_slit_corners(img: np.ndarray, r: float=50) -> list:
    """
    Description
    -----------
    Detect four corners (sub-pixel) formed by the four balde slits commonly 
    used at 1ID@APS.

    Parameters
    ----------
    img: np.ndarray
        input images, slit baldes must be visible within the image
    r: float
        domain size, will be automatically adjusted to avoid out-of-bound
        issue
    
    Returns
    -------
    list[upper_left, lower_left, lower_right, upper_right]
        List of the sub-pixel positions of the four corners in the
        counter-clock-wise order

    NOTE
    ----
    The location of the corner is affected by the size of the domain (r). A
    consistent value is recommended for quantitative analysis, such as detector
    drift correction.
    """
    # guess the rough location first
    # by default use boost contrast, if failed, use raw image
    try:
        edges = guess_slit_box(img, boost=True)
        le,re,te,be = edges['left'], edges['right'], edges['top'], edges['bot']
        
        r_row, r_col = min(r, be-te-1), min(r, re-le-1)
        
        safe_domain = lambda row, col: [(max(row - r_row, 0), min(row + r_row + 1, img.shape[0])), 
                                        (max(col - r_col, 0), min(col + r_col + 1, img.shape[1])),
                                    ]
        
        cnrs = [(te, le), (be, le), (be, re), (te, re)]  # (row, col)
        
        for i, cnr in enumerate(cnrs):
            
            rowrange, colrange = safe_domain(*cnr)
            domain = img[rowrange[0]:rowrange[1], colrange[0]:colrange[1]]
            
            horizontal_lp = np.average(domain, axis=0)
            vertical_lp   = np.average(domain, axis=1)
            
            popt, _ = fit_sigmoid(np.arange(len(vertical_lp)), vertical_lp)
            _row = popt[0]
            popt, _ = fit_sigmoid(np.arange(len(horizontal_lp)), horizontal_lp)
            _col = popt[0]
            
            cnrs[i] = (rowrange[0]+_row, colrange[0]+_col)
    except:
        print("boost contrast leads to error, use raw image instead")
        edges = guess_slit_box(img, boost=False)
        le,re,te,be = edges['left'], edges['right'], edges['top'], edges['bot']
        
        r_row, r_col = min(r, be-te-1), min(r, re-le-1)
        
        safe_domain = lambda row, col: [(max(row - r_row, 0), min(row + r_row + 1, img.shape[0])), 
                                        (max(col - r_col, 0), min(col + r_col + 1, img.shape[1])),
                                    ]
        
        cnrs = [(te, le), (be, le), (be, re), (te, re)]  # (row, col)
        
        for i, cnr in enumerate(cnrs):
            
            rowrange, colrange = safe_domain(*cnr)
            domain = img[rowrange[0]:rowrange[1], colrange[0]:colrange[1]]
            
            horizontal_lp = np.average(domain, axis=0)
            vertical_lp   = np.average(domain, axis=1)
            
            popt, _ = fit_sigmoid(np.arange(len(vertical_lp)), vertical_lp)
            _row = popt[0]
            popt, _ = fit_sigmoid(np.arange(len(horizontal_lp)), horizontal_lp)
            _col = popt[0]
            
            cnrs[i] = (rowrange[0]+_row, colrange[0]+_col)
    
    return cnrs


def detect_rotation_center(
    projs: np.ndarray, 
    omegas: np.ndarray,
    index_good: np.ndarray,
    ) -> float:
    """
    Description
    -----------
    Use the phase-contrast method provided in tomopy to detect the rotation
    center of given tomo image stack.

    Parameters
    ----------
    projs: np.ndarray
        Tomo imagestack with [axis_omega, axis_row, axis_col]
    omegas: np.ndarray
        rotary position array
    index_good: np.ndarray
        indices of uncorrupted frames

    Returns
    -------
    Rotation center horizontal position
    """
    # assume equal step, find the index range equals to 180 degree
    dn = np.rint(np.pi/(omegas[1] - omegas[0])).astype(int)

    with cf.ProcessPoolExecutor() as e:
        _jobs = [
            e.submit(
                tomopy.find_center_pc,
                rescale_image(binded_minus_log(projs[nimg,:,:])), 
                rescale_image(binded_minus_log(projs[nimg+dn,:,:])), 
            )
            for nimg in range(dn)
        ]
    rot_cnts = [me.result() for me in _jobs]
    rot_cnts = np.array(rot_cnts + rot_cnts)

    return np.average(rot_cnts[index_good])


if __name__ == "__main__":
    projs = np.random.random((360,60,60))
    thetas = np.linspace(0, np.pi*2, 360)
    detect_corrupted_proj(projs, thetas)
