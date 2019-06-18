#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Contain functions operating on tomography sinograsm

NOTE:
    Different experiment requires different type of correction, the choice of
    which should be established via trial and error.
"""

import numpy                   as     np

from   typing                  import Optional
from   typing                  import Tuple
from   scipy.signal            import medfilt2d
from   scipy.ndimage           import gaussian_filter
from   tomoproc.prep.detection import detect_sample_in_sinogram
from   tomoproc.prep.detection import detect_corrupted_proj
from   tomoproc.util.logger    import logger_default
from   tomoproc.util.logger    import log_exception


@log_exception(logger_default)
def denoise(
        sino: np.ndarray,
        method: str='smd',
        config: dict={
            'kernel_size': 3,
            'threshold': 0.5,
            },
    ) -> np.ndarray:
    """
    Description
    -----------
        Use selected method 
            [median, selective median, SVD] 
        to reduce the impulse noise in the sinogram.  All methods used here are
        lossy, especially the SVD method.
    
    Parameters
    ----------
    sino: np.ndarray
        single sinograms for denoising
    method: str, [median, smd, SVD]
        method for denoising includes
        **median**
            simple 2D median filtering using scipy.signal.medfilt2d
            config = {'kernel_size': 3}
        **selective median filtering**
            only replace the noisy pixels with the corresponding median value
            config = {'kernel_size': 3, 'threshold': 0.05}
        **singular value decomposition**
            retain upto given threshold level of details in the corresponding
            eigen space
            config = {'threshold': 0.95}
    config: dict
        parameters for different methods
    
    Returns
    -------
    np.ndarray
        denoised sinogram
    """
    if method.lower() in ['median', 'simple']:
        return medfilt2d(sino, kernel_size=config['kernel_size'])
    elif method.lower() in ['selective median', 'smd']:
        sino_md = medfilt2d(sino, kernel_size=config['kernel_size'])
        diff = np.absolute(sino_md - sino)/(sino+1)
        return np.where(diff>config['threshold'], sino_md, sino)
    elif method.lower() in ['svd']:
        U, S, V = np.linalg.svd(sino, full_matrices=True)
        eigen_cut = int(min(U.shape[1], V.shape[0])*config['threshold'])
        return (U[:,:eigen_cut]*S[:eigen_cut])@V[:eigen_cut,:]
    else:
        raise NotImplementedError


@log_exception(logger_default)
def beam_intensity_fluctuation_correction(
        sino: np.ndarray,
        detect_bg: bool=True,
        bg_pixel:  int=5,
        interpolate: bool=True,
    ) -> np.ndarray:
    """
    Description
    -----------
    The beam intensity always varies during an experiment, leading to varying
    background (non-sample) region.  This artifacts will leads to strong linear
    artifacts in the final reconstruction, therefore need to be corrected by
    forcing all non-sample region (background) to be one (zero attenuation).
    By default, the sample bound (left and right edge) will be automatically 
    detected with 
        tomoproc.prep.detection.detect_sample_in_sinogram
    which can be bypassed (for speed) if the sample limit is known.

    Parameters
    ----------
    sino: np.ndarray
        sinogram as attenuation map (befor the minus-log step)
    detect_bg: bool
        whether to use automated background pixel detection
    bg_pixel: int
        designated background pixels, superceeded by detect_bg
    interpolate: bool
        whether to interpolate background or not
        ========
        | NOTE |
        ========
        linear interpolation is recommended as the beam are not alawys stable,
        which could lead to intensity shift that cannot be correct through
        background removal.

    Returns
    -------
    np.ndarray
        sinogram with non-sample region (background) normalized to one
            one -> zero attenuation

    NOTE
    ----
    * This method will not work if the sample occupy the entire filed of view.
    * Sometimes two or four iterations are necessary for unstable beam.
    """
    sino = np.sqrt(sino)  # for better auto background detection
    
    # get sample location
    if detect_bg:
        ledge, redge = detect_sample_in_sinogram(sino)
    else:
        ledge, redge = bg_pixel, sino.shape[1]-bg_pixel

    # locate the left and right background
    # NOTE:
    #   Due to hardware issue, the first and last pixels are not always
    #   reliable, therefore throw them out...
    lbg = np.average(sino[:, 1:ledge],   axis=1)
    rbg = np.average(sino[:, redge:-1],  axis=1)

    # calculate the correction matrix alpha
    alpha = np.ones_like(sino)
    if interpolate:
        for n in range(alpha.shape[0]):
            alpha[n,:] = np.linspace(lbg[n], rbg[n], alpha.shape[1])
    else:
        alpha *= ((lbg+rbg)/2)[:,None]
    
    # apply the correction
    return (sino/alpha)**2


@log_exception(logger_default)
def remove_corrupted_projs(
    projs: np.ndarray,
    omegas: np.ndarray,
    idx_good: Optional[np.ndarray]=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Remove corrupted prjections/frames in a given tomo image stack

    Parameters
    ----------
    projs: np.ndarray
        Tomo image stack [axis_omega, axis_imgrow, axis_imgcol]
    idx_good: np.ndarray|None
        index (along omega axis) for good frames 

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Pruned tomo image stack and corresponding angular position (omegas)
    """
    # get the index of good frames if not provided
    idx_good = detect_corrupted_proj(projs, omegas)[1] if idx_good is None else idx_good

    return projs[idx_good,:,:], omegas[idx_good]


@log_exception(logger_default)
def correct_horizontal_jittering(
    tomostack: np.ndarray,
    omegas: np.ndarray,
    remove_bad_frames: bool=True,
    )->np.ndarray:
    """
    Description
    -----------

    Parameters
    ----------

    Returns 
    -------
    
    """
    pass


def correct_detector_drifting():
    pass


if __name__ == "__main__":
    testimg = np.random.random((500,500))
    sino = denoise(testimg, method='svd')