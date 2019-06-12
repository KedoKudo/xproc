#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contain functions operating on tomography sinograsm
"""

import numpy                as     np
from   scipy.signal         import medfilt2d
from   scipy.ndimage        import gaussian_filter
from   tomoproc.util.logger import logger_default
from   tomoproc.util.logger import log_exception


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
        to reduce the impulse noise in the sinogram
    
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
def normalize_background(
        sino: np.ndarray,
        detect_bg: bool=True,
        bg_pixel:  int=11,
        interpolate: bool=True,
    ) -> np.ndarray:
    """
    Description
    -----------
    Auto-detect background (non-sample) region and normalize the background of
    the attentuation map (sinogram).
    This method will not work if the sample occupy the entire filed of view.

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
    Sometimes two or four iterations are necessary if the beam is extremely
    unstable
    """
    sino = np.sqrt(sino)  # for better auto background detection
    
    if detect_bg:
        # use median filter and gaussian filter to locate the sample region 
        # -- median filter is to counter impulse noise
        # -- gaussian filter is for estimating the sample location
        tmp = gaussian_filter(medfilt2d(sino, kernel_size=3), sigma=50)

        # use gradient of the summed (over omega) linear proflie to locate
        # background pixels
        prf = np.gradient(np.sum(tmp, axis=0))

        # find the left and right bound of the sample 
        edgeLeft  = max(prof.argmin(), 11) #
        edgeRigth = min(prof.argmax(), sino.shape[1]-11)
    else:
        edgeLeft  = bg_pixel
        edgeRigth = sino.shape[1]-bg_pixel

    # locate the left and right background
    # NOTE:
    #   Due to hardware issue, the first and last pixels are not always
    #   reliable, therefore throw them out...
    bgL = np.average(sino[:, 1:edgeLeft],   axis=1)
    bgR = np.average(sino[:, edgeRigth:-1], axis=1)

    # calculate the correction matrix alpha
    alpha = np.ones_like(sino)
    if interpolate:
        for n in range(alpha.shape[0]):
            alpha[n,:] = np.linspace(bgL[n], bgR[n], alpha.shape[1])
    else:
        alpha *= ((bgL+bgR)/2)[:,None]
    
    # apply the correction
    return (sino/alpha)**2


if __name__ == "__main__":
    testimg = np.random.random((500,500))
    sino = denoise(testimg, method='svd')
