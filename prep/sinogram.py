#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contain functions operating on tomography sinograsm
"""

import numpy                as     np
from   scipy.signal         import medfilt2d
from   tomoproc.util.logger import logger_default
from   tomoproc.util.logger import log_exception


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


def normalize_background(
        sino: np.ndarray, 
        method: str='interpolate',
    ) -> np.ndarray:
    """
    Description
    -----------

    Parameters
    ----------

    Returns
    -------
    np.ndarray
        sinogram with non-sample region (background) normalized to one
            one -> zero attenuation
    """
    pass


if __name__ == "__main__":
    testimg = np.random.random((500,500))
    img = denoise(testimg, method='svd')
