#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contain functions operating on tomography sinograsm
"""

import numpy        as     np
from   tomoproc.util.logger import logger_default
from   tomoproc.util.logger import log_exception


def denoise(
        sino: np.ndarray,
        method: str='smd',
        config: dict={
            kernel_size: 3,
            threshold: 0.5,
            niter: 1,
            },
    ) -> np.ndarray:
    """
    """
    pass


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
    pass