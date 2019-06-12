#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provide functions that operate on projections
"""
from   typing               import Tuple
import numpy                as     np
from   tomopy               import minus_log
from   tomopy               import find_center_pc
from   tomoproc.util.logger import logger_default
from   tomoproc.util.logger import log_exception


def detect_bad_proj(
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
