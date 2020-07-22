#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory related tools to assisst the decision making in tomoproc

NOTE:
    The tomography image-stack can be too big to fit in the memory, therefore 
    one might have to perform certain operation one slice/sinogram at a time.

For example:

>>> fit_in_memory('float', (10,1024,1024))
True
"""

import psutil
import numpy as np


def fit_in_memory(datatype="float", size=(1024, 1024), overhead_factor=1.1):
    """
    Description
    -----------
        Check if the image stack can fit in the memory

    Parameters
    ----------
    datatype: str
        data type
    size: tuple
        image stack (numpy array) size
    overhead_factor: int
        factor to include potential overhead (should be relatively small)
    
    Return
    ------
    bool:
        whether the image stack will fit in memory or not
    """
    mem_available = psutil.virtual_memory().available
    # evaluated with np.random.random(1024,dtype=datatype)
    # TODO:
    #   The calculation here is off
    _factor = {
        "float": 8192 / 1024,
        "float64": 8192 / 1024,
        "float32": 4096 / 1024,
        "float16": 2048 / 1024,
        "int": 8192 / 1024,
        "int64": 8192 / 1024,
        "int32": 4096 / 1024,
        "int16": 2048 / 1024,
        "bool": 1024 / 1024,
    }[datatype.lower()] * overhead_factor
    mem_need = np.prod(size) * _factor
    return mem_available > mem_need


if __name__ == "__main__":
    import doctest

    doctest.testmod()
