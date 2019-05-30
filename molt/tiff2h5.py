#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Using the experiment config file (YAML) to convert tiff images to a single
HDF5 archive that has the same structure from direct output of area detector.
"""

import os
import h5py
from   tifffile             import imread
from   tomoproc.util.file   import load_yaml
from   tomoproc.util.file   import recursive_save_dict_to_h5group
from   tomoproc.util.logger import logger_default
from   tomoproc.util.logger import log_exception


@log_exception(logger_default)
def pack_tiff_to_hdf5(fconfig:  str) -> None:
    """
    Description
    -----------
        Convert the tiff images to a single HDF5 archive based on given
        configuration YAML file
    
    Parameters
    ----------
    fconfig: str
        configuration file, must contain a 'molt' block to guide the 
        conversion

    Returns
    -------
    None

    Note
    ----
        Since it is most likely that the physical memory cannot hold the
        entire image stack, we are relying on the h5py dynamic writing
        support to add one image at a time.
    """
    cfg_all = load_yaml(fconfig)
    cfg = cfg_all['molt']
    fpath = cfg['file_path']
    fnpre = cfg['file_prefix']
    ffmt  = cfg['file_format']

    # craete HDF5 file handle
    h5f = h5py.File(os.path.join(fpath, f"{fnpre}.h5"))

    # write the configuration
    recursive_save_dict_to_h5group(h5f, '/config', cfg_all)

    # get example image
    _padding = cfg['numpadding']
    _fn = f"{fnpre}_{str(cfg['front_white'][0]).zfill(_padding)}.{ffmt}"
    _img = imread(os.path.join(fpath, _fn))
    _nrow, _ncol = _img.shape
    _dtype = _img.dtype

    # write four types of img stack to h5 archive
    for k, v in {
        'front_white': '/exchange/data_white_pre',
        'projections': '/exchange/data',
        'back_white' : '/exchange/data_white_post',
        'back_dark'  : '/exchange/data_white_dark',
        }:
        _nimg = cfg[k][1] - cfg[k][0] + 1
        _dst = h5f.create_dataset(v, (_nimg, _nrow, _ncol), dtype=_dtype)
        for n in range(cfg[k][0],  cfg[k][1]+1):
            _fn = f"{fnpre}_{str(n).zfill(_padding)}.{ffmt}"
            _dst[n,:,:] = imread(os.path.join(fpath, _fn))
    
    # close file
    h5f.close()


if __name__ == "__main__":
    pass
