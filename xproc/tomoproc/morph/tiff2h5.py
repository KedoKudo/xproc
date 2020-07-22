#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Using the experiment config file (YAML) to convert tiff images to a single
HDF5 archive that has the same structure from direct output of area detector.
"""

import os
import h5py
import numpy as np
from tifffile import imread
from xproc.util.io import load_yaml
from xproc.util.io import recursive_save_dict_to_h5group
from xproc.util.memory import fit_in_memory


def pack_tiff_to_hdf5(fconfig: str) -> None:
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
    cfg = cfg_all["morph"]
    fpath = cfg["file_path"]
    fnpre = cfg["file_prefix"]
    ffmt = cfg["file_format"]

    # with h5py.File(os.path.join(fpath, f"{fnpre}.h5")) as h5f:
    with h5py.File(f"{fnpre}.h5") as h5f:
        # write the configuration
        recursive_save_dict_to_h5group(h5f, "/config/", cfg_all)

        # get example image
        # NOTE:
        # assuming that _ is used to concatenate parital string names
        _padding = cfg["numpadding"]
        _fn = f"{fnpre}_{str(cfg['front_white'][0]).zfill(_padding)}.{ffmt}"
        # _img = imread(os.path.join(fpath, _fn))
        _img = imread(fpath + _fn)
        _nrow, _ncol = _img.shape
        _dtype = _img.dtype

        # add the theta array
        omega_start = cfg["omegas"]["start"]
        omega_delta = cfg["omegas"]["step"]
        omega_len = cfg["projections"][1] - cfg["projections"][0] + 1
        omegas = np.arange(
            omega_start, omega_start + (omega_len - 0.5) * omega_delta, omega_delta
        )
        omegas = (
            np.radians(omegas)
            if cfg["omegas"]["unit"].lower() in ["deg", "degree", "degrees"]
            else omegas
        )
        _dst = h5f.create_dataset("omegas", data=omegas)

        # write four types of img stack to h5 archive
        for k, v in {
            "front_white": "/exchange/data_white_pre",
            "projections": "/exchange/data",
            "back_white": "/exchange/data_white_post",
            "back_dark": "/exchange/data_dark",
        }.items():
            _nimgs = np.arange(cfg[k][0], cfg[k][1] + 1)
            if fit_in_memory(
                datatype="int16", size=(len(_nimgs), _nrow, _ncol), overhead_factor=0.5
            ):
                fns = [f"{fnpre}_{str(n).zfill(_padding)}.{ffmt}" for n in _nimgs]
                fls = [os.path.join(fpath, me) for me in fns]
                projs = np.empty((len(_nimgs), _nrow, _ncol), dtype=_dtype)
                for n, fn in enumerate(fls):
                    projs[n, :, :] = imread(fn)
                # use best compression mode
                _dst = h5f.create_dataset(
                    v,
                    data=projs,
                    chunks=True,
                    compression="gzip",
                    compression_opts=9,
                    shuffle=True,
                )
            else:
                _dst = h5f.create_dataset(
                    v,
                    (len(_nimgs), _nrow, _ncol),
                    chunks=True,
                    dtype=_dtype,
                    compression="gzip",
                    compression_opts=9,
                )
                # TODO:
                # this loop is really slow, needs better ways to deal with it...
                from tqdm import tqdm

                for idx, n in tqdm(enumerate(_nimgs)):
                    _fn = f"{fnpre}_{str(n).zfill(_padding)}.{ffmt}"
                    _dst[idx, :, :] = imread(os.path.join(fpath, _fn))


@log_exception(logger_default)
def pack_h5_to_hdf5(fconfig: str) -> None:
    """
    Description
    -----------
        Combine smaller HDF5 archives from wedge scans to a single HDF5
        archive based on given information

    Parameters
    ----------
    fconfig: str
        configuration file, must contain a 'molt' block to guide the 
        conversion

    Returns
    -------
    None
    """
    raise NotImplementedError("coming soon")


if __name__ == "__main__":
    # quick testing
    import sys

    pack_tiff_to_hdf5(sys.argv[1])
