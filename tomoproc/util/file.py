#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common file loading/writing functions.

h5 loader/writer ref:
https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
"""

import yaml
import h5py
import numpy as np
from tomoproc.util.logger import log_exception
from tomoproc.util.logger import logger_default


@log_exception(logger_default)
def load_yaml(fname: str) -> dict:
    """generic safe_load wrapper for yaml archive"""
    # NOTE: error handeling is done via logger
    with open(fname, "r") as f:
        dataMap = yaml.safe_load(f)

    return dataMap


@log_exception(logger_default)
def write_yaml(fname: str, data: dict) -> None:
    """generic output handler for yaml archive"""
    with open(fname, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


@log_exception(logger_default)
def write_h5(fname: str, data: dict, mode: str = "a") -> None:
    """generic simper HDF5 archive writer"""
    with h5py.File(fname, mode) as f:
        recursive_save_dict_to_h5group(f, "/", data)


@log_exception(logger_default)
def load_h5(fname: str, path: str = "/") -> dict:
    """generic simple HDF5 archive loader"""
    with h5py.File(fname, "r") as f:
        dataMap = recursive_load_h5group_to_dict(f, path)
    return dataMap


@log_exception(logger_default)
def recursive_load_h5group_to_dict(h5file: "h5py.File", path: str,) -> dict:
    """recursively load data from HDF5 archive as dict"""
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursive_load_h5group_to_dict(h5file, f"{path}{key}/")
    return ans


@log_exception(logger_default)
def recursive_save_dict_to_h5group(h5file: "h5py.File", path: str, dic: dict,) -> None:
    """recursively write data to HDF5 archive"""
    for key, item in dic.items():
        if isinstance(item, dict):
            recursive_save_dict_to_h5group(h5file, path + key + "/", item)
        elif isinstance(item, list):
            h5file.create_dataset(
                path + key,
                data=np.array(item),
                chunks=True,
                compression="gzip",
                compression_opts=9,
                shuffle=True,
            )
            # h5file[path + key] = np.array(item)
        else:
            h5file[path + key] = item


if __name__ == "__main__":
    pass
