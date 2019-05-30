#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common file loading/writing functions.

h5 loader/writer ref:
https://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py
"""

import yaml
import h5py
from   tomoproc.util.logger import log_exception
from   tomoproc.util.logger import logger_default


@log_exception(logger_default)
def load_yaml(fname: str) -> dict:
    """generic safe_load wrapper for yaml archive"""
    # NOTE: error handeling is done via logger
    with open(fname, 'r') as f:
        dataMap = yaml.safe_load(f)

    return dataMap


@log_exception(logger_default)
def write_yaml(fname: str, data: dict) -> None:
    """generic output handler for yaml archive"""
    with open(fname, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)


@log_exception(logger_default)
def write_h5(fname: str, data: dict) -> None:
    """generic simper HDF5 archive writer"""
    with h5py.File(fname, 'w') as f:
        recursively_save_dict_contents_to_group(f,'/',data)


@log_exception(logger_default)
def load_h5(fname: str, path: str='/') -> dict:
    """generic simple HDF5 archive loader"""
    with h5py.File(fname, 'r') as f:
        dataMap = recursively_load_dict_contents_from_group(f, path)
    return dataMap


@log_exception(logger_default)
def recursively_load_dict_contents_from_group(h5file: "h5py.File", 
                                              path: str,
        ) -> dict:
    """recursively load data from HDF5 archive as dict"""
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, f"{path}{key}/")
    return ans


@log_exception(logger_default)
def recursively_save_dict_contents_to_group(h5file: "h5py.File", 
                                            path: str, 
                                            dic: dict,
        ) -> None:
    """recursively write data to HDF5 archive"""
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes,int,float,np.bool_)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError(f'Cannot save {item} type')


if __name__ == "__main__":
    pass
