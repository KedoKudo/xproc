#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common file loader functions.
"""

import yaml

from tomoproc.util.logger import log_exception
from tomoproc.util.logger import logger_default


@log_exception(logger_default)
def load_yaml(fname: str) -> dict:
    """generic safe_load wrapper for yaml archive"""
    # NOTE: error handeling is done via logger
    with open(fname, 'r') as f:
        dataMap = yaml.safe_load(f)

    return dataMap


if __name__ == "__main__":
    pass
