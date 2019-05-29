#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common file loader functions.
"""

import yaml

def load_yaml(fname: str) -> dict:
    """generic safe_load wrapper for yaml archive"""
    try:
        with open(fname, 'r') as f:
            dataMap = yaml.safe_load(f)
    except IOError as e:
        print(f"Cannot open YAML file {fname}")
        print(f"IOError: {e}")
    
    return dataMap