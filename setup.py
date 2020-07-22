#!/usr/bin/env python

import glob
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xproc",
    version="0.0.2",
    scripts=["xproc/xprocer.py"],
    author="KedoKudo",
    author_email="kedokudo@protonmail.com",
    description="Data processing package for HT-HEDM instrument at Advanced Photon Source",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KedoKudo/xproc.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
