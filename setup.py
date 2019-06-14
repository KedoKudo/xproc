#!/usr/bin/env python

import glob
import setuptools
import tomoproc

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='tomoproc',  
     version='0.0.1',
     scripts=['tomoprocer.py'],
     author="KedoKudo",
     author_email="chenzhang8722@gmail.com",
     description="Automated tomography reconstruction meta-package",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/KedoKudo/tomoproc.git",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )