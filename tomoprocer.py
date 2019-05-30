#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tomoprocer
    molt: legacy data support
        convert tiff|binary to HDF5 archive for subsequent analysis
    prep: perform (* optional, ! mandetoray)
        * sinogram background normalization
        * slit detection
        * detector drift (translate+rotate) correction
        * data reduction (corpping)
        * stitching
        ! -ln(img)
        ! action logging to H5 archive
    recon: perform tomography reconstruction using external engine specified
           in configuration file
        * tomopy
        * tomoMPI
        * MIDAS (upcoming)
    analyze: perform specified analysis on reconstruction volume
        * porosity characterization
        * phase boundary detection
        * crack network visualization (vtk)

Usage:
    tomoprocer.py molt     <CONFIGFILE>
    tomoprocer.py prep     <CONFIGFILE>
    tomoprocer.py recon    <CONFIGFILE>
    tomoprocer.py analyze  <CONFIGFILE>
    tomoprocer.py -h | --help
    tomoprocer.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
"""

from docopt import docopt
from tomoproc.molt.tiff2h5 import pack_tiff_to_hdf5

if __name__ == "__main__":
    argvs = docopt(__doc__, argv=None, help=True, version="tomoprocer v0.0.1")
    print(argvs)

    if argvs['molt']:
        try:
            pack_tiff_to_hdf5(argvs['<CONFIGFILE>'])
        except:
            raise FileExistsError('remove previous generated H5 archive')
    elif argvs['prep']:
        pass
    elif argvs['recon']:
        pass
    elif argvs['analyze']:
        pass
    else:
        raise ValueError('Please use --help to check available optoins')
