#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tomoprocer
    morph: legacy data support
        convert tiff|binary to HDF5 archive for subsequent analysis
    prep:  preprocessing tomography data
                            ----------------
                            |avaiable modes|
                            ----------------
        | ==========| === | === | === | === | ==== | ==== | === |
        |           | inr | bgn | cdd | dcp | bifc | crop | dtc |
        | --------- | --- | --- | --- | --- | ---- | ---- | --- |
        | express   |  y  |  y  |  n  |  n  |  n   |  n   |  n  |
        | lite      |  y  |  y  |  y  |  y  |  n   |  y   |  n  |
        | royal     |  y  |  y  |  y  |  y  |  y   |  y   |  y  |
        | customize |  ?  |  ?  |  ?  |  ?  |  ?   |  ?   |  ?  |
        | ==========| === | === | === | === | ==== | ==== | === |
        
        * [inr]  impulse_noise_removal
        * [bgn]  background_normalization               # sinogram method
        * [cdd]  correct_detector_drifting              # through slit detection
        * [dcp]  detect_corrupted_proj                  # through 180 deg pair matching
        * [bifc] beam_intensity_fluctuation_correction  # through sinogram
        # [crop] data reduction (corpping)
        * [dtc]  correct_detector_tilt                  # rotation axis tilt correction

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
    tomoprocer.py morph    <CONFIGFILE>
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

if __name__ == "__main__":
    argvs = docopt(__doc__, argv=None, help=True, version="tomoprocer v0.0.1")
    print(argvs)

    if argvs['morph']:
        # lazy import
        from tomoproc.morph.tiff2h5 import pack_tiff_to_hdf5
        try:
            pack_tiff_to_hdf5(argvs['<CONFIGFILE>'])
        except:
            raise FileExistsError('remove previous generated H5 archive')
    elif argvs['prep']:
        # lazy import
        from tomoproc.prep.detection  import detect_slit_corners
        from tomoproc.prep.detection  import detect_corrupted_proj
        from tomoproc.prep.detection  import detect_rotation_center
        from tomoproc.prep.correction import correct_detector_drifting
        from tomoproc.prep.correction import correct_detector_tilt
        from tomoproc.prep.correction import denoise
        from tomoproc.prep.correction import beam_intensity_fluctuation_correction as bifc
        from tomoproc.util.file       import load_h5
        from tomoproc.util.file       import load_yaml
        # step_0: parse the configuration file
        cfg = load_yaml(argvs['<CONFIGFILE>'])['tomo']  # only load the tomography related section

        # step_1: based on the prep plan selected [express|full|customize] to provide a quick 
        #         summary of the processing plan

        # step_2: perform the auto-processing as described in previous step

        # step_3: export the data to a HDF archive, compressed by default

    elif argvs['recon']:
        pass
    elif argvs['analyze']:
        pass
    else:
        raise ValueError('Please use --help to check available optoins')
