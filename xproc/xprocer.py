#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""xprocer
xprocer is the CLI for automated data processing of HT-HEDM data.

Usage:
    xprocer.py -h | --help
    xprocer.py --version

Options:
    -h --help     Show this screen.
    --version     Show version.
    -v --verbose  Verbose output
"""

import luigi
import os
import datetime
from docopt import docopt

from xproc import tomoproc
from xproc import ffproc 
from xproc import nfproc

## --- Tomography Reconstruction --- ##
class TomoMorph(luigi.Task):
    """Create a HDF archive from a set of TIFF files"""
    date = luigi.DateIntervalParameter(default=datetime.date.today())
    conf = luigi.Parameter(default='morph.yml')  # configuration for converting TIFF/RAW images into HDF5 archive

    def run(self):
        # lazy import
        tomoproc.morph.tiff2h5.pack_tiff_to_hdf5(self.conf)
        # write the marker file
        with open(f'morph_complete_{date}.txt', 'w') as f:
            f.write(f'TIFF -> HDF5 conversion finished at {datetime.datetime.now()}')

    def output(self):
        return luigi.LocalTarget(f'morph_complete_{self.date}.txt')


class TomoRecon(luigi.Task):
    """Top level tomography reconstruction task"""

    # NOTE:
    # - Xproc is using harddrive caching to be compatible with Luigi.
    # - This design means that a final cleanup is necesary before committing proessed
    #   data to Voyager
    # - It would be beneficial to find a way to work around this, caching data in memory
    #
    hdf5 = luigi.Parameter(default="test.h5")  # HDF5 archive from Tomo experiment
    conf = luigi.Parameter(default="test.yml")  # Experiment control YAML file
    wdir = luigi.Parameter(
        default="xproc_wd"
    )  # Working directory, caching intermedia results

    def run(self):
        pass

    def output(self):
        pass

    def requires(self):
        return []


class TomoNoiseReduction(luigi.Task):
    """Reduce salt&pepper noise with selective median filter"""

    hdf5 = luigi.Parameter(default="test.h5")  # HDF5 archive from Tomo experiment
    conf = luigi.Parameter(default="test.yml")  # Experiment control YAML file
    wdir = luigi.Parameter(
        default="xproc_wd"
    )  # Working directory, caching intermedia results

    def run(self):
        pass

    def output(self):
        pass


class TomoMotionCorrection(luigi.Task):
    """Correct for tomo detector drifting"""

    def run(self):
        pass

    def ouptut(self):
        pass

    def requires(self):
        pass


class TomoFOVReduction(luigi.Task):
    """Use the detected slit corners to reduce img to FOV only"""

    def run(self):
        pass

    def output(self):
        pass

    def requires(self):
        pass


class TomoDetectCorruptedFrames(luigi.Task):
    """Detect corrupted frames"""

    def run(self):
        pass

    def output(self):
        pass

    def requires(self):
        pass


class TomoBackgroundCorrection(luigi.Task):
    """Normalize and remove background in sinograms"""

    def run(self):
        pass

    def output(self):
        pass

    def requires(self):
        pass


## --- ff-HEDM reconstruction --- ##
class FFHEDMRecon(luigi.Task):
    """Top level ff reconstruction control"""

    def run(self):
        pass

    def output(self):
        pass

    def requires(self):
        pass


class FFHEDMCaliCeO2(luigi.Task):
    """Use CeO2 to calibrate detector"""

    def run(self):
        pass

    def output(self):
        pass

    def requires(self):
        pass


class FFHEDMCaliAu(luigi.Task):
    """Use Au to calibrate detector"""

    def run(self):
        pass

    def output(self):
        pass

    def requires(self):
        pass


# --- nf-HEDM reconstruction --- #
# TODO:


if __name__ == "__main__":
    argvs = docopt(__doc__, argv=None, help=True, version="xprocer v0.0.1")
    verbose_output = argvs["--verbose"]

    if verbose_output:
        print(argvs)

    luigi.run()
