# tomoproc

This repository, __tomoproc__ provides necessary toolkits for the pre&amp;post processing of tomography data collected at

* 1-ID@APS
* 6-BM-A@APS
* 6-ID-D@APS

## Quick start

```bash
>> tomorpocer --help
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
```

## Required 3rd-party libraries

* __tomopy__: conda install -c conda-forge tomopy
* __tifffile__:  pip install tifffile
* __docopt__: pip install docopt
* __vtk__: pip install vtk