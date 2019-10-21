# xproc

This repository, __xproc__ provides necessary toolkits for the pre&amp;post processing of tomography, FF, and NF data collected at

* 1-ID@APS
* 6-BM-A@APS
* 6-ID-D@APS

## Tomoproc

The subpackage `tomoproc` is for process tomography data.

### Quick start

```bash
>> tomorpocer --help
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
```

### Required 3rd-party libraries

* __tomopy__: conda install -c conda-forge tomopy
* __docopt__: pip install docopt
* __vtk__: pip install vtk


### Docker

Container building recipe for using tomoproc.

## ffproc

The subpackge `ffproc` is for processing ff-HEDM data.

## nfproc

The subpackge `nfproc` is for processing nf-HEDM data.
