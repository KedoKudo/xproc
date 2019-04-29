import numpy as np
import sys
import math
import time
import datetime
from array import *
from PIL import Image
import Transform
import avgTransform
import projTransform
import matReader
import Stitching_InitConv_TIFFtoHDF4_single
import Stitching_InitConv_TIFFtoHDF4

def prefListToArrayD(prefList):
    if isinstance(prefList, list):
        prefList = array('d', prefList)
    return  prefList

def prefListToArrayI(prefList):
    if isinstance(prefList, list):
        prefList = array('i', prefList)
    else:
        prefList = array('i', [prefList])
    return  prefList

def prefListToListArrayI(prefList):
    if isinstance(prefList, list):
        prefList =  [ array('i', prefList) ]
    return  prefList

def prefListToListArrayD(prefList):
    if isinstance(prefList, list):
        prefList =  [ array('d', prefList) ]
    return  prefList

def prefIntToListD(prefList):
    if isinstance(prefList, int):
        prefList =  [ prefList*1.0 ]
    return  prefList

def prefIntToListI(prefList):
    if isinstance(prefList, int):
        prefList =  [ prefList ]
    else:
        prefList = prefList
    return  prefList

def prefInitMatlab():
    f = matReader.loadmat('test.mat')
    prefs = f['prefs']

    prefs['series'] = prefListToArrayI(prefs['series'])

    prefs['vis']['projection']['imgRange'] = prefListToArrayD(prefs['vis']['projection']['imgRange'])
    prefs['vis']['beamProfileInitial']['imgRange'] = prefListToArrayD(prefs['vis']['beamProfileInitial']['imgRange'])
    prefs['vis']['beamProfileFinal']['imgRange'] = prefListToArrayD(prefs['vis']['beamProfileFinal']['imgRange'])
    prefs['vis']['dark']['imgRange'] = prefListToArrayD(prefs['vis']['dark']['imgRange'])
    prefs['vis']['stitching']['ColorCodeLim'] = prefListToArrayD(prefs['vis']['stitching']['ColorCodeLim'])

    prefs['ROIx'] = prefListToListArrayI(prefs['ROIx'])
    prefs['ROIy'] = prefListToListArrayI(prefs['ROIy'])

    prefs['RotationAngle'] = prefIntToListD(prefs['RotationAngle'])


    prefs['cropROIx'][:] = [x-1 for x in prefs['cropROIx']]
    prefs['cropROIy'][:] = [x-1 for x in prefs['cropROIy']]
    prefs['cropROIx'] = prefListToListArrayI(prefs['cropROIx'])
    prefs['cropROIy'] = prefListToListArrayI( prefs['cropROIy'])

    prefs['rotAxis'] = prefIntToListD(prefs['rotAxis'])

    prefs['vis']['projection']['imgRange'] = prefListToArrayD(prefs['vis']['projection']['imgRange'])
    prefs['slits']['slitsnum'] = prefListToArrayD(prefs['slits']['slitsnum'])

    for i in range(len(prefs['slits']['ROIx'])):
        prefs['slits']['ROIx'][i] = prefListToArrayD(prefs['slits']['ROIx'][i])
    for i in range(len(prefs['slits']['ROIy'])):
        prefs['slits']['ROIy'][i] = prefListToArrayD(prefs['slits']['ROIy'][i])

    prefs['slits']['referenceNum'] = prefListToArrayD(prefs['slits']['referenceNum'])

    prefs['slitposcheck']['projection']['Num'] = prefListToListArrayD(prefs['slitposcheck']['projection']['Num'])

    prefs['correl']['smallROIy'] = prefListToArrayD(prefs['correl']['smallROIy'])
    prefs['correl']['smallROIx'] = prefListToArrayD(prefs['correl']['smallROIx'])
    prefs['correl']['bigROIx'] = prefListToArrayD(prefs['correl']['bigROIx'])

    prefs['CropROIxAfterStitch'] = prefListToArrayD(prefs['CropROIxAfterStitch'])
    prefs['CropROIyAfterStitch'] = prefListToArrayD(prefs['CropROIyAfterStitch'])

    prefs['out']['ROIx'] = prefListToArrayD(prefs['out']['ROIx'])
    prefs['out']['ROIy'] = prefListToArrayD(prefs['out']['ROIy'])

    prefs['translx'] = prefIntToListI(prefs['translx'])
    prefs['transly'] = prefIntToListI(prefs['transly'])

    prefs['beamProfileInitial']['Num']=prefListToListArrayI(prefs['beamProfileInitial']['Num'])
    prefs['beamProfileFinal']['Num']=prefListToListArrayI(prefs['beamProfileFinal']['Num'])
    prefs['projection']['Num']=prefListToListArrayI(prefs['projection']['Num'])
    prefs['dark']['Num']=prefListToListArrayI(prefs['dark']['Num'])

    prefs['filePath'] = str(prefs['filePath'])
    prefs['beamProfileInitial']['Prefix'] = [str(prefs['beamProfileInitial']['Prefix'])]
    prefs['beamProfileInitial']['extension'] = str(prefs['beamProfileInitial']['extension'])
    prefs['beamProfileFinal']['Prefix'] = [str(prefs['beamProfileFinal']['Prefix'])]
    prefs['beamProfileFinal']['extension'] = str(prefs['beamProfileFinal']['extension'])
    prefs['dark']['Prefix'] = [str(prefs['dark']['Prefix'])]
    prefs['dark']['extension'] = str(prefs['dark']['extension'])
    prefs['userID'] = str(prefs['userID'])
    prefs['out']['filePrefix'] = str(prefs['out']['filePrefix'])
    prefs['out']['dirname'] = str(prefs['out']['dirname'])
    prefs['out']['filePath'] = str(prefs['out']['filePath'])
    prefs['out']['logFile'] = str(prefs['out']['logFile'])
    prefs['slits']['slitfname'] = str(prefs['slits']['slitfname'])

    prefs['projection']['Prefix'] = [str(prefs['projection']['Prefix'])]

    return prefs

def prefInitSinglePy():
    prefs = dict()
    prefs = Stitching_InitConv_TIFFtoHDF4_single.initPyPref(prefs)
    return prefs

def prefInitPy():
    prefs = dict()
    prefs = Stitching_InitConv_TIFFtoHDF4.initPyPref(prefs)
    return prefs

def dataPrepare(prefs):
    fileID = '/local/kyue/test/txt' # change the fileId here
    monitor=[]
    nOut = 0; moncnt = 0;


    prefs, nOut, monitor, moncnt, dataWhite1 = avgTransform.avgTransform(prefs, nOut, 'WHITE1', 'beamProfileInitial', fileID, moncnt);

    nOutst = nOut;

    prefs, nOut, monitor, moncnt, dataProj = projTransform.projTransform(prefs, nOutst, 'PROJ', 'projection', fileID, moncnt);

    prefs, nOut, monitor, moncnt, dataWhite2 = avgTransform.avgTransform(prefs, nOut, 'WHITE2', 'beamProfileFinal', fileID, moncnt);

    prefs, nOut, monitor, moncnt, dataDark = avgTransform.avgTransform(prefs, nOut, 'DARK', 'dark', fileID, moncnt);

    return dataWhite1, dataProj, dataWhite2, dataDark


