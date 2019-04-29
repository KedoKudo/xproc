import numpy as np
import sys
import math
import time
import datetime
from array import *
from PIL import Image

from skimage.transform import rotate

import scipy


def paddingZero(num, fullsize):
    string = str(num)
    strRes = string.zfill(fullsize)
    return strRes

def stitching_calcbefore(prefs):
    # if isinstance(prefs['series'], float):
    #     j = 0
    #     orixsize = len(prefs['ROIx'][j])
    #     oriysize = len(prefs['ROIy'][j])
    #     prefs['out']['xposInBBox'].append(int((prefs['bboxx'] - orixsize)/2) + prefs['translx'][j] + prefs['cropROIx'][j][0])
    #     prefs['out']['yposInBBox'].append(int((prefs['bboxy'] - oriysize)/2) + prefs['transly'][j] + prefs['cropROIy'][j][0])
    if isinstance(prefs['series'], array):
        count = 0
        while (count < prefs['series'].__len__()):
            j = int(prefs['series'][count])-1
            orixsize = len(prefs['ROIx'][j])
            oriysize = len(prefs['ROIy'][j])
            # for python initialize file
            prefs['out']['xposInBBox'].append(int (math.ceil((prefs['bboxx'] - orixsize)/2.)) + prefs['translx'][j] + prefs['cropROIx'][j][0])
            prefs['out']['yposInBBox'].append(int (math.ceil((prefs['bboxy'] - oriysize)/2.)) + prefs['transly'][j] + prefs['cropROIy'][j][0])
            # for matfile initiaize file
            # prefs['out']['xposInBBox'].append(int (math.ceil((prefs['bboxx'] - orixsize)/2.)) + prefs['translx'][j] + prefs['cropROIx'][j][0]-1)
            # prefs['out']['yposInBBox'].append(int (math.ceil((prefs['bboxy'] - oriysize)/2.)) + prefs['transly'][j] + prefs['cropROIy'][j][0]-1)
            count = count + 1
    return prefs

# def slitposcorr(dataArray, prefs, i, kind, num):
#     if( (kind == 'projection') or (kind == 'beamProfileInitial') or (kind == 'beamProfileFinal') ):
#         if(len(prefs[kind]['Num'][i]) == len(prefs[slits][slitpos][kind][medselect][y][i]) ):
#             rowShift = -round(prefs[slits][slitpos][kind][medselect][y][i][num])
#             columnShift = -round(prefs[slits][slitpos][kind][medselect][x][i][num])
#             dataArray = np.roll(dataArray, rowShift, axis = 0 )
#             dataArray = np.roll(dataArray, columnShift, axis = 1)
#         else:
#             print "ERROR: The length of the slitpos array does not match with that of the %s array.\n Try to regenerate the slitpos info: prefs.slits.usePrevious=F." % kind
#             return dataArray
#     else:
#         print "No slitpos correction for the %s.\n" % kind
#     return dataArray

def preprocess(dataArray, prefs, i): #outROIx will strictly equals to the index in matlab, if used as index need to minus 1
    outROIx = range(1, len(dataArray.T)+1)
    outROIy = range(1, len(dataArray)+1)

    if prefs['shouldRotate']:
        dataArray = scipy.ndimage.interpolation.rotate(dataArray, prefs['RotationAngle'][i])
        outROIx = range(1, len(dataArray.T)+1) # it is still use the same number as matlab, but if using as the index, will need to change
        outROIy = range(1, len(dataArray)+1)

    if prefs['shouldCrop']:
        cropROIyf=np.array(prefs['cropROIy'][i])
        cropROIy = cropROIyf.astype(int)
        cropROIxf=np.array(prefs['cropROIx'][i])
        cropROIx= cropROIxf.astype(int)
        dataArray = dataArray[ np.ix_(cropROIy, cropROIx)]
        outROIx = range((prefs['ROIx'][i][0]+prefs['cropROIx'][i][0]-1), prefs['ROIx'][i][0]+prefs['cropROIx'][i][-1])
        outROIy = range((prefs['ROIy'][i][0]+prefs['cropROIy'][i][0]-1), prefs['ROIy'][i][0]+prefs['cropROIy'][i][-1])

    if prefs['shouldColumnAverage']:
        dataArray = int( dataArray.mean(0) )
        outROIy = 1
        outROIx = range(1, dataArray.shape[1])


    aveint = 0
    if prefs['vis']['intMonitor']:
        aveint = dataArray.mean()
    return dataArray, outROIx, outROIy, aveint

def stitching(dataArray, prefs):
    stitchedArray = np.zeros(shape=( int(prefs['bboxy']), int(prefs['bboxx']))) # prefs['out']['fileType']
    maxx = 0
    maxy = 0
    minx = sys.maxint
    miny = sys.maxint

    if isinstance(prefs['series'], array):
        count = 0
        while (count < prefs['series'].__len__()):
            j = int(prefs['series'][count])-1 #euqal to prefs.series doing this, because returned might be from matlab, which could be a float
            px = int(prefs['out']['xposInBBox'][j])+1
            py = int(prefs['out']['yposInBBox'][j])+1
            dataArray[j] = dataArray[j].astype(int)
            stitchedArray[(py-1):(py+dataArray[j].shape[0]-1), (px-1):(px+dataArray[j].shape[1]-1)]= dataArray[j]
            minx = min(minx, px)
            maxx = max(maxx,(px+dataArray[j].shape[1]-1))
            miny = min(miny, py)
            maxy = max(maxy,(py+dataArray[j].shape[0]-1))

            count = count + 1

    if not prefs['doVis']['stitching']:
        stitchedArray = stitchedArray[(miny-1):maxy, (minx-1):maxx]

    if prefs['shouldRotateAfterStitch'] and (not prefs['doVis']['stitching']):
        stitchedArray = scipy.ndimage.interpolation.rotate(stitchedArray, prefs['RotationAngleAfterStitch'])
        # stitchedArrayImage = Image.fromarray(stitchedArray)
        # stitchedArrayImage = stitchedArrayImage.rotate(prefs['RotationAngleAfterStitch'], Image.BILINEAR )
        # stitchedArray = np.array(stitchedArrayImage)

    stitchedArrayNew = np.zeros(shape=( int(prefs['bboxy']), int(prefs['bboxx'])))

    if prefs['shouldCropAfterStitch'] and (not prefs['doVis']['stitching']):
        miny = miny + int(prefs['CropROIyAfterStitch'][0])-1
        maxy = miny + prefs['CropROIyAfterStitch'].__len__()-1
        minx = minx + int(prefs['CropROIxAfterStitch'][0])-1
        maxx = minx + prefs['CropROIxAfterStitch'].__len__()-1
        CropROIyAfterStitchf=np.array(prefs['CropROIyAfterStitch'])
        cropROIy = CropROIyAfterStitchf.astype(int)
        CropROIxAfterStitchf=np.array(prefs['CropROIxAfterStitch'])
        cropROIx= CropROIxAfterStitchf.astype(int)
        stitchedArrayNew[(miny-1):maxy, (minx-1):maxx]=stitchedArray[np.ix_(cropROIy, cropROIx)]
    else:
        if not prefs['doVis']['stitching']:
            stitchedArrayNew[(miny-1):maxy, (minx-1):maxx] = stitchedArray
        else:
            stitchedArrayNew = stitchedArray

    stitchedArray = stitchedArrayNew

    if not prefs['doVis']['stitching']:
        stitchedArray = stitchedArray[(miny-1):maxy, (minx-1):maxx]
    else:
        print " do vis stitich is true"


    return stitchedArray, minx, maxx

def stitching_calcafter(prefs, minx, maxx, sizex, sizey):
    if not prefs['doVis']['stitching']:
        if isinstance(prefs['series'], array):
            count = 0
            while (count < prefs['series'].__len__()):
                j = int(prefs['series'][count])-1
                # prefs['out']['rotAxis'][j]=prefs['out']['xposInBBox'][j]-prefs['ROIx'][j][0]+1-prefs['cropROIx'][j][0]+1+prefs['rotAxis'][j]-minx+1
                # print prefs['rotAxis'][j]
                prefs['out']['rotAxis'].append(prefs['out']['xposInBBox'][j]-prefs['ROIx'][j][0]+1-prefs['cropROIx'][j][0]+1+prefs['rotAxis'][j]-minx+1)
                prefs['out']['shift'].append((maxx-minx+1)/2.0-prefs['out']['rotAxis'][j])
                count = count + 1
    else:
        if isinstance(prefs['series'], array):
            count = 0
            while (count < prefs['series'].__len__()):
                j = int(prefs['series'][count])-1
                prefs['out']['rotAxis'].append(prefs['out']['xposInBBox'][j]-prefs['ROIx'][j][0]+1-prefs['cropROIx'][j][0]+1+prefs['rotAxis'][j])
                prefs['out']['shift'].append((maxx-minx+1)/2.0-prefs['out']['rotAxis'][j])
                count = count + 1

    prefs['out']['ROIx']=range(1,int(sizex))
    prefs['out']['ROIy']=range(1,int(sizey))

    return prefs


