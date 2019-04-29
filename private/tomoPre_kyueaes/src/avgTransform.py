import numpy as np
import sys
import math
import time
import datetime
from array import *
from PIL import Image
import Transform
import matReader
import Stitching_InitConv_TIFFtoHDF4_single
import ctypes
from multiprocessing import Pool,Process, Array

def child(i, procs, step, totalimage, prefs, kind, j, dataAve, monitor, nOut):

    Start = i*step;
    End = (i+1)*step;
    if(Start >= totalimage):
        return 0
    if(End > totalimage):
        End = totalimage;

    for iavg in range(Start, End ):
        filename = "%s/%s%s.%s" % ( prefs['filePath'], prefs[kind]['Prefix'][0], Transform.paddingZero(int(prefs[kind]['Num'][j][iavg]), prefs['numberDigit'] ), prefs[kind]['extension'] )
        im = Image.open(filename)
        imarray = np.array(im)
        dataAve[iavg] = imarray[(prefs['ROIy'][j][0]-1):prefs['ROIy'][j][-1] , (prefs['ROIx'][j][0]-1):prefs['ROIx'][j][-1]] # corp the array
        # if prefs['slits']['shouldCorr']: #comment out for this momoent, if need corr, need to initialize in the initialize script
        # dataAve[iavg] = slitposcorr(dataAve[iavg], prefs, j, kind, iavg)
        # print iavg
        if prefs['vis']['intMonitor']:
            _, _, _, aveint = Transform.preprocess(dataAve[iavg], prefs, j)
            monitor[iavg][j] = array('d', [aveint, prefs[kind]['Num'][j][iavg], nOut])

    return 1

def avgTransform(prefs, nOut, sType, kind, fileID, moncnt):
    # monitor=[]

    nOut = nOut + 1;
    # outfilename = "%s/%s_%s.hdf" % (prefs['out']['filePath'], prefs['projection']['Prefix'], Transform.paddingZero(nOut, int(prefs['out']['numberDigit'])))

    dataArray = []
    subROIx = []
    subROIy = []

    # print type(prefs['cropROIx'][0])
    prefs = Transform.stitching_calcbefore(prefs)

    maxx = 0; minx = sys.maxint

    # if isinstance(prefs['series'], float):
    #     j = int (prefs['series'] - 1) #euqal to prefs.series
    #     subROIx.append(0)
    #     subROIy.append(0)
    #     dataAve = np.zeros(shape=( len(prefs[kind]['Num'][j]), len(prefs['ROIy'][j]), len(prefs['ROIx'][j]) ))
    #
    #     for iavg in range(0, len(prefs[kind]['Num'][j]) ):
    #         filename = "%s/%s%s.%s" % ( prefs['filePath'], prefs[kind]['Prefix'], paddingZero(prefs[kind]['Num'][j][iavg], prefs['numberDigit'] ), prefs[kind]['extension'] )
    #         im = Image.open(filename)
    #         imarray = np.array(im)
    #         dataAve[iavg] = imarray[int(prefs['ROIy'][j][0])-1: int(prefs['ROIy'][j][-1]) , int(prefs['ROIx'][j][0])-1 : int(prefs['ROIx'][j][0][-1])  ]
    #
    #         # if prefs['slits']['shouldCorr']:
    #             # dataAve[iavg] = slitposcorr(dataAve[iavg], prefs, j, kind, iavg)
    #
    #         if prefs['vis']['intMonitor']:
    #             _, _, _, aveint = preprocess(dataAve[iavg], prefs, j, im)
    #             monitor.append(array('d', [aveint, prefs[kind]['Num'][j][iavg]]))
    #
    #     dataArray.append(dataAve.mean(0))
    #
    #     dataArray[j], subROIx[j], subROIy[j], _ = preprocess(dataArray[j], prefs, j)

        # dataArray, minx, maxx = stitching(dataArray, prefs)

    # if isinstance(prefs['series'], array):


    for j in prefs['series']:
        j = j -1
        # dataAve = np.zeros(shape=( len(prefs[kind]['Num'][j]), len(prefs['ROIy'][j]), len(prefs['ROIx'][j])  ))

        sharedArray = Array(ctypes.c_double, len(prefs[kind]['Num'][j])*len(prefs['ROIy'][j])*len(prefs['ROIx'][j]), lock=False)
        sharedNpArray = np.frombuffer(sharedArray, dtype=ctypes.c_double)
        dataAve = sharedNpArray.reshape(len(prefs[kind]['Num'][j]), len(prefs['ROIy'][j]), len(prefs['ROIx'][j]))

        totalimage = len(prefs[kind]['Num'][j])

        # sharedMonitor = Array(ctypes.c_double, totalimage*3, lock=False)
        # sharedNpMonitor = np.frombuffer(sharedMonitor, dtype=ctypes.c_double)
        # monitor = sharedNpMonitor.reshape(totalimage, 3)

        sharedMonitor = Array(ctypes.c_double, totalimage*len(prefs['series'])*3, lock=False)
        sharedNpMonitor = np.frombuffer(sharedMonitor, dtype=ctypes.c_double)
        monitor = sharedNpMonitor.reshape(totalimage, len(prefs['series']), 3)

        step = 1
        workjob = []


        procs = totalimage/step + 1

        if step == 0:
            step =1

        for i in range(0, procs):
            process = Process(target=child, args = (i,procs, step, totalimage, prefs, kind, j, dataAve, monitor, nOut))
            workjob.append(process)

        for pIndex in workjob:
            pIndex.start()

        for pIndex in workjob:
            pIndex.join()

        dataArray.append(dataAve.mean(0).astype(int))
        subROIx.append(0)
        subROIy.append(0)
        dataArray[j], subROIx[j], subROIy[j], _ = Transform.preprocess(dataArray[j], prefs, j)

    dataResultArray, minx, maxx = Transform.stitching(dataArray, prefs)
    # print dataArray
    prefs = Transform.stitching_calcafter(prefs, minx, maxx, dataResultArray.shape[1], dataResultArray.shape[0])

    moncnt = moncnt + len(prefs[kind]['Num'][0])
    # print prefs['out']['xposInBBox'][0]

    #write dataArray to the file

    return prefs, nOut, monitor, moncnt, dataResultArray


# def avgTransform(prefs, nOut, sType, kind, fileID, monitor):
#     return prefs, nOut



# prefs = dict()
# f = spio.loadmat('test.mat') # read in a mat file
# print prefs['NumProj']
# prefs = matdata[0,0]
# prefs['ROIy'] = matdata['ROIy']
# a = np.array(prefs['series'])
# print type(a),a, a[0],a[0][0]
# print prefs['ROIy'], type(prefs['ROIy'])

# prefs = dict()
# prefs = Stitching_InitConv_TIFFtoHDF4_single.initPyPref(prefs)

fileID = '/local/kyue/test/txt'
# nOut = 0; moncnt = 0;
# avgTransform(prefs, nOut, 'WHITE1', 'beamProfileInitial', fileID, moncnt);
