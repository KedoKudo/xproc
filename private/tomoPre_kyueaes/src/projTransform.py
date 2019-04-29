import numpy as np
import sys
import math
import time
import datetime
from array import *
from PIL import Image
import Transform
import Stitching_InitConv_TIFFtoHDF4_single
import ctypes
from multiprocessing import Pool,Process, Array, Queue


def child(i, step, totalimage, prefs, kind, finalData, monitor, nOutstart):

    Start = i*step;
    End = (i+1)*step;
    if(Start >= totalimage):
        return 0
    if(End > totalimage):
        End = totalimage;

    for iproj in range(Start,End):
        nOut = nOutstart+iproj
        # outfilename = "%s/%s_%s.hdf" % (prefs['out']['filePath'], prefs['projection']['Prefix'], Transform.paddingZero(nOut, int(prefs['out']['numberDigit'])))
        dataArray = []

        for i in prefs['series']:
            i = i -1
            filename = "%s/%s%s.%s" % ( prefs['filePath'], prefs[kind]['Prefix'][0], Transform.paddingZero(int(prefs[kind]['Num'][i][iproj]), prefs['numberDigit'] ), prefs[kind]['extension'] )
            im = Image.open(filename)
            imarray = np.array(im)
            dataArray.append(0)
            dataArray[i] = imarray[(prefs['ROIy'][i][0]-1):prefs['ROIy'][i][-1] , (prefs['ROIx'][i][0]-1):prefs['ROIx'][i][-1]] # corp the array
            # if prefs['slits']['shouldCorr']: #comment out for this momoent, if need corr, need to initialize in the initialize script
            # dataAve[iavg] = slitposcorr(dataAve[iavg], prefs, j, kind, iavg)
            dataArray[i], _, _, aveint = Transform.preprocess(dataArray[i], prefs, i)
            if prefs['vis']['intMonitor']:
                monitor[iproj][i]=array('d', [aveint, prefs[kind]['Num'][i][iproj], nOut])

        dataResutArray, sminx, smaxx = Transform.stitching(dataArray, prefs)
        finalData[iproj] = dataResutArray

    return 1


def projTransform(prefs, nOutstart, sType, kind, fileID, moncnt):

    prefs = Transform.stitching_calcbefore(prefs)

    maxx = 0; minx = sys.maxint
    sArrayX = 0; sArrayY = 0

    iproj = 0
    dataArray = []
    for i in prefs['series']:
        i = i -1
        filename = "%s/%s%s.%s" % ( prefs['filePath'], prefs[kind]['Prefix'][0], Transform.paddingZero(int(prefs[kind]['Num'][i][iproj]), prefs['numberDigit'] ), prefs[kind]['extension'] )
        # filename = "b_image.tif" # this tif image is for testing purpose
        im = Image.open(filename)
        imarray = np.array(im)
        dataArray.append(0)
        dataArray[i] = imarray[(prefs['ROIy'][i][0]-1):prefs['ROIy'][i][-1] , (prefs['ROIx'][i][0]-1):prefs['ROIx'][i][-1]] # corp the array
        # if prefs['slits']['shouldCorr']: #comment out for this momoent, if need corr, need to initialize in the initialize script
        # dataAve[iavg] = slitposcorr(dataAve[iavg], prefs, j, kind, iavg)
        dataArray[i], _, _, aveint = Transform.preprocess(dataArray[i], prefs, i)
    dataResutArray, sminx, smaxx = Transform.stitching(dataArray, prefs)


    totalimage = len(prefs[kind]['Num'][0])
    # totalimage = 2
    step = 100
    workjob = []
    procs = totalimage/step + 1


    # finalData = np.zeros(shape=( len(prefs[kind]['Num'][0]), dataResutArray.shape[0], dataResutArray.shape[1] ))

    sharedArray = Array(ctypes.c_double, len(prefs[kind]['Num'][0])*dataResutArray.shape[0]*dataResutArray.shape[1], lock=False)
    sharedNpArray = np.frombuffer(sharedArray, dtype=ctypes.c_double)
    finalData = sharedNpArray.reshape(len(prefs[kind]['Num'][0]), dataResutArray.shape[0], dataResutArray.shape[1])

    sharedMonitor = Array(ctypes.c_double, totalimage*len(prefs['series'])*3, lock=False)
    sharedNpMonitor = np.frombuffer(sharedMonitor, dtype=ctypes.c_double)
    monitor = sharedNpMonitor.reshape(totalimage, len(prefs['series']), 3)

    for i in range(0, procs):
        process = Process(target=child, args = (i, step, totalimage, prefs, kind, finalData, monitor, nOutstart))
        workjob.append(process)

    for j in workjob:
        j.start()

    for j in workjob:
        j.join()

    # for iproj in range(0,len(prefs[kind]['Num'][0])):
    #     nOut = nOutstart+iproj
    #     # outfilename = "%s/%s_%s.hdf" % (prefs['out']['filePath'], prefs['projection']['Prefix'], Transform.paddingZero(nOut, int(prefs['out']['numberDigit'])))
    #     dataArray = []
    #
    #     for i in prefs['series']:
    #         i = i -1
    #         filename = "%s/%s%s.%s" % ( prefs['filePath'], prefs[kind]['Prefix'][0], Transform.paddingZero(int(prefs[kind]['Num'][i][iproj]), prefs['numberDigit'] ), prefs[kind]['extension'] )
    #         # filename = "b_image.tif" # this tif image is for testing purpose
    #         im = Image.open(filename)
    #         imarray = np.array(im)
    #         dataArray.append(0)
    #         dataArray[i] = imarray[(prefs['ROIy'][i][0]-1):prefs['ROIy'][i][-1] , (prefs['ROIx'][i][0]-1):prefs['ROIx'][i][-1]] # corp the array
    #         # if prefs['slits']['shouldCorr']: #comment out for this momoent, if need corr, need to initialize in the initialize script
    #         # dataAve[iavg] = slitposcorr(dataAve[iavg], prefs, j, kind, iavg)
    #         dataArray[i], _, _, aveint = Transform.preprocess(dataArray[i], prefs, i)
    #         if prefs['vis']['intMonitor']:
    #             monitor.append(array('d', [aveint, prefs[kind]['Num'][i][iproj], nOut]))
    #
    #     dataResutArray, sminx, smaxx = Transform.stitching(dataArray, prefs)
    #
    #     if iproj == 0:
    #         finalData = np.zeros(shape=( len(prefs[kind]['Num'][0]), dataResutArray.shape[0], dataResutArray.shape[1] ))
    #
    #     finalData[iproj] = dataResutArray




        # d = array('i', finalData)
    # minx = min(minx, sminx)
    # maxx = max(maxx, smaxx)
    #
    # sArrayX = max(sArrayX, dataResutArray.shape[1])
    # sArrayY = max(sArrayY, dataResutArray.shape[0])

        # write the data array to the file


    prefs = Transform.stitching_calcafter(prefs, minx, maxx, sArrayX, sArrayY)

    moncnt = moncnt + len(prefs[kind]['Num'][0])
    nOutarg = nOutstart+len(prefs[kind]['Num'][0])

    return prefs, nOutarg, monitor, moncnt, finalData


# prefs = dict()
# prefs = Stitching_InitConv_TIFFtoHDF4_single.initPyPref(prefs)
# fileID = '/local/kyue/test/txt'
# nOut = 0; moncnt = 0;
# projTransform(prefs, nOut, 'WHITE1', 'beamProfileInitial', fileID, moncnt);