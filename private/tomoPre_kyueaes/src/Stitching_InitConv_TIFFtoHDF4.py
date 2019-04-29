# Author: Ke Yue
# Time 2015 Dec 15

import numpy as np
import math
import time
import datetime
from array import *


def initPyPref(prefs):
    # prefs['series']= [1]

    prefs['series']= array('d', [1, 2, 3, 4])

    prefs['sN'] = len(prefs['series'])

    prefs['filePath'] = "/home/beams12/S1IDUSER/mnt/s1c/%s/tomo"

    prefs['numberDigit'] = 6

    # prefs.projection.Prefix = {'Bar1_LLZO_LGS_D5_', 'Bar1_LLZO_LGS_D5_', 'Bar1_LLZO_LGS_D5_'}; % Write '_' to the end, if any
    # prefs.projection.Prefix = {'Bar1_LLZO_LGS_D130_', 'Bar1_LLZO_LGS_D130_'}; % Write '_' to the end, if any
    # prefs.projection.Prefix = {'Disc1_LLZO_LGF_D255_', 'Disc1_LLZO_LGF_D255_','Disc1_LLZO_LGF_D255_', 'Disc1_LLZO_LGF_D255_'}; % Write '_' to the end, if any

    prefs['projection'] = {};   prefs['projection']['Prefix'] = ['Disc2_LLZO_SGF_D255_', 'Disc2_LLZO_SGF_D255_','Disc2_LLZO_SGF_D255_', 'Disc2_LLZO_SGF_D255_']
    prefs['projection']['extension'] = "tif"


    prefs['beamProfileInitial'] = {}
    prefs['beamProfileInitial']['Prefix'] = prefs['projection']['Prefix']
    prefs['beamProfileInitial']['extension'] = prefs['projection']['extension']
    prefs['beamProfileFinal'] = {}
    prefs['beamProfileFinal']['Prefix'] = prefs['projection']['Prefix']
    prefs['beamProfileFinal']['extension'] = prefs['projection']['extension']
    prefs['dark']= {}
    prefs['dark']['Prefix'] = prefs['projection']['Prefix']
    prefs['dark']['extension'] = prefs['projection']['extension']


    prefs['VolShift']  = 0; # Shift of the image numbers to the next volume (default 0)
    prefs['NumWF1'] = 10;  # Number of frames for the initial beam profile (white field)
    prefs['NumProj'] = 1801; # Number of frames for the projections
    prefs['NumWF2'] = 10;  # Number of frames for the initial beam profile (white field)
    prefs['NumDarks'] = 10; # Number of frames for the dark field

    prefs['SkipInitialBeamProfile'] = False;
    prefs['doVis'] = {}
    prefs['doVis']['projection'] = False;
    prefs['doVis']['beamProfileInitial'] = True;
    prefs['doVis']['beamProfileFinal'] = True;
    prefs['doVis']['dark'] = True;

    prefs['vis'] = {}
    prefs['vis']['projection'] = {}
    prefs['vis']['projection']['imgRange'] = array('d', [200, 3000])

    prefs['vis']['beamProfileInitial'] = {}
    prefs['vis']['beamProfileInitial']['imgRange'] = array('d', [0,4096])

    prefs['vis']['beamProfileFinal'] = {}
    prefs['vis']['beamProfileFinal']['imgRange'] = array('d', [0,4096])

    prefs['vis']['dark'] = {}
    # prefs['vis']['dark']['imgRange'] = array('d', [95, 100])
    prefs['vis']['dark']['imgRange'] = array('d', [0, 20])

    prefs['vis']['intMonitor'] = True

    prefs['doVis']['stitching'] = False;


    prefs['vis']['stitching'] = {}
    prefs['vis']['stitching']['ColorCodeLim'] =array('d', [10, 2000])
    prefs['vis']['stitching']['init'] = True
    prefs['vis']['stitching']['ViewTransm'] = False
    prefs['stitchproj'] = 1

    prefs['ROIx'] = [ array('d', range(1,2049)), array('d', range(1,2049)), array('d', range(1,2049)), array('d', range(1,2049)) ]  # different between matlab and python
    prefs['ROIy'] = [ array('d', range(1,2049)), array('d', range(1,2049)), array('d', range(1,2049)), array('d', range(1,2049)) ]
    prefs['shouldRotate'] = False
    prefs['RotationAngle'] = [0, 0, 0]
    prefs['shouldCrop'] = False
    prefs['cropROIx'] = [ array('d', range(1,2049)), array('d', range(1,2049)), array('d', range(1,2049)), array('d', range(1,2049)) ]
    prefs['cropROIy'] = [ array('d', range(1,2049)), array('d', range(1,2049)), array('d', range(1,2049)), array('d', range(1,2049)) ]


    prefs['shouldColumnAverage'] = False
    prefs['translx'] = [-1700, 0, 1700]
    prefs['transly'] = [0, 0, 0]
    prefs['bboxx'] = 7500
    prefs['bboxy'] = 3200

    prefs['shouldRotateAfterStitch'] = False
    prefs['RotationAngleAfterStitch'] = 0.0
    prefs['shouldCropAfterStitch'] = False
    prefs['CropROIxAfterStitch'] = array('d', range(70, 5341))
    prefs['CropROIyAfterStitch'] = array('d', range(50, (50+1792)))

    prefs['rotAxis'] = [1026+1700, 1026, 1026-1700]

    prefs['out'] = {}
    prefs['out']['filePath'] = "O:/ForgetToGiveAPath/raw"
    prefs['out']['filePrefix'] = "mp90s0.2_tomo"
    prefs['out']['numberDigit'] = 5
    prefs['out']['fileType'] = "uint16"
    prefs['out']['VertSizeWarningAt'] = 128

    prefs['VolShift'] = 0; prefs['NumWF1'] = 10; prefs['NumProj'] = 1801; prefs['NumWF2'] = 10; prefs['NumDarks'] = 10
    # prefs = GenImageNums(prefs, 23959, 1, 1); % Disc1
    prefs = GenImageNums(prefs, 31283, 1, 1)

    prefs['shouldRotate'] = True
    prefs['RotationAngle'] = [180, 180, 180, 180]
    prefs['shouldCrop'] = True
    prefs['cropROIx'] = [ array('d', range(1,2049)), array('d', range(60,2049)), array('d', range(1,2049)), array('d', range(1,2049)) ]
    prefs['cropROIy'] = [ array('d', range(1,2049)), array('d', range(1,2049)), array('d', range(1,2049)), array('d', range(1,2049)) ]

    prefs['shouldColumnAverage'] = False
    prefs['translx'] = [-2939, -975, 975, 2975]
    prefs['transly'] = [9, 0, 0, 0]
    prefs['bboxx'] = 9000
    prefs['bboxy'] = 6500
    prefs['shouldRotateAfterStitch'] = False
    prefs['RotationAngleAfterStitch'] = 0.02
    prefs['shouldCropAfterStitch'] = False
    prefs['CropROIxAfterStitch'] = array('d', range(1, 3694))
    prefs['CropROIyAfterStitch'] = array('d', range(2, (2+1280)))

# prefs.out.filePath = '/home/beams12/S1IDUSER/mnt/orthros/Faber_Feb14_rec/Disc1_LLZO_LGF_D255/raw';
# prefs.out.filePath = '/home/beams12/S1IDUSER/mnt/orthros/Faber_Feb14_rec/Disc2_LLZO_SGF_D255/raw';
    prefs['out']['filePath'] = "/home/beams12/S1IDUSER/mnt/orthros/Faber_Feb14_rec/Disc2_LLZO_SGF_D255/raw"
    prefs['out']['filePrefix'] = "mp90s0.2_tomo"
    prefs['out']['numberDigit'] = 5
    prefs['out']['fileType'] = "uint16"
    prefs['out']['VertSizeWarningAt'] = 128

    prefs['doVis']['projection'] = False
    prefs['vis']['stitching']['ViewTransm'] = True
    prefs['stitchproj'] = 1
    prefs['vis']['stitching']['ColorCodeLim'] =array('d', [0, 4000])
    prefs['rotAxis'] = [4000, 2000, 50, -2050]

    prefs['slits']={}
    prefs['slits']['shouldCorr']= False
    # prefs['slits']['usePrevious'] = False
    prefs['slits']['slitsnum'] =  array('d', [3,4])
    prefs['slits']['tlx'] = 11+50+40; prefs['slits']['tly'] = 10
    prefs['slits']['brx'] = 2039-50-40; prefs['slits']['bry'] = 1270
    prefs['slits']['ROISizex'] = 20+100; prefs['slits']['ROISizey'] = 300

    prefs['slits']['ROIx'] = [array('d', range(int(prefs['slits']['tlx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['tlx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                              array('d', range(int(prefs['slits']['tlx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['tlx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                              array('d', range(int(prefs['slits']['tlx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['tlx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                              array('d', range(int(prefs['slits']['tlx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['tlx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                                array('d', range(int(prefs['slits']['brx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['brx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                              array('d', range(int(prefs['slits']['brx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['brx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                              array('d', range(int(prefs['slits']['brx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['brx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                              array('d', range(int(prefs['slits']['brx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['brx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                                array('d', range(int(prefs['slits']['tlx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['tlx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                              array('d', range(int(prefs['slits']['tlx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['tlx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                              array('d', range(int(prefs['slits']['tlx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['tlx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                              array('d', range(int(prefs['slits']['tlx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['tlx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                                array('d', range(int(prefs['slits']['brx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['brx']+math.ceil(prefs['slits']['ROISizex']/2.)))), \
                              array('d', range(int(prefs['slits']['brx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['brx']+math.ceil(prefs['slits']['ROISizex']/2.)))), \
                              array('d', range(int(prefs['slits']['brx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['brx']+math.ceil(prefs['slits']['ROISizex']/2.)))), \
                              array('d', range(int(prefs['slits']['brx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['brx']+math.ceil(prefs['slits']['ROISizex']/2.))))]

    prefs['slits']['ROIy'] = [array('d', range(int(prefs['slits']['tly']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['tly']+math.ceil(prefs['slits']['ROISizey']/2.))) ) ,\
                              array('d', range(int(prefs['slits']['tly']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['tly']+math.ceil(prefs['slits']['ROISizey']/2.))) ) ,\
                              array('d', range(int(prefs['slits']['tly']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['tly']+math.ceil(prefs['slits']['ROISizey']/2.))) ) ,\
                              array('d', range(int(prefs['slits']['tly']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['tly']+math.ceil(prefs['slits']['ROISizey']/2.))) ) ,\
                               array('d', range(int(prefs['slits']['tly']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['tly']+math.ceil(prefs['slits']['ROISizey']/2.)))) ,\
                              array('d', range(int(prefs['slits']['tly']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['tly']+math.ceil(prefs['slits']['ROISizey']/2.)))) ,\
                              array('d', range(int(prefs['slits']['tly']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['tly']+math.ceil(prefs['slits']['ROISizey']/2.)))) ,\
                              array('d', range(int(prefs['slits']['tly']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['tly']+math.ceil(prefs['slits']['ROISizey']/2.)))) ,\
                               array('d', range(int(prefs['slits']['bry']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['bry']+math.ceil(prefs['slits']['ROISizey']/2.))))  ,\
                              array('d', range(int(prefs['slits']['bry']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['bry']+math.ceil(prefs['slits']['ROISizey']/2.))))  ,\
                              array('d', range(int(prefs['slits']['bry']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['bry']+math.ceil(prefs['slits']['ROISizey']/2.))))  ,\
                              array('d', range(int(prefs['slits']['bry']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['bry']+math.ceil(prefs['slits']['ROISizey']/2.))))  ,\
                               array('d', range(int(prefs['slits']['bry']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['bry']+math.ceil(prefs['slits']['ROISizey']/2.)))) , \
                              array('d', range(int(prefs['slits']['bry']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['bry']+math.ceil(prefs['slits']['ROISizey']/2.)))), \
                              array('d', range(int(prefs['slits']['bry']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['bry']+math.ceil(prefs['slits']['ROISizey']/2.)))), \
                              array('d', range(int(prefs['slits']['bry']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['bry']+math.ceil(prefs['slits']['ROISizey']/2.))))]

    prefs['slits']['tempFrameSizex'] = 7;prefs['slits']['tempFrameSizey'] = 60;

    prefs['slits']['refkind'] = "projection"
    prefs['slits']['referenceNum'] = [array('d', [1, 1, 1, 1])]
    prefs['slits']['doVisROIs'] = True

    prefs['slits']['vis'] = {}
    prefs['slits']['vis']['ColorCodeLim'] = array('d', [0, 500])

    prefs['slitposcheck'] = {}
    prefs['slitposcheck']['projection'] = {}
    prefs['slitposcheck']['projection']['Num'] = []

    prefs['slitposcheck']['projection']['Num'].append( prefs['projection']['Num'][0][0:1800:50])
    prefs['slitposcheck']['projection']['Num'].append( prefs['projection']['Num'][1][0:1800:50])
    prefs['slitposcheck']['projection']['Num'].append( prefs['projection']['Num'][2][0:1800:50])
    prefs['slitposcheck']['projection']['Num'].append( prefs['projection']['Num'][3][0:1800:50])

    prefs['slits']['usePrevious'] = True

    prefs['template'] = [ array('d', [2545, 3172, 15, 50]), array('d', [2545, 3450, 15, 50]), array('d', [2545, 3850, 15, 50]), array('d', [2545, 4100, 15, 50])]
    prefs['searchrange'] = [ array('d', [25, 100]), array('d', [25, 100]), array('d', [25, 100]), array('d', [25, 100])]

    prefs['out']['ROIx'] = array('d', [])
    prefs['out']['ROIy'] = array('d', [])
    prefs['out']['rotAxis'] = []
    prefs['out']['shift'] = []
    prefs['out']['xposInBBox'] = []
    prefs['out']['yposInBBox'] = []

    ts= time.time()
    prefs['out']['timestamp'] = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    prefs['out']['logFile'] = "../ConvToHDF_%s.log" % prefs['out']['timestamp']

    prefs['slits']['slitpos']={}
    prefs['slits']['slitpos']['x'] = [max(prefs['slits']['slitsnum'], prefs['sN'])]
    prefs['slits']['slitpos']['y'] = [max(prefs['slits']['slitsnum'], prefs['sN'])]
    prefs['slits']['slitpos']['Num'] = [prefs['sN']]
    prefs['slits']['slitpos']['ncc'] = [prefs['sN']]
    prefs['slits']['slitfname'] = "%s/%slastslitsfield.mat" % (prefs['out']['filePath'], prefs['projection']['Prefix'])

    return prefs

# def defaultfnames(prefs):
#     prefs['projection']['extension'] = "tif"
#     prefs['beamProfileInitial'] = {}
#     prefs['beamProfileInitial']['Prefix'] = prefs['projection']['Prefix']
#     prefs['beamProfileInitial']['extension'] = prefs['projection']['extension']
#     prefs['beamProfileFinal'] = {}
#     prefs['beamProfileFinal']['Prefix'] = prefs['projection']['Prefix']
#     prefs['beamProfileFinal']['extension'] = prefs['projection']['extension']
#     prefs['dark']= {}
#     prefs['dark']['Prefix'] = prefs['projection']['Prefix']
#     prefs['dark']['extension'] = prefs['projection']['extension']
#     return prefs

def GenImageNums(prefs, start, FOVstep, imgstep):
    s0 = start
    if imgstep == 0:
        print("ERROR: The imgstep cannot be zero!\n")
        return prefs
    if imgstep != int(imgstep):
        print("ERROR: The imgstep cannot be zero!\n")
        return prefs
    if FOVstep >= 1:
        series = range(1,prefs['sN']+1, FOVstep)
    elif FOVstep <= -1:
        series = range(prefs['sN'],1+1, FOVstep)
    else:
        print("Warning: Not proper parameter for FOVstep")

    prefs['beamProfileInitial']['Num']=[]
    prefs['projection']['Num'] = []
    prefs['beamProfileFinal']['Num'] = []
    prefs['dark']['Num'] = []

    for i in series:
        prefs['beamProfileInitial']['Num'].append(array('d', [x+prefs['VolShift'] for x in range(s0,(s0+prefs['NumWF1']))]))
        # print("list", prefs['beamProfileInitial']['Num'])
        s0 = s0 + prefs['NumWF1']
        if imgstep >= 1:
            imglist = range(s0,(s0+prefs['NumProj']), imgstep)
        elif imgstep <= 1:
            imglist = range((s0+prefs['NumProj']-1), s0-1, imgstep)
        prefs['projection']['Num'].append( array('d', [x+prefs['VolShift'] for x in imglist] ) )
        s0 = s0 + prefs['NumProj']
        prefs['beamProfileFinal']['Num'].append( array('d', [x+prefs['VolShift'] for x in range(s0,(s0+prefs['NumWF2']))]) )
        s0 = s0 + prefs['NumWF2']
        prefs['dark']['Num'].append( array('d', [x+prefs['VolShift'] for x in range(s0,(s0+prefs['NumDarks']))]) )
        s0 = s0 + prefs['NumDarks']

        if imgstep <= 1:
            temp = prefs['beamProfileInitial']['Num'][-1]
            prefs['beamProfileInitial']['Num'][-1] = prefs['beamProfileFinal']['Num'][-1]
            prefs['beamProfileFinal']['Num'][-1] = temp

    prefs['frameNum']  = prefs['NumWF1'] + prefs['NumProj'] + prefs['NumWF2'] + prefs['NumDarks']
    return  prefs



# Prefs = {}
# initPyPref(Prefs)




