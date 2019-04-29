import numpy as np
import math
import time
import datetime
from array import *


def initPyPref(prefs):

    prefs['series']= array('i', [1])


    prefs['sN'] = len(prefs['series'])

    prefs['numberDigit'] = 6

    prefs['NumWF1'] = 10;  # Number of frames for the initial beam profile (white field)
    prefs['NumProj'] = 901; # Number of frames for the projections
    prefs['NumWF2'] = 10;  # Number of frames for the initial beam profile (white field)
    prefs['NumDarks'] = 10; # Number of frames for the dark field

    prefs['frameNum']  = prefs['NumWF1'] + prefs['NumProj'] + prefs['NumWF2'] + prefs['NumDarks']; # Image numbers in one part of the stitching
    prefs['VolShift']  = 0; # Shift of the image numbers to the next volume (default 0)
    # %prefs.VolShift = prefs.sN *prefs.frameNum; % One volume shift in image numbers

    prefs['RotationAngle'] = 0.0; # default 0, Rotation axis tilt correction in degrees, positive: rotating the image CCW, i.e. rotating the axis CW
    prefs['rotAxis'] = [1024];  # default
    #aka prefs.rotAxis = -prefs.translx+1024;
        # The position of the rotation axis on the full detector images measured during the alignment
        # Warning: This is the coulumn number starting
       # from 1, during alignment this index usually
        # goes from 0
    prefs['rotax'] = {}
    prefs['rotax']['fittedshift']=0 # The shift determined from the FindRotax script. This will be used for the reconstruction as a starter value
    prefs['SkipInitialBeamProfile'] = False;
    prefs['doVis'] = {}
    prefs['doVis']['projection'] = False;
    prefs['doVis']['beamProfileInitial'] = True;
    prefs['doVis']['beamProfileFinal'] = True;
    prefs['doVis']['dark'] = True;

    prefs['vis'] = {}
    prefs['vis']['projection'] = {}
    prefs['vis']['projection']['imgRange'] = array('d', [000, 4100])

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
    prefs['vis']['stitching']['ColorCodeLim'] =array('d', [100, 2000])
    prefs['vis']['stitching']['init'] = True
    prefs['vis']['stitching']['ViewTransm'] = True
    prefs['stitchproj'] = 1

    # prefs['ROIx'] = range(1,2049) # different between matlab and python
    # prefs['ROIy'] = range(1,2049)

    prefs['userID'] = 'sangid_nov15'
    # prefs['filePath'] = "/home/beams12/S1IDUSER/mnt/s1c/%s/tomo" % prefs['userID']

    prefs['filePath'] = "/data2/PeterData/testsangid_nov15"

    # prefs['projection'] = {}; prefs['projection']['Prefix'] = "J5_In718_Load01_";prefs['firstfile'] = 823;prefs['loadstep'] =  "";prefs['rotax']['fittedshift']=11
    # prefs['projection'] = {}; prefs['projection']['Prefix'] = "J5_In718_Load02_"; prefs['firstfile'] = 1639; prefs['loadstep'] =  ""; prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {}; prefs['projection']['Prefix'] = "J5_In718_Load05_"; prefs['firstfile'] = 4143; prefs['loadstep'] =  ""; prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {}; prefs['projection']['Prefix'] = "J5_In718_Load03_"; prefs['firstfile'] = 2510; prefs['loadstep'] =  ""; prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {}; prefs['projection']['Prefix'] = "J5_In718_Load04_"; prefs['firstfile'] = 3327; prefs['loadstep'] =  ""; prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {}; prefs['projection']['Prefix'] = "J5_In718_Load06_";prefs['firstfile'] = 6592;prefs['loadstep'] =  ""; prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {};prefs['projection']['Prefix'] = "J5_In718_Load05_top_";prefs['firstfile'] = 4960;prefs['loadstep'] =  "_a"; prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {}; prefs['projection']['Prefix'] = "J5_In718_Load05_top_"; prefs['firstfile'] = 5776; prefs['loadstep'] =  "_b";prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {}; prefs['projection']['Prefix'] = "J5_In718_Load08_"; prefs['firstfile'] = 8615; prefs['loadstep'] =  ""; prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {}; prefs['projection']['Prefix'] = "J5_In718_Load09_"; prefs['firstfile'] = 9431; prefs['loadstep'] =  "";prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {};prefs['projection']['Prefix'] = "J5_In718_Load10_"; prefs['firstfile'] = 10247; prefs['loadstep'] =  "";prefs['rotax']['fittedshift']=10
    prefs['projection'] = {};prefs['projection']['Prefix'] = ["J5_In718_Load11_"];prefs['firstfile'] = 11063; prefs['loadstep'] =  ""; prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {}; prefs['projection']['Prefix'] = "J5_In718_Load12_"; prefs['firstfile'] = 11879;prefs['loadstep'] =  "";  prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {}; prefs['projection']['Prefix'] = "J5_In718_Load14_"; prefs['firstfile'] = 12766; prefs['loadstep'] =  ""; prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {};   prefs['projection']['Prefix'] = "E5_In718_PP_";   prefs['firstfile'] = 14399;  prefs['loadstep'] =  "";  prefs['rotax']['fittedshift']=10
    # prefs['projection'] = {};   prefs['projection']['Prefix'] = ["5D_In718_SR_"]; prefs['firstfile'] = 15215;  prefs['loadstep'] =  ""; prefs['rotax']['fittedshift']=10

    prefs['numberDigit'] = 6

    defaultfnames(prefs)

    prefs['VolShift'] = 0; prefs['NumWF1'] = 10; prefs['NumProj'] = 786; prefs['NumWF2'] = 10; prefs['NumDarks'] = 10

    GenImageNums(prefs, prefs['firstfile'], 1, -1)

    prefs['out'] = {}
    prefs['out']['filePrefix'] = "p11p168s0.2_tomo"
    prefs['out']['dirname'] = "%s_rec/%s%s/raw" % ( prefs['userID'], prefs['projection']['Prefix'][0],prefs['loadstep'] )
    prefs['out']['filePath'] = "/home/beams12/S1IDUSER/mnt/orthros/%s" % prefs['out']['dirname']

    prefs['ROIx'] = [ array('i', range(180,1901)) ]  # different between matlab and python change from d to i
    prefs['ROIy'] = [ array('i', range(700,1751)) ] # change d to i for easy manipulation
    prefs['shouldRotate'] = True
    prefs['RotationAngle'] = [0.0]
    prefs['shouldCrop'] = True
    prefs['cropROIx'] = [ array('i', range(9,9+1700)) ]
    prefs['cropROIy'] = [ array('i', range(11,11+1024)) ]

    prefs['doVis']['projection'] = False
    prefs['SkipInitialBeamProfile'] = False
    prefs['rotAxis'] = [1020]
    prefs['vis']['projection']['imgRange'] = array('d', [300, 4000])


    prefs['slits']={}
    prefs['slits']['shouldCorr']= False
    prefs['slits']['usePrevious'] = False
    prefs['slits']['slitsnum'] =  array('d', [1,2,3,4])
    prefs['slits']['tlx'] = 167; prefs['slits']['tly'] = 777
    prefs['slits']['brx'] = 1917; prefs['slits']['bry'] = 1681
    prefs['slits']['ROISizex'] = 150; prefs['slits']['ROISizey'] = 150

    prefs['slits']['ROIx'] = [array('d', range(int(prefs['slits']['tlx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['tlx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                                array('d', range(int(prefs['slits']['brx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['brx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                                array('d', range(int(prefs['slits']['tlx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['tlx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ,\
                                array('d', range(int(prefs['slits']['brx']-math.floor(prefs['slits']['ROISizex']/2.)),int(prefs['slits']['brx']+math.ceil(prefs['slits']['ROISizex']/2.)))) ]
    prefs['slits']['ROIy'] = [array('d', range(int(prefs['slits']['tly']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['tly']+math.ceil(prefs['slits']['ROISizey']/2.))) ) ,\
                              array('d', range(int(prefs['slits']['tly']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['tly']+math.ceil(prefs['slits']['ROISizey']/2.)))) ,\
                              array('d', range(int(prefs['slits']['bry']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['bry']+math.ceil(prefs['slits']['ROISizey']/2.))))  ,\
                              array('d', range(int(prefs['slits']['bry']-math.floor(prefs['slits']['ROISizey']/2.)),int(prefs['slits']['bry']+math.ceil(prefs['slits']['ROISizey']/2.)))) ]

    prefs['slits']['tempFrameSizex'] = 20;prefs['slits']['tempFrameSizey'] = 20;

    prefs['slits']['refkind'] = "projection"
    prefs['slits']['referenceNum'] = [array('d', [1])]
    prefs['slits']['doVisROIs'] = False

    prefs['slitposcheck'] = {}
    prefs['slitposcheck']['projection'] = {}
    prefs['slitposcheck']['projection']['Num'] = []

    prefs['slitposcheck']['projection']['Num'].append( prefs['projection']['Num'][0][0::int(math.ceil((len(prefs['projection']['Num'][0])-1)/72.))] )

    prefs['correl'] = {}
    prefs['correl']['VertCorr'] = False
    prefs['correl']['dark'] = min(5,len(prefs['dark']['Num']))
    prefs['correl']['white'] = min(5,len(prefs['beamProfileInitial']['Num']))
    prefs['correl']['first'] = 1
    prefs['correl']['last'] = prefs['correl']['first']+len(prefs['projection']['Num'])-1

    prefs['correl']['smallROIy'] =  array('d', range(90,951) )
    prefs['correl']['smallROIx'] = array('d', range(206, 1517))
    prefs['correl']['bigROIx'] = array('d', range(100, 1601))
    prefs['correl']['rowstep'] = 4
    prefs['correl']['bandwidth'] = 2

    prefs['shouldColumnAverage'] = False
    prefs['translx'] = [0]
    prefs['transly'] = [0]
    prefs['bboxx'] = 3048
    prefs['bboxy'] = 3048

    prefs['shouldRotateAfterStitch'] = False
    prefs['RotationAngleAfterStitch'] = -0.37
    prefs['shouldCropAfterStitch'] = False
    prefs['CropROIxAfterStitch'] = array('d', range(10, 7052))
    prefs['CropROIyAfterStitch'] = array('d', range(24, 489))

    prefs['out']['numberDigit'] = 5
    prefs['out']['fileType'] = "uint16"
    prefs['out']['VertSizeWarningAt'] = 128
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
    prefs['slits']['slitpos']['x'] = []
    prefs['slits']['slitpos']['y'] = []
    prefs['slits']['slitpos']['Num'] = []
    prefs['slits']['slitpos']['ncc'] = []
    prefs['slits']['slitfname'] = "%s/%slastslitsfield.mat" % (prefs['out']['filePath'], prefs['projection']['Prefix'])

    if False:
        prefs['ROIx'] = [ array('d', range(1,2049)) ]  # different between matlab and python
        prefs['ROIy'] = [ array('d', range(1,2049)) ]
        prefs['doVis']['projection'] = True

        prefs['shouldRotate'] = False
        prefs['RotationAngle'] = [0.0]
        prefs['shouldCrop'] = False
        prefs['cropROIx'] = [ array('d', range(1,2049)) ]
        prefs['cropROIy'] = [ array('d', range(100,2001,400)) ]
        print("Number of layers for the quick reconstruction: %d", len(prefs['cropROIy']))


    if False:
        prefs['ROIx'] = [ array('d', range(1,2049)) ]# different between matlab and python
        prefs['ROIy'] = [ array('d', range(1,2049)) ]
        prefs['doVis']['projection'] = True

        prefs['RotationAngle'] = [-0.5729]
        prefs['RotationAngle'] = [-1.187]
        prefs['shouldCrop'] = True
        prefs['cropROIx'] = [ array('d', range(1,2049)) ]
        prefs['cropROIy'] = [ array('d', range(1,2049)) ]

    return prefs

def defaultfnames(prefs):
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
    return prefs

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




