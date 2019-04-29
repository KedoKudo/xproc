%% Preferences for converting tiffs to hdf
% convention: image #1 is direct beam; take n tomographic images; 
%     image #(n+2) is the direct beam again; image #(n+3) is the dark beam
% Presently, the images must cover the interval of 0-180 degrees
%
% Written by P. Kenesei, APS

% DO NOT PUT THEM BACK! It will cause error.
% They will be running at the Stitching_Helper or at the Conversion.
%  format compact;
%  close all hidden;
%  fclose('all');
%  clear all;

% Abbreviation for 'true' and 'false'
T=true; F=false;

% Numbers of series made for stitching 
%prefs.series = [1 3 2]; % Here you can provide the Z order of the series
                % [1 2 3] means 3rd overwrites 2nd which overwrites 1st, 
                % series defined by the order in the cell arrays below.
                % This is effective mostly at the stitching stage.
prefs.series = [1 2 3 4];
%prefs.series = [2 1];
prefs.sN = length(prefs.series);
% INPUT files (Note: use '/' for paths)
prefs.filePath = '/home/beams12/S1IDUSER/mnt/orthros/Faber_Feb14/Tomo';
%prefs.filePath = 'O:/Faber_Feb14/Tomo';
prefs.numberDigit = 6;
%prefs.projection.Prefix = {'XROa4_', 'XROa4_', 'XROa4_'}; % Write '_' to the end, if any
prefs.projection.Prefix = {'Bar1_LLZO_LGS_D5_', 'Bar1_LLZO_LGS_D5_', 'Bar1_LLZO_LGS_D5_'}; % Write '_' to the end, if any
prefs.projection.Prefix = {'Bar1_LLZO_LGS_D130_', 'Bar1_LLZO_LGS_D130_'}; % Write '_' to the end, if any
prefs.projection.Prefix = {'Disc1_LLZO_LGF_D255_', 'Disc1_LLZO_LGF_D255_','Disc1_LLZO_LGF_D255_', 'Disc1_LLZO_LGF_D255_'}; % Write '_' to the end, if any
prefs.projection.Prefix = {'Disc2_LLZO_SGF_D255_', 'Disc2_LLZO_SGF_D255_','Disc2_LLZO_SGF_D255_', 'Disc2_LLZO_SGF_D255_'}; % Write '_' to the end, if any
prefs.projection.extension = 'tif'; % No dot at the beginning
prefs.beamProfileInitial.Prefix = prefs.projection.Prefix; % WHITE1
prefs.beamProfileInitial.extension = prefs.projection.extension;
prefs.beamProfileFinal.Prefix = prefs.projection.Prefix; % WHITE2
prefs.beamProfileFinal.extension = prefs.projection.extension;
prefs.dark.Prefix = prefs.projection.Prefix;  % DARK
prefs.dark.extension = prefs.projection.extension;
prefs.VolShift = 0; % Shift of the image numbers to the next volume (default 0)
prefs.NumWF1 = 10;  % Number of frames for the initial beam profile (white field)
prefs.NumProj = 1801; % Number of frames for the projections
prefs.NumWF2 = 10;  % Number of frames for the initial beam profile (white field)
prefs.NumDarks = 10; % Number of frames for the dark field

%prefs.beamProfileInitial.Num={prefs.beamProfileInitial.Num{1}(1), prefs.beamProfileInitial.Num{2}(1), prefs.beamProfileInitial.Num{3}(1)};

% Visualization options
prefs.SkipInitialBeamProfile = false; % Skip to generate WF1 for saving  time during test runs
prefs.doVis.projection = false;  % Should be done any visualization on the images
prefs.doVis.beamProfileInitial = true;
prefs.doVis.beamProfileFinal = true;
prefs.doVis.dark = true;
prefs.vis.projection.imgRange = [200 3000];
prefs.vis.beamProfileInitial.imgRange = [0 4096];
prefs.vis.beamProfileFinal.imgRange = [0 4096];
%prefs.vis.dark.imgRange = [95 100];
prefs.vis.dark.imgRange = [0 20];
prefs.vis.intMonitor = true; % Should we calculate the average intensities 
                             % on the preprocessed images to monitor the success of acquisition
prefs.doVis.stitching = false; % This must be simply false
prefs.vis.stitching.ColorCodeLim = [10 2000]; % Initial colorbar min/max value
prefs.vis.stitching.init = true; % It must be true for the first image initialization
prefs.vis.stitching.ViewTransm = F; % Should we show corrected (transmission) images, or the raw projections 
prefs.stitchproj = 1; % The projection which we will visually stitch at Stitching_Helper startup

% ROIs for reducing the data size and making some pre-processing
% Vertical size should be a 128x value: 128,256,384,512,640,768,896,1024,1152,1280
% 1408,1536,1664,1792,1920,2048,2176
% The part of the file (ROI) that will be read in
% prefs.ROIx = {1:2048, 1:2048, 1:2048};   % Horizontal size
% prefs.ROIy = {1:2048, 1:2048, 1:2048};   % Vertical size
prefs.ROIx = {1:2048, 1:2048, 1:2048, 1:2048};   % Horizontal size
prefs.ROIy = {1:2048, 1:2048, 1:2048, 1:2048};   % Vertical size
prefs.shouldRotate = F; % Rotation according to the detector rotation or rot. axis tilt
% Rotation angle (deg); direction according to right-hand-rule
prefs.RotationAngle = { 0,  0,  0}; % Kenesei_Nov11 Yb2Si2O7 stood D503mm
prefs.shouldCrop = F;   % Cropping within the ROI defined above
prefs.cropROIx = {1:2048, 1:2048, 1:2048, 1:2048};   % Horizontal cropping
prefs.cropROIy = {1:2048, 1:2048, 1:2048, 1:2048};   % Vertical cropping
prefs.shouldColumnAverage = F; % Averaging the columns to one row image

% Translations (pixels) for stitching
prefs.translx = {-1700, 0, 1700};  % Translations (pixels) for stitching (horizontal)
prefs.transly = {0, 0, 0};         % Vertical
prefs.bboxx = 7500; % Horizontal size of the bounding box, where the images can be stitched
prefs.bboxy = 3200; % Vertical

prefs.shouldRotateAfterStitch = F; % Should the image be rotated after the stitching
prefs.RotationAngleAfterStitch = 0.0;
prefs.shouldCropAfterStitch = F; % Should the image be cropped after the stitching and optional rotation
% These numbers are relative to the minimum bounding box of the stitched image
prefs.CropROIxAfterStitch = [70:5340]; % Kenesei_Nov11 Yb2Si2O7 stood D503mm  Horiz. The image region to be cropped after stitching
prefs.CropROIyAfterStitch = [50:(50+1792-1)]; % Vert.

prefs.rotAxis = {1026+1700, 1026, 1026-1700};
% aka prefs.rotAxis = -prefs.translx+1026.5; 
    % The position of the rotation axis on the full detector images measured during the alignment
    % Warning: This is the coulumn number starting
    % from 1, during alignment this index usually
    % goes from 0

% OUTPUT files (Note: use '/' for paths)
prefs.out.filePath = 'O:/ForgetToGiveAPath/raw';
% prefs.out.filePrefix = 'm153.1m26.1s0.2_tomo'; % No '_' at the end
%prefs.out.filePrefix = 'mp90s0.1_tomo'; % No '_' at the end
prefs.out.filePrefix = 'mp90s0.2_tomo'; % No '_' at the end
prefs.out.numberDigit = 5;
prefs.out.fileType = 'uint16';
prefs.out.VertSizeWarningAt = 128; % Warning messages will be sent out if the size is not a multiplier of this value. No warning if it is zero.



% % Faber_Jul13 Red Oak XROa4 86keV (3x1 stitching) shift -5
% prefs.VolShift = 0; prefs.NumWF1 = 10; prefs.NumProj = 701; prefs.NumWF2 = 10; prefs.NumDarks = 10; 
% prefs = GenImageNums(prefs, 16165, 1, 1); 
% prefs.ROIx = {1:2048, 1:2048, 1:2048};   %  Horizontal size
% prefs.ROIy = {500:1530, 500:1530, 500:1530};   % Vertical size
% prefs.shouldRotate = F; % Rotation according to the detector rotation or rot. axis tilt
% prefs.RotationAngle = { 0,  0,  0}; % Kenesei_Nov11 Yb2Si2O7 stood D503mm
% prefs.shouldCrop = T;   % Cropping within the ROI defined above
% prefs.cropROIx = {1:2048, 95:1855, 1:2048};   % Horizontal cropping
% prefs.cropROIy = {2:2+1024-1, 2:2+1024-1, 2:2+1024-1};   % Vertical cropping
% prefs.shouldColumnAverage = F; % Averaging the columns to one row image
% prefs.translx = {1607, 0, -1606};  % Translations (pixels) for stitching (horizontal)
% prefs.transly = {4, 0, -6};         % Vertical
% prefs.bboxx = 6200; % Horizontal size of the bounding box, where the images can be stitched
% prefs.bboxy = 6200; % Vertical
% prefs.shouldRotateAfterStitch = F; % Should the image be rotated after the stitching
% prefs.RotationAngleAfterStitch = 0.0;
% prefs.shouldCropAfterStitch = T; % Should the image be cropped after the stitching and optional rotation
% % These numbers are relative to the minimum bounding box of the stitched image
% prefs.CropROIxAfterStitch = [75:5211]; %The image region to be cropped after stitching
% prefs.CropROIyAfterStitch = [4:(4+1024-1)]; % Vert.
% prefs.out.filePath = 'O:/Faber_Jul13_rec/XROa4/raw';
% prefs.out.filePrefix = 'm27p113s0.2_tomo'; % No '_' at the end
% prefs.out.numberDigit = 5;
% prefs.out.fileType = 'uint16';
% prefs.out.VertSizeWarningAt = 128; % Warning messages will be sent out if the size is not a multiplier of this value. No warning if it is zero.
% prefs.doVis.projection = F;
% prefs.stitchproj = 132;
% prefs.vis.stitching.ColorCodeLim = [990 2900];

% % Faber_Feb14 Bar1_LLZO_LGS_D5 92keV (2x1 stitching) 7.5X, upside down, shift
% prefs.VolShift = 0; prefs.NumWF1 = 10; prefs.NumProj = 901; prefs.NumWF2 = 10; prefs.NumDarks = 10; 
% prefs = GenImageNums(prefs, 22097, 1, 1); 
% prefs.ROIx = {1:2048, 1:2048};   %  Horizontal size
% prefs.ROIy = {1:1300, 1:1300};   % Vertical size
% prefs.shouldRotate = T; % Rotation according to the detector rotation or rot. axis tilt
% prefs.RotationAngle = { 180, 180 }; 
% prefs.shouldCrop = T;   % Cropping within the ROI defined above
% prefs.cropROIx = {50:1827, 1:2000};   % Horizontal cropping
% prefs.cropROIy = {10:10+1280-1, 10:10+1280-1};   % Vertical cropping
% prefs.shouldColumnAverage = F; % Averaging the columns to one row image
% prefs.translx = {-875, 867};  % Translations (pixels) for stitching (horizontal)
% prefs.transly = {0, -3};         % Vertical
% prefs.bboxx = 5000; % Horizontal size of the bounding box, where the images can be stitched
% prefs.bboxy = 4500; % Vertical
% prefs.shouldRotateAfterStitch = F; % Should the image be rotated after the stitching
% prefs.RotationAngleAfterStitch = 0.0;
% prefs.shouldCropAfterStitch = T; % Should the image be cropped after the stitching and optional rotation
%                                  % This should be False when using Stitching_Helper
% % These numbers are relative to the minimum bounding box of the stitched image
% prefs.CropROIxAfterStitch = [1:3693]; %The image region to be cropped after stitching
% prefs.CropROIyAfterStitch = [2:(2+1280-1)]; % Vert.
% prefs.out.filePath = '/home/beams12/S1IDUSER/mnt/orthros/Faber_Feb14_rec/Bar1_LLZO_LGS_D5/raw';
% prefs.out.filePrefix = 'mp90s0.2_tomo'; % No '_' at the end
% prefs.out.numberDigit = 5;
% prefs.out.fileType = 'uint16';
% prefs.out.VertSizeWarningAt = 128; % Warning messages will be sent out if the size is not a multiplier of this value. No warning if it is zero.
% prefs.doVis.projection = F;
% prefs.vis.stitching.ViewTransm = F; % Showing normalized images
% prefs.stitchproj = 516; % the projection shown in the GUI
% prefs.vis.stitching.ColorCodeLim = [0 4000];
% prefs.rotAxis = {1026+1750, 1026};

% % Faber_Feb14 Bar1_LLZO_LGS_D130 92keV (2x1 stitching) 7.5X, upside down,
% % shift=0, left handed aero
% % redone with the new stitching
% prefs.VolShift = 0; prefs.NumWF1 = 10; prefs.NumProj = 901; prefs.NumWF2 = 10; prefs.NumDarks = 10; 
% prefs = GenImageNums(prefs, 20235, 1, -1); 
% prefs.ROIx = {1:2048, 1:2048};   %  Horizontal size
% prefs.ROIy = {1:1300, 1:1300};   % Vertical size
% prefs.shouldRotate = T; % Rotation according to the detector rotation or rot. axis tilt
% prefs.RotationAngle = { 180, 180}; 
% prefs.shouldCrop = T;   % Cropping within the ROI defined above
% prefs.cropROIx = {50:1930, 1:2000};   % Horizontal cropping
% prefs.cropROIy = {10:10+1280-1, 10:10+1280-1};   % Vertical cropping
% prefs.shouldColumnAverage = F; % Averaging the columns to one row image
% prefs.translx = {-874, 889};  % Translations (pixels) for stitching (horizontal)
% prefs.transly = {0, 2}; %{0, 3};         % Vertical
% prefs.bboxx = 5000; % Horizontal size of the bounding box, where the images can be stitched
% prefs.bboxy = 4500; % Vertical
% prefs.shouldRotateAfterStitch = T; % Should the image be rotated after the stitching
% prefs.RotationAngleAfterStitch = 0.02;
% prefs.shouldCropAfterStitch = T; % Should the image be cropped after the stitching and optional rotation
%                                  % This should be False when using Stitching_Helper
% % These numbers are relative to the minimum bounding box of the stitched image
% prefs.CropROIxAfterStitch = [1:3693]; %The image region to be cropped after stitching
% prefs.CropROIyAfterStitch = [2:(2+1280-1)]; % Vert.
% prefs.out.filePath = '/home/beams12/S1IDUSER/mnt/orthros/Faber_Feb14_rec/Bar1_LLZO_LGS_D130/raw';
% prefs.out.filePrefix = 'mp90s0.2_tomo'; % No '_' at the end
% prefs.out.numberDigit = 5;
% prefs.out.fileType = 'uint16';
% prefs.out.VertSizeWarningAt = 128; % Warning messages will be sent out if the size is not a multiplier of this value. No warning if it is zero.
% prefs.doVis.projection = F;
% prefs.vis.stitching.ViewTransm = T; % Showing normalized images
% prefs.stitchproj = 516; % the projection shown in the GUI
% prefs.vis.stitching.ColorCodeLim = [0 4000];
% prefs.rotAxis = {1026+1750, 1026};
% %%%% slits
% prefs.slits.shouldCorr = F; % Should correct the images based on slit positions
% prefs.slits.slitsnum=[1 2 3 4]; % Number of slit positions used for translation calculation
% % The ROIs where the slit positions will be checked. Cell array of slits: columns --> the stitching regions, rows -> the ROI for different slit positions
% prefs.slits.tlx=105; prefs.slits.tly=695+100;  % top-left and bottom-right slit corner positions
% prefs.slits.brx=1965; prefs.slits.bry=1525-100;
% prefs.slits.ROISizex = 100; prefs.slits.ROISizey = 300;  % ROI size around the corners in pixels, at least (tempSize+3)^2
% prefs.slits.ROIx = {[prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1]; [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1]; [prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1] ; [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1]}; % Corner ROIs of the slits, the corner should be near the center for the reference image
% prefs.slits.ROIy = {[prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1]; [prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1]; [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1] ; [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1]}; 
% prefs.slits.tempFrameSizex = 30; prefs.slits.tempFrameSizey = 20;  % Frame size around the Template in pixels (at least a 3x3 image must be left after removing this frame from the ROI)
% prefs.slits.refkind = 'projection'; % This kind will be used for reference
% prefs.slits.referenceNum = {1}; % The number in the 'refkind' series which will be used as a reference for the whole scan, given for each stitch
% prefs.slits.doVisROIs = F; % Visualizing the cropped out slit ROIs (slow)
% prefs.slitposcheck.projection.Num = {prefs.projection.Num{1}(1:20:900)}; % To make the SlitPosCheck script faster, fewer images can be selected.
% prefs.slits.usePrevious = T; % Whether to use the previously saved slitposition data 'lastslitpos.mat' for saving time or regenerate the data.
% %%%% define ROI here
% prefs.pattern_match.roi(1).template = [2480 1800 50 50];
% prefs.pattern_match.roi(2).template = [2480 2100 50 50];
% prefs.pattern_match.roi(3).template = [2480 2400 50 50];
% prefs.pattern_match.roi(4).template = [2480 2700 50 50];
% prefs.pattern_match.roi(1).search_range = [80 80];
% prefs.pattern_match.roi(2).search_range = [80 80];
% prefs.pattern_match.roi(3).search_range = [80 80];
% prefs.pattern_match.roi(4).search_range = [80 80];


% Faber_Feb14 Disc1_LLZO_LGF_D255 keV (4x1 stitching) 7.5X, upside down,
% shift=0, left handed aero
prefs.VolShift = 0; prefs.NumWF1 = 10; prefs.NumProj = 1801; prefs.NumWF2 = 10; prefs.NumDarks = 10; 
prefs = GenImageNums(prefs, 23959, 1, 1); % Disc1
prefs = GenImageNums(prefs, 31283, 1, 1); % Disc2 
% prefs.ROIx = {1:2048, 1:2048,1:2048, 1:2048};   %  Horizontal size
% prefs.ROIy = {1:1300, 1:1300, 1:1300, 1:1300};   % Vertical size
prefs.shouldRotate = T; % Rotation according to the detector rotation or rot. axis tilt
prefs.RotationAngle = { 180, 180, 180, 180 }; 
prefs.shouldCrop = T;   % Cropping within the ROI defined above
prefs.cropROIx = {1:2048,60:2048,1:2048, 1:2048};   % Horizontal cropping
prefs.cropROIY = {1:2048, 1:2048,1:2048, 1:2048};   
%prefs.cropROIy = {10:10+1280-1, 10:10+1280-1,10:10+1280-1, 10:10+1280-1};   % Vertical cropping
prefs.shouldColumnAverage = F; % Averaging the columns to one row image
prefs.translx = {-2939, -975, 975, 2975};  % Translations (pixels) for stitching (horizontal)
prefs.transly = {9, 0, 0, 0};         % Vertical
prefs.bboxx = 9000; % Horizontal size of the bounding box, where the images can be stitched
prefs.bboxy = 6500; % Vertical
prefs.shouldRotateAfterStitch = F; % Should the image be rotated after the stitching
prefs.RotationAngleAfterStitch = 0.02;
prefs.shouldCropAfterStitch = F; % Should the image be cropped after the stitching and optional rotation
                                 % This should be False when using Stitching_Helper
% These numbers are relative to the minimum bounding box of the stitched image
prefs.CropROIxAfterStitch = [1:3693]; %The image region to be cropped after stitching
prefs.CropROIyAfterStitch = [2:(2+1280-1)]; % Vert.
prefs.out.filePath = '/home/beams12/S1IDUSER/mnt/orthros/Faber_Feb14_rec/Disc1_LLZO_LGF_D255/raw';
prefs.out.filePath = '/home/beams12/S1IDUSER/mnt/orthros/Faber_Feb14_rec/Disc2_LLZO_SGF_D255/raw';
prefs.out.filePrefix = 'mp90s0.2_tomo'; % No '_' at the end
prefs.out.numberDigit = 5;
prefs.out.fileType = 'uint16';
prefs.out.VertSizeWarningAt = 128; % Warning messages will be sent out if the size is not a multiplier of this value. No warning if it is zero.
prefs.doVis.projection = F;
prefs.vis.stitching.ViewTransm = T; % Showing normalized images
prefs.stitchproj = 1; % the projection shown in the GUI
prefs.vis.stitching.ColorCodeLim = [0 4000];
prefs.rotAxis = {4000, 2000, 50, -2050};
%%%% slits
prefs.slits.shouldCorr = F; % Should correct the images based on slit positions
prefs.slits.slitsnum=[3 4]; % Number of slit positions used for translation calculation
% The ROIs where the slit positions will be checked. Cell array of slits: columns --> the stitching regions, rows -> the ROI for different slit positions
prefs.slits.tlx=11+50+40; prefs.slits.tly=10;  % top-left and bottom-right slit corner positions
prefs.slits.brx=2039-50-40; prefs.slits.bry=1270;
prefs.slits.ROISizex = 20+100; prefs.slits.ROISizey = 300;  % ROI size around the corners in pixels, at least (tempSize+3)^2
prefs.slits.ROIx = {...
    [prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1], [prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1], [prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1], [prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1];...
    [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1], [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1], [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1], [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1];...
    [prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1], [prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1], [prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1], [prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1];...
    [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1], [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1], [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1], [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1]}; % Corner ROIs of the slits, the corner should be near the center for the reference image
prefs.slits.ROIy = {...
    [prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1], [prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1], [prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1], [prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1];...
    [prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1], [prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1], [prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1], [prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1];...
    [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1], [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1], [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1], [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1];...
    [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1], [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1], [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1], [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1]}; 
prefs.slits.tempFrameSizex = 7; prefs.slits.tempFrameSizey = 60;  % Frame size around the Template in pixels (at least a 3x3 image must be left after removing this frame from the ROI)
prefs.slits.refkind = 'projection'; % This kind will be used for reference
prefs.slits.referenceNum = {1, 1, 1, 1}; % The number in the 'refkind' series which will be used as a reference for the whole scan, given for each stitch
prefs.slits.doVisROIs = T; % Visualizing the cropped out slit ROIs (slow)
prefs.slits.vis.ColorCodeLim = [0 500];
prefs.slitposcheck.projection.Num = {prefs.projection.Num{1}(1:50:1800), prefs.projection.Num{2}(1:50:1800), prefs.projection.Num{3}(1:50:1800), prefs.projection.Num{4}(1:50:1800)}; % To make the SlitPosCheck script faster, fewer images can be selected.
prefs.slits.usePrevious = T; % Whether to use the previously saved slitposition data 'lastslitpos.mat' for saving time or regenerate the data.
%%%% define ROIs here for stitching-matching
prefs.pattern_match.roi(1).template = [2545 3172 15 50]; % Template size as [x_cent y_cent x_dim y_dim]
prefs.pattern_match.roi(2).template = [2545 3450 15 50];
prefs.pattern_match.roi(3).template = [2545 3850 15 50];
prefs.pattern_match.roi(4).template = [2545 4100 15 50];
prefs.pattern_match.roi(1).search_range = [25 100];   % Search area size as [x_dim y_dim], cannot be smaller than template
prefs.pattern_match.roi(2).search_range = [25 100];
prefs.pattern_match.roi(3).search_range = [25 100];
prefs.pattern_match.roi(4).search_range = [25 100];

% Vertical size should be a 128x value: 128,256,384,512,640,768,896,1024,1152,1280
% 1408,1536,1664,1792,1920,2048,2176


prefs.out.ROIx = [];        % Will be calculated for output size
prefs.out.ROIy = [];
prefs.out.rotAxis = {};     % The calculated positions of the rotation axes in the stitched image for each frames
prefs.out.shift = {};       % The calculated shift values for GridRec for each frames
prefs.out.xposInBBox = {};  % The position of each stitched frames within the bounding box
prefs.out.yposInBBox = {};
prefs.out.timestamp = timestamp(clock);
prefs.out.logFile = ['../ConvToHDF_' prefs.out.timestamp '.log']; % Logs in the destination directory 

%%% some slits initializations
prefs.slits.slitpos.x= cell(max(prefs.slits.slitsnum), prefs.sN);
prefs.slits.slitpos.y= cell(max(prefs.slits.slitsnum), prefs.sN);
prefs.slits.slitpos.Num= cell(prefs.sN);
prefs.slits.slitpos.ncc= cell(prefs.sN);
prefs.slits.slitfname=[prefs.out.filePath '/' prefs.projection.Prefix{1} 'lastslitsfield.mat'];

