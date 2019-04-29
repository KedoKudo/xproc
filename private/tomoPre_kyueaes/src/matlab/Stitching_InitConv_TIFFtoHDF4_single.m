 %% Preferences for converting tiffs to hdf
% convention: image #1 is direct beam; take n tomographic images; 
%     image #(n+2) is the direct beam again; image #(n+3) is the dark beam
% Presently, the images must cover the interval of 0-180 degrees
%
% Written by P. Kenesei, APS

% DO NOT PUT THEM BACK! It will cause error:
% They will be running at the Stitching_Helper or at the Conversion.
%  format compact;
%  close all hidden;
%  fclose('all');
%  clear all;

% Abbreviation for 'true' and 'false'
T = true; F = false;

% Numbers of series made for stitching 
prefs.series = [1]; % Here you can provide the Z order of the series
                % [1 2 3] means 3rd overwrites 2nd which overwrites 1st, 
                % series defined by the order in the cell arrays below.
                % This is effective mostly at the stitching stage.
prefs.sN = length(prefs.series);


%%% DEFAULTS that can be overwritten in the sample specific section
prefs.numberDigit = 6;  % Number of digits in the file numbering, used for putting the leading zeros in file names

% % INPUT files (Note: use '/' for paths, no trailing '/', PREFIX: With trailing '_', if any)
% % prefs.filePath = '/home/beams12/S1IDUSER/mnt/orthros/birkedal_jun14/Tomo';
% %     prefs.projection.Prefix = {'SevenCaps_'};
% 
% prefs.projection.extension = 'tif'; % No dot at the beginning
% prefs.beamProfileInitial.Prefix = prefs.projection.Prefix; % WHITE1
% prefs.beamProfileInitial.extension = prefs.projection.extension;
% prefs.beamProfileFinal.Prefix = prefs.projection.Prefix; % WHITE2
% prefs.beamProfileFinal.extension = prefs.projection.extension;
% prefs.dark.Prefix = prefs.projection.Prefix;  % DARK
% prefs.dark.extension = prefs.projection.extension;
% Default scan numbering
prefs.NumWF1 = 10;  % Number of frames for the initial beam profile (white field)
prefs.NumProj = 901; % Number of frames for the projections
prefs.NumWF2 = 10;  % Number of frames for the initial beam profile (white field)
prefs.NumDarks = 10; % Number of frames for the dark field
prefs.frameNum = prefs.NumWF1 + prefs.NumProj + prefs.NumWF2 + prefs.NumDarks; % Image numbers in one part of the stitching
prefs.VolShift = 0; % Shift of the image numbers to the next volume (default 0)
%prefs.VolShift = prefs.sN *prefs.frameNum; % One volume shift in image numbers

prefs.RotationAngle= 0.0; % default 0, Rotation axis tilt correction in degrees, positive: rotating the image CCW, i.e. rotating the axis CW
prefs.rotAxis = {1024};  % default
% aka prefs.rotAxis = -prefs.translx+1024; 
    % The position of the rotation axis on the full detector images measured during the alignment
    % Warning: This is the coulumn number starting
    % from 1, during alignment this index usually
    % goes from 0
prefs.rotax.fittedshift=0; % The shift determined from the FindRotax script. This will be used for the reconstruction as a starter value

% Visualization options
prefs.SkipInitialBeamProfile = F; % Skip to generate WF1 for saving  time during test runs
prefs.doVis.projection = F;  % Should be done any visualization on the images
prefs.doVis.beamProfileInitial = T;
prefs.doVis.beamProfileFinal = T;
prefs.doVis.dark = T;
prefs.vis.projection.imgRange = [000 4100];
prefs.vis.beamProfileInitial.imgRange = [0 4096];
prefs.vis.beamProfileFinal.imgRange = [0 4096];
prefs.vis.dark.imgRange = [95 100]; % for CoolSnap
prefs.vis.dark.imgRange = [0 20]; % for Retiga
prefs.vis.intMonitor = T; % Should we calculate the average intensities 
                             % on the preprocessed images to monitor the success of acquisition
prefs.doVis.stitching = F; % This must be simply false (used as a flag internally for indicating the StitchGUI is used)
prefs.vis.stitching.ColorCodeLim = [100 2000]; % Initial colorbar min/max value
prefs.vis.stitching.init = T; % It must be true for the first image initialization
prefs.vis.stitching.ViewTransm = T; % Should we show normalized (transmission) images, or the raw projections 
prefs.stitchproj = 1; % The projection which we will visually stitch at Stitching_Helper startup
prefs.ROIx = {1:2048};   % Horizontal size % The part of the file (ROI) that will be read in
prefs.ROIy = {1:2048};  % Vertical size

% ROIs for reducing the data size and making some pre-processing
% Vertical size should be a 128x value: 128,256,384,512,640,768,896,1024,1152,1280
% 1408,1536,1664,1792,1920,2048,2176

% New type of configuration:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% For SlitPosCheck tests
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % sangid_nov15, 5X, tomo detector D=90mm, 71.676keV, still images, aero left handed 11deg-169deg, compact loadframe
% prefs.userID='sangid_nov15';
% prefs.filePath = sprintf('/home/beams12/S1IDUSER/mnt/s1c/%s/tomo', prefs.userID);
%    prefs.projection.Prefix = {'J5_In718_Load00_'}; prefs.firstfile=2; prefs.loadstep=''; prefs.rotax.fittedshift = 4; % Put leading '_' for the loadstep, or '' for skipping
% prefs.numberDigit = 6;
% prefs = defaultfnames(prefs);
% prefs.VolShift = 0; prefs.NumWF1 = 10; prefs.NumProj = 791; prefs.NumWF2 = 10; prefs.NumDarks = 10; 
% prefs = GenImageNums(prefs, prefs.firstfile, 1, -1); % last param -1 for the left handed aero
% %prefs.out.filePrefix = 'm180p0s0.2_tomo'; % No '_' at the end
% %prefs.out.filePrefix = 'mp180s0.2_tomo'; % No '_' at the end
% %prefs.out.filePrefix = 'p11p169s0.2_tomo'; % No '_' at the end % 791 frames
% prefs.out.filePrefix = 'p11p168s0.2_tomo'; % No '_' at the end % 786 frames
% prefs.projection.Num{1}=prefs.projection.Num{1}(6:end); prefs.NumProj = 786; prefs.frameNum=786+10+10+10;
% % DO NOT USE '~', because the HDF4 writer is not compatible with it.
% prefs.out.dirname=sprintf('%s_rec/%s%s/raw', prefs.userID, prefs.projection.Prefix{1}(1:end-1), prefs.loadstep);
% prefs.out.filePath = sprintf('/home/beams12/S1IDUSER/mnt/orthros/%s', prefs.out.dirname);
% prefs.ROIx = {180:1900};   % Horizontal size % The part of the file (ROI) that will be read in
% prefs.ROIy = {700:1750};  % Vertical size
% prefs.shouldRotate = T; % Rotation according to the detector rotation
% prefs.RotationAngle = {0.0}; 
% prefs.shouldCrop = T;   % Cropping within the ROI defined above
% prefs.cropROIx = {10:10+1700-1};    % Horizontal cropping 
% prefs.cropROIy = {12:12+1024-1};   % Vertical  % MUST BE MULTIPLE OF 128
% prefs.doVis.projection = F;
% prefs.SkipInitialBeamProfile = F; % Skip to generate WF1 for saving  time during test runs
% prefs.rotAxis = {1020};  % assumed rot.ax position on the detector image
% prefs.vis.projection.imgRange = [300 4000];
% %%%% slitpos correction
% prefs.slits.shouldCorr = F; % Should correct the images based on slit positions
% prefs.slits.usePrevious = F; % Whether to use the previously saved slitposition data 'lastslitpos.mat' for saving time or regenerate the data.
% prefs.slits.slitsnum=[1 2 3 4]; % Number of slit positions used for translation calculation
% % The ROIs where the slit positions will be checked. Cell array of slits: columns --> the stitching regions, rows -> the ROI for different slit positions
% prefs.slits.tlx=167; prefs.slits.tly=777;  % top-left and bottom-right slit corner positions
% prefs.slits.brx=1917; prefs.slits.bry=1681;
% prefs.slits.ROISizex = 150; prefs.slits.ROISizey = 150;  % ROI size around the corners in pixels, at least (2*tempSize+3)
% prefs.slits.ROIx = {[prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1]; [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1]; [prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1] ; [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1]}; % Corner ROIs of the slits, the corner should be near the center for the reference image
% prefs.slits.ROIy = {[prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1]; [prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1]; [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1] ; [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1]}; 
% prefs.slits.tempFrameSizex = 20; prefs.slits.tempFrameSizey = 20;  % Frame size around the Template in pixels (at least a 3x3 image must be left after removing this frame from the ROI)
% prefs.slits.refkind = 'projection'; % This kind will be used for reference
% prefs.slits.referenceNum = {1}; % The number in the 'refkind' series which will be used as a reference for the whole scan, given for each stitch
% prefs.slits.doVisROIs = F; % Visualizing the cropped out slit ROIs (slow)
% prefs.slitposcheck.projection.Num = {prefs.projection.Num{1}(1:ceil((length(prefs.projection.Num{1})-1)/72):end)}; % To make the SlitPosCheck script faster, fewer images can be selected.
% %%% correlation
% prefs.correl.VertCorr = F; % default=F, Horizontal lines will be crosscorrelated, othervise vertical (experimental)
% prefs.correl.dark = min(5, length(prefs.dark.Num{1}) ); % dark image number in the Num field
% prefs.correl.white = min(5, length(prefs.beamProfileInitial.Num{1}) ); % initWF image number in the Num field
% prefs.correl.first = 1;  % First image for correlation in the Num field, 'last' is the symmetric pair of it
% prefs.correl.last = 791; %prefs.correl.first+(length(prefs.projection.Num{1})-1)/2+1;
% prefs.correl.smallROIy = [90:950]; % vertical 
% prefs.correl.smallROIx = [206:1516]; % template roi
% prefs.correl.bigROIx = [100:1600]; % image roi
% prefs.correl.rowstep = 4; % Size of the steps in the ROIy
% prefs.correl.bandwidth = 2; % size of the xproduced band
% %prefs.rotax.fittedshift = -28; % The result from FindRotAx, used for slice reconstructions.

% sangid_nov15, 5X, tomo detector D=90mm, 71.676keV, still images, aero left handed 11deg-169deg, compact loadframe
prefs.userID='sangid_nov15';
% prefs.filePath = sprintf('/home/beams12/S1IDUSER/mnt/s1c/%s/tomo', prefs.userID);

prefs.filePath = '/home/beams/S1IDUSER/opt/tomoproc/testsangid_nov15';


   prefs.projection.Prefix = {'J5_In718_Load01_'}; prefs.firstfile=823; prefs.loadstep=''; prefs.rotax.fittedshift = 11; % Put leading '_' for the loadstep, or '' for skipping
   prefs.projection.Prefix = {'J5_In718_Load02_'}; prefs.firstfile=1639; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping
   prefs.projection.Prefix = {'J5_In718_Load05_'}; prefs.firstfile=4143; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping
   prefs.projection.Prefix = {'J5_In718_Load03_'}; prefs.firstfile=2510; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping
   prefs.projection.Prefix = {'J5_In718_Load04_'}; prefs.firstfile=3327; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping
   prefs.projection.Prefix = {'J5_In718_Load06_'}; prefs.firstfile=6592; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping
   prefs.projection.Prefix = {'J5_In718_Load05_top_'}; prefs.firstfile=4960; prefs.loadstep='_a'; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping
   prefs.projection.Prefix = {'J5_In718_Load05_top_'}; prefs.firstfile=5776; prefs.loadstep='_b'; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping
   prefs.projection.Prefix = {'J5_In718_Load08_'}; prefs.firstfile=8615; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping
 prefs.projection.Prefix = {'J5_In718_Load09_'}; prefs.firstfile=9431; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping
 prefs.projection.Prefix = {'J5_In718_Load10_'}; prefs.firstfile=10247; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping 
  prefs.projection.Prefix = {'J5_In718_Load11_'}; prefs.firstfile=11063; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping 
%     prefs.projection.Prefix = {'J5_In718_Load12_'}; prefs.firstfile=11879; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping  
%     prefs.projection.Prefix = {'J5_In718_Load14_'}; prefs.firstfile=12766; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping  
%       prefs.projection.Prefix = {'E5_In718_PP_'}; prefs.firstfile=14399; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping 
%           prefs.projection.Prefix = {'5D_In718_SR_'}; prefs.firstfile=15215; prefs.loadstep=''; prefs.rotax.fittedshift = 10; % Put leading '_' for the loadstep, or '' for skipping  
 prefs.numberDigit = 6;
prefs = defaultfnames(prefs);
prefs.VolShift = 0; prefs.NumWF1 = 10; prefs.NumProj = 786; prefs.NumWF2 = 10; prefs.NumDarks = 10; 
prefs = GenImageNums(prefs, prefs.firstfile, 1, -1); % last param -1 for the left handed aero
%prefs.out.filePrefix = 'm180p0s0.2_tomo'; % No '_' at the end
%prefs.out.filePrefix = 'mp180s0.2_tomo'; % No '_' at the end
%prefs.out.filePrefix = 'p11p169s0.2_tomo'; % No '_' at the end % 791 frames
prefs.out.filePrefix = 'p11p168s0.2_tomo'; % No '_' at the end % 786 frames
% prefs.projection.Num{1}=prefs.projection.Num{1}(6:end); prefs.NumProj = 786; prefs.frameNum=786+10+10+10;
% DO NOT USE '~', because the HDF4 writer is not compatible with it.
prefs.out.dirname=sprintf('%s_rec/%s%s/raw', prefs.userID, prefs.projection.Prefix{1}(1:end-1), prefs.loadstep);
prefs.out.filePath = sprintf('/home/beams12/S1IDUSER/mnt/orthros/%s', prefs.out.dirname);
prefs.ROIx = {180:1900};   % Horizontal size % The part of the file (ROI) that will be read in
prefs.ROIy = {700:1750};  % Vertical size
prefs.shouldRotate = T; % Rotation according to the detector rotation
prefs.RotationAngle = {0.0}; 
prefs.shouldCrop = T;   % Cropping within the ROI defined above
prefs.cropROIx = {10:10+1700-1};    % Horizontal cropping 
prefs.cropROIy = {12:12+1024-1};   % Vertical  % MUST BE MULTIPLE OF 128
prefs.doVis.projection = F;
prefs.SkipInitialBeamProfile = F; % Skip to generate WF1 for saving  time during test runs
prefs.rotAxis = {1020};  % assumed rot.ax position on the detector image
prefs.vis.projection.imgRange = [300 4000];
%%%% slitpos correction
prefs.slits.shouldCorr = F; % Should correct the images based on slit positions
prefs.slits.usePrevious = F; % Whether to use the previously saved slitposition data 'lastslitpos.mat' for saving time or regenerate the data.
prefs.slits.slitsnum=[1 2 3 4]; % Number of slit positions used for translation calculation
% The ROIs where the slit positions will be checked. Cell array of slits: columns --> the stitching regions, rows -> the ROI for different slit positions
prefs.slits.tlx=167; prefs.slits.tly=777;  % top-left and bottom-right slit corner positions
prefs.slits.brx=1917; prefs.slits.bry=1681;
prefs.slits.ROISizex = 150; prefs.slits.ROISizey = 150;  % ROI size around the corners in pixels, at least (2*tempSize+3)


%comment out for python 1*N cell incomptatiable
prefs.slits.ROIx = {[prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1]; [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1]; [prefs.slits.tlx-floor(prefs.slits.ROISizex/2):prefs.slits.tlx+ceil(prefs.slits.ROISizex/2)-1] ; [prefs.slits.brx-floor(prefs.slits.ROISizex/2):prefs.slits.brx+ceil(prefs.slits.ROISizex/2)-1]}; % Corner ROIs of the slits, the corner should be near the center for the reference image
prefs.slits.ROIy = {[prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1]; [prefs.slits.tly-floor(prefs.slits.ROISizey/2):prefs.slits.tly+ceil(prefs.slits.ROISizey/2)-1]; [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1] ; [prefs.slits.bry-floor(prefs.slits.ROISizey/2):prefs.slits.bry+ceil(prefs.slits.ROISizey/2)-1]}; 
prefs.slits.tempFrameSizex = 20; prefs.slits.tempFrameSizey = 20;  % Frame size around the Template in pixels (at least a 3x3 image must be left after removing this frame from the ROI)
prefs.slits.refkind = 'projection'; % This kind will be used for reference
prefs.slits.referenceNum = {1}; % The number in the 'refkind' series which will be used as a reference for the whole scan, given for each stitch
prefs.slits.doVisROIs = F; % Visualizing the cropped out slit ROIs (slow)
prefs.slitposcheck.projection.Num = {prefs.projection.Num{1}(1:ceil((length(prefs.projection.Num{1})-1)/72):end)}; % To make the SlitPosCheck script faster, fewer images can be selected.
%%% correlation
prefs.correl.VertCorr = F; % default=F, Horizontal lines will be crosscorrelated, othervise vertical (experimental)
prefs.correl.dark = min(5, length(prefs.dark.Num{1}) ); % dark image number in the Num field
prefs.correl.white = min(5, length(prefs.beamProfileInitial.Num{1}) ); % initWF image number in the Num field
prefs.correl.first = 1;  % First image for correlation in the Num field, 'last' is the symmetric pair of it
prefs.correl.last = prefs.correl.first+(length(prefs.projection.Num{1})-1); %prefs.correl.first+(length(prefs.projection.Num{1})-1)/2+1;
prefs.correl.smallROIy = [90:950]; % vertical 
prefs.correl.smallROIx = [206:1516]; % template roi
prefs.correl.bigROIx = [100:1600]; % image roi
prefs.correl.rowstep = 4; % Size of the steps in the ROIy
prefs.correl.bandwidth = 2; % size of the xproduced band
%prefs.rotax.fittedshift = -28; % The result from FindRotAx, used for slice reconstructions.

% Vertical size should be a 128x value: 128,256,384,512,640,768,896,1024,1152,1280
% 1408,1536,1664,1792,1920,2048,2176


% 1 slice
%prefs.cropROIy = {8+231:8+231};   % Vertical cropping

prefs.shouldColumnAverage = F; % Averaging the columns to one row image
prefs.translx = {0};  % Translations (pixels) for stitching (horizontal)
prefs.transly = {0};         % Vertical
prefs.bboxx = 3048; % Horizontal size of the bounding box, where the images can be stitched
prefs.bboxy = 3048; % Vertical

prefs.shouldRotateAfterStitch = F; % Should the image be rotated after the stitching
prefs.RotationAngleAfterStitch = -0.37;
prefs.shouldCropAfterStitch = F; % Should the image be cropped after the stitching and optional rotation
prefs.CropROIxAfterStitch = [10:7051]; % Horiz. The image region to be cropped after stitching
prefs.CropROIyAfterStitch = [24:488]; % Vert.

%%%% SOME INITIALIZATIONS

prefs.out.numberDigit = 5; % for tomompi this is 5
prefs.out.fileType = 'uint16'; % for tomompi this is typically 'uint16'
prefs.out.VertSizeWarningAt = 128; % Warning messages will be sent out if the size is not a multiplier of this value. No warning if it is zero. This is necessary because of tomompi restrictions
prefs.out.ROIx = [];        % Will be calculated for output size 
prefs.out.ROIy = [];
prefs.out.rotAxis = {};     % The calculated positions of the rotation axes in the stitched image for each frames
prefs.out.shift = {};       % The calculated shift values for GridRec for each frames
prefs.out.xposInBBox = {};  % The position of each stitched frames within the bounding box
prefs.out.yposInBBox = {};
prefs.out.timestamp = timestamp(clock);
prefs.out.logFile = ['../ConvToHDF_' prefs.out.timestamp '.log']; % Logs in the destination directory 

%%% some slits initializations  comment out as the N*1 cell can not be pass
%%% to python

prefs.slits.slitpos.x= cell(max(prefs.slits.slitsnum), prefs.sN);
prefs.slits.slitpos.y= cell(max(prefs.slits.slitsnum), prefs.sN);


prefs.slits.slitpos.Num= cell(prefs.sN);
prefs.slits.slitpos.ncc= cell(prefs.sN);
prefs.slits.slitfname=[prefs.out.filePath '/' prefs.projection.Prefix{1} 'lastslitsfield.mat'];


% Quick tomo reconstruction on some layers for image rotation: 
% the parametrization fits to the general settings

% For quick parameter setting and visualization
if 0
%   prefs.selectedLayers = [100 500 1000 1600]
    prefs.ROIx = {1:2048};   % Horizontal size
    prefs.ROIy = {1:2048};   % Vertical size
    prefs.doVis.projection = T;
 %   prefs.ROIy = {[]};
 %   prefs.layerEmbedding = 10;
  %  for i = prefs.selectedLayers
  %      prefs.ROIy{1} = horzcat(prefs.ROIx{1}, [(i-prefs.layerEmbedding):(i+prefs.layerEmbedding)]); % Vertical size
  %  end
    prefs.shouldRotate = F; % Rotation according to the detector rotation
    prefs.RotationAngle = {0.0};
    prefs.shouldCrop = F;   % Cropping within the ROI defined above
    prefs.cropROIx = {1:2048};   % Horizontal cropping
    prefs.cropROIy = {100:400:2000};   % Vertical
    disp(['Number of layers for the quick reconstruction:', length(prefs.cropROIy{1})])
end

% Quick transforms for full reconstruction: the parametrization fits to the general
% settings
if F
    prefs.ROIx = {1:2048};   % Horizontal size
    prefs.ROIy = {1:2048};  % Vertical size
    prefs.shouldRotate = T; % Rotation according to the detector rotation
    prefs.RotationAngle = {-0.5729};
    prefs.RotationAngle = {-1.187}; % Alshibli Nov12 F75 7.5X
    prefs.shouldCrop = T;   % Cropping within the ROI defined above
    prefs.cropROIx = {1:2048};   % Horizontal cropping
    prefs.cropROIy = {1:2048};   % Vertical
end

prefs