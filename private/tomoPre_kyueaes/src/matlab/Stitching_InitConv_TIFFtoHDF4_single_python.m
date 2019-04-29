% Author : Ke Yue
% Time: 2015-Dec-15
% File for using Stitching_InitConv_TIFFtoHDF4_single.py to initialize the
% prefs

% clear classes
mod = py.importlib.import_module('Stitching_InitConv_TIFFtoHDF4_single');
py.reload(mod);

prefs = py.dict();


P = py.Stitching_InitConv_TIFFtoHDF4_single.initPyPref(prefs);



prefs = struct(P);
% prefs.which = 'single';

prefs.series = double(prefs.series);
prefs.sN = double(prefs.sN);
prefs.numberDigit = double(prefs.numberDigit);
prefs.NumWF1 = double(prefs.NumWF1);
prefs.NumProj = double(prefs.NumProj);
prefs.NumWF2 = double(prefs.NumWF2);
prefs.NumDarks = double(prefs.NumDarks);
prefs.frameNum = double(prefs.frameNum);
prefs.VolShift = double(prefs.VolShift);
prefs.RotationAngle = cellfun(@double,cell(prefs.RotationAngle), 'UniformOutput', false);
prefs.rotAxis = cellfun(@double,cell(prefs.rotAxis), 'UniformOutput', false);

prefs.rotax = struct(prefs.rotax);
prefs.rotax.fittedshift = double(prefs.rotax.fittedshift);

prefs.doVis = struct(prefs.doVis);

prefs.vis = struct(prefs.vis);
prefs.vis.projection = struct(prefs.vis.projection);
prefs.vis.projection.imgRange = double(prefs.vis.projection.imgRange);

prefs.vis.beamProfileInitial = struct(prefs.vis.beamProfileInitial);
prefs.vis.beamProfileInitial.imgRange= double(prefs.vis.beamProfileInitial.imgRange);

prefs.vis.beamProfileFinal = struct(prefs.vis.beamProfileFinal);
prefs.vis.beamProfileFinal.imgRange= double(prefs.vis.beamProfileFinal.imgRange);

prefs.vis.dark = struct(prefs.vis.dark);
prefs.vis.dark.imgRange= double(prefs.vis.dark.imgRange);

prefs.vis.stitching = struct(prefs.vis.stitching);
prefs.vis.stitching.ColorCodeLim = double (prefs.vis.stitching.ColorCodeLim);

prefs.stitchproj=double(prefs.stitchproj);

prefs.userID = char(prefs.userID);
prefs.filePath = char(prefs.filePath);

prefs.projection = struct(prefs.projection);


% prefs.projection.Prefix = cell(prefs.projection.Prefix);
% prefs.projection.Prefix = char(prefs.projection.Prefix);
prefs.projection.Prefix = cellfun(@char,cell(prefs.projection.Prefix), 'UniformOutput', false);
prefs.projection.extension = char(prefs.projection.extension);
prefs.projection.Num = cellfun(@double,cell(prefs.projection.Num), 'UniformOutput', false);


prefs.firstfile = double(prefs.firstfile);
prefs.loadstep = char(prefs.loadstep);
prefs.numberDigit = double(prefs.numberDigit);

prefs.beamProfileInitial = struct(prefs.beamProfileInitial);
prefs.beamProfileInitial.Prefix = cellfun(@char,cell(prefs.beamProfileInitial.Prefix), 'UniformOutput', false);
prefs.beamProfileInitial.extension = char(prefs.beamProfileInitial.extension);
prefs.beamProfileInitial.Num = cellfun(@double,cell(prefs.beamProfileInitial.Num), 'UniformOutput', false);


prefs.beamProfileFinal = struct(prefs.beamProfileFinal);
prefs.beamProfileFinal.Prefix = cellfun(@char,cell(prefs.beamProfileFinal.Prefix), 'UniformOutput', false);
prefs.beamProfileFinal.extension = char(prefs.beamProfileFinal.extension);
prefs.beamProfileFinal.Num = cellfun(@double,cell(prefs.beamProfileFinal.Num), 'UniformOutput', false);


prefs.dark = struct(prefs.dark);
prefs.dark.Prefix = cellfun(@char,cell(prefs.dark.Prefix), 'UniformOutput', false);
prefs.dark.extension = char(prefs.dark.extension);
prefs.dark.Num = cellfun(@double,cell(prefs.dark.Num), 'UniformOutput', false);

prefs.out = struct(prefs.out);
prefs.out.filePath = char(prefs.out.filePath);
prefs.out.dirname = char(prefs.out.dirname);
prefs.out.filePath = char(prefs.out.filePath);
prefs.out.numberDigit = double(prefs.out.numberDigit);
prefs.out.fileType = char(prefs.out.fileType);
prefs.out.filePrefix = char(prefs.out.filePrefix);
prefs.out.VertSizeWarningAt = double(prefs.out.VertSizeWarningAt);
prefs.out.ROIx = double(prefs.out.ROIx);
prefs.out.ROIy = double(prefs.out.ROIy);
prefs.out.rotAxis = cell(prefs.out.rotAxis);
prefs.out.shift = cell(prefs.out.shift);
prefs.out.xposInBBox = cell(prefs.out.xposInBBox);
prefs.out.yposInBBox = cell(prefs.out.yposInBBox);
prefs.out.timestamp = char(prefs.out.timestamp);
prefs.out.logFile = char(prefs.out.logFile);

prefs.ROIx = cellfun(@double,cell(prefs.ROIx), 'UniformOutput', false);
prefs.ROIy = cellfun(@double,cell(prefs.ROIy), 'UniformOutput', false);

prefs.cropROIx = cellfun(@double,cell(prefs.cropROIx), 'UniformOutput', false);
prefs.cropROIy = cellfun(@double,cell(prefs.cropROIy), 'UniformOutput', false);

prefs.slits= struct(prefs.slits);
prefs.slits.slitsnum= double(prefs.slits.slitsnum);
prefs.slits.tlx = double(prefs.slits.tlx);
prefs.slits.tly = double(prefs.slits.tly);
prefs.slits.brx = double(prefs.slits.brx);
prefs.slits.bry = double(prefs.slits.bry);
prefs.slits.ROISizex = double(prefs.slits.ROISizex);
prefs.slits.ROISizey = double(prefs.slits.ROISizey);

prefs.slits.ROIx = cellfun(@double,cell(prefs.slits.ROIx), 'UniformOutput', false);
prefs.slits.ROIy = cellfun(@double,cell(prefs.slits.ROIy), 'UniformOutput', false);

prefs.slits.tempFrameSizex = double(prefs.slits.tempFrameSizex);
prefs.slits.tempFrameSizey = double(prefs.slits.tempFrameSizey);
prefs.slits.refkind = char(prefs.slits.refkind);

prefs.slits.referenceNum = cellfun(@double,cell(prefs.slits.referenceNum), 'UniformOutput', false);

prefs.slits.slitpos = struct(prefs.slits.slitpos);
prefs.slits.slitpos.x = cell(prefs.slits.slitpos.x);
prefs.slits.slitpos.y = cell(prefs.slits.slitpos.y);
prefs.slits.slitpos.Num = cell(prefs.slits.slitpos.Num);
prefs.slits.slitpos.ncc = cell(prefs.slits.slitpos.ncc);

prefs.slits.slitfname = char(prefs.slits.slitfname);

prefs.slitposcheck = struct(prefs.slitposcheck);
prefs.slitposcheck.projection = struct(prefs.slitposcheck.projection);
prefs.slitposcheck.projection.Num = cellfun(@double,cell(prefs.slitposcheck.projection.Num), 'UniformOutput', false);

prefs.correl = struct(prefs.correl);
prefs.correl.dark = double(prefs.correl.dark);
prefs.correl.white = double(prefs.correl.white);
prefs.correl.first = double(prefs.correl.first);
prefs.correl.last = double(prefs.correl.last);
prefs.correl.smallROIy = double(prefs.correl.smallROIy);
prefs.correl.smallROIx = double(prefs.correl.smallROIx);
prefs.correl.bigROIx = double(prefs.correl.bigROIx);
prefs.correl.rowstep = double(prefs.correl.rowstep);
prefs.correl.bandwidth = double(prefs.correl.bandwidth);
prefs.translx = cellfun(@double,cell(prefs.translx), 'UniformOutput', false);
prefs.transly = cellfun(@double,cell(prefs.transly), 'UniformOutput', false);
prefs.bboxx = double(prefs.bboxx);
prefs.bboxy = double(prefs.bboxy);
prefs.RotationAngleAfterStitch = double(prefs.RotationAngleAfterStitch);
prefs.CropROIxAfterStitch = double(prefs.CropROIxAfterStitch);
prefs.CropROIyAfterStitch = double(prefs.CropROIyAfterStitch);



