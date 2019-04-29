% Author : Ke Yue
% Time: 2015-Dec-15
% File for using Stitching_InitConv_TIFFtoHDF4_single.py to initialize the
% prefs

P = py.Stitching_InitConv_TIFFtoHDF4.initPyPref(prefs);

prefs = struct(P);
prefs.series = double(prefs.series);
prefs.sN = double(prefs.sN);
prefs.filePath = char(prefs.filePath);
prefs.numberDigit = double(prefs.numberDigit);

prefs.projection = struct(prefs.projection);
prefs.projection.Prefix = cellfun(@char,cell(prefs.projection.Prefix), 'UniformOutput', false);
prefs.projection.extension = char(prefs.projection.extension);
prefs.projection.Num = cellfun(@double,cell(prefs.projection.Num), 'UniformOutput', false);


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

prefs.VolShift = double(prefs.VolShift);
prefs.NumWF1 = double(prefs.NumWF1);
prefs.NumProj = double(prefs.NumProj);
prefs.NumWF2 = double(prefs.NumWF2);
prefs.NumDarks = double(prefs.NumDarks);

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

prefs.ROIx = cellfun(@double,cell(prefs.ROIx), 'UniformOutput', false);
prefs.ROIy = cellfun(@double,cell(prefs.ROIy), 'UniformOutput', false);

prefs.RotationAngle = cellfun(@double,cell(prefs.RotationAngle), 'UniformOutput', false);

prefs.cropROIx = cellfun(@double,cell(prefs.cropROIx), 'UniformOutput', false);
prefs.cropROIy = cellfun(@double,cell(prefs.cropROIy), 'UniformOutput', false);

prefs.translx = cellfun(@double,cell(prefs.translx), 'UniformOutput', false);
prefs.transly = cellfun(@double,cell(prefs.transly), 'UniformOutput', false);
prefs.bboxx = double(prefs.bboxx);
prefs.bboxy = double(prefs.bboxy);

prefs.RotationAngleAfterStitch = double(prefs.RotationAngleAfterStitch);
prefs.CropROIxAfterStitch = double(prefs.CropROIxAfterStitch);
prefs.CropROIyAfterStitch = double(prefs.CropROIyAfterStitch);

prefs.rotAxis = cellfun(@double,cell(prefs.rotAxis), 'UniformOutput', false);

prefs.out = struct(prefs.out);
prefs.out.filePath = char(prefs.out.filePath);
prefs.out.filePrefix = char(prefs.out.filePrefix);
prefs.out.numberDigit = double(prefs.out.numberDigit);
prefs.out.fileType = char(prefs.out.fileType);
prefs.out.VertSizeWarningAt = double(prefs.out.VertSizeWarningAt);

prefs.frameNum = double(prefs.frameNum);

prefs.slits= struct(prefs.slits);
prefs.slits.slitsnum= double(prefs.slits.slitsnum);
prefs.slits.tlx = double(prefs.slits.tlx);
prefs.slits.tly = double(prefs.slits.tly);
prefs.slits.brx = double(prefs.slits.brx);
prefs.slits.bry = double(prefs.slits.bry);
prefs.slits.ROISizex = double(prefs.slits.ROISizex);
prefs.slits.ROISizey = double(prefs.slits.ROISizey);

prefs.slits.ROIx = cell(prefs.slits.ROIx);
prefs.slits.ROIx = {double(prefs.slits.ROIx{1}), double(prefs.slits.ROIx{2}), double(prefs.slits.ROIx{3}), double(prefs.slits.ROIx{4}) ; ...
    double(prefs.slits.ROIx{5}),double(prefs.slits.ROIx{6}),double(prefs.slits.ROIx{7}),double(prefs.slits.ROIx{8}); ...
    double(prefs.slits.ROIx{9}), double(prefs.slits.ROIx{10}), double(prefs.slits.ROIx{11}), double(prefs.slits.ROIx{12}) ;... 
    double(prefs.slits.ROIx{13}), double(prefs.slits.ROIx{14}), double(prefs.slits.ROIx{15}), double(prefs.slits.ROIx{16})};

prefs.slits.ROIy = cell(prefs.slits.ROIy);
prefs.slits.ROIy = {double(prefs.slits.ROIy{1}), double(prefs.slits.ROIy{2}), double(prefs.slits.ROIy{3}), double(prefs.slits.ROIy{4}) ; ...
    double(prefs.slits.ROIy{5}),double(prefs.slits.ROIy{6}),double(prefs.slits.ROIy{7}),double(prefs.slits.ROIy{8}); ...
    double(prefs.slits.ROIy{9}), double(prefs.slits.ROIy{10}), double(prefs.slits.ROIy{11}), double(prefs.slits.ROIy{12}) ;... 
    double(prefs.slits.ROIy{13}), double(prefs.slits.ROIy{14}), double(prefs.slits.ROIy{15}), double(prefs.slits.ROIy{16})};

prefs.slits.tempFrameSizex = double(prefs.slits.tempFrameSizex);
prefs.slits.tempFrameSizey = double(prefs.slits.tempFrameSizey);

prefs.slits.refkind = char(prefs.slits.refkind);
prefs.slits.referenceNum = cellfun(@double,cell(prefs.slits.referenceNum), 'UniformOutput', false);

prefs.slits.vis= struct(prefs.slits.vis);
prefs.slits.vis.ColorCodeLim = double(prefs.slits.vis.ColorCodeLim);

prefs.slitposcheck = struct(prefs.slitposcheck);
prefs.slitposcheck.projection = struct(prefs.slitposcheck.projection);
prefs.slitposcheck.projection.Num = cellfun(@double,cell(prefs.slitposcheck.projection.Num), 'UniformOutput', false);

prefs.pattern_match.roi(1).template = double(prefs.template{1}); 
prefs.pattern_match.roi(2).template = double(prefs.template{2});
prefs.pattern_match.roi(3).template = double(prefs.template{3});
prefs.pattern_match.roi(4).template = double(prefs.template{4});
prefs.pattern_match.roi(1).search_range = double(prefs.searchrange{1});   
prefs.pattern_match.roi(2).search_range = double(prefs.searchrange{2});
prefs.pattern_match.roi(3).search_range = double(prefs.searchrange{3});
prefs.pattern_match.roi(4).search_range = double(prefs.searchrange{4});

prefs.out.ROIx = double(prefs.out.ROIx);
prefs.out.ROIy = double(prefs.out.ROIy);
prefs.out.rotAxis = cell(prefs.out.rotAxis);
prefs.out.shift = cell(prefs.out.shift);
prefs.out.xposInBBox = cell(prefs.out.xposInBBox);
prefs.out.yposInBBox = cell(prefs.out.yposInBBox);
prefs.out.timestamp = char(prefs.out.timestamp);
prefs.out.logFile = char(prefs.out.logFile);

prefs.slits.slitpos = struct(prefs.slits.slitpos);
prefs.slits.slitpos.x = cell(prefs.slits.slitpos.x);
prefs.slits.slitpos.y = cell(prefs.slits.slitpos.y);
prefs.slits.slitpos.Num = cell(prefs.slits.slitpos.Num);
prefs.slits.slitpos.ncc = cell(prefs.slits.slitpos.ncc);
prefs.slits.slitfname = char(prefs.slits.slitfname);

prefs = rmfield(prefs, 'template');
prefs = rmfield(prefs, 'searchrange');