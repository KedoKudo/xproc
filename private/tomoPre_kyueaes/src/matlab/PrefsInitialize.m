%%%% PrefsInitialize

% Initializing the conversion parameters, clearing the variables
format compact;
close all hidden;
fclose('all');
drawnow;

prefs.which = 'single';

disp(''); disp(['Program started: ' timestamp(clock)]);
% Loading prefs
switch prefs.which
    case 'simul'
        InitSimulConv, prefs.initfilename = 'InitSimulConv';
        %InitConv_TIFFtoHDF4, prefs.initfilename = 'InitConv_TIFFtoHDF4';
    case 'stitch'
        %%%% Stitching
        Stitching_InitConv_TIFFtoHDF4, prefs.initfilename = 'Stitching_InitConv_TIFFtoHDF4';
%         Stitching_InitConv_TIFFtoHDF4_Python, prefs.initfilename = 'Stitching_InitConv_TIFFtoHDF4';
    case 'single'
        %%%% Single
        Stitching_InitConv_TIFFtoHDF4_single, prefs.initfilename = 'Stitching_InitConv_TIFFtoHDF4_single';
%         Stitching_InitConv_TIFFtoHDF4_single_python, prefs.initfilename = 'Stitching_InitConv_TIFFtoHDF4_single';
    otherwise
        prefs.initfilename=nativefn(prefs.which);
end
prefs.ticStart = tic;
% Miscellenous operations: OS independency
prefs.filePath = nativefn(prefs.filePath);
prefs.projection.Prefix = nativefn(prefs.projection.Prefix);
prefs.beamProfileInitial.Prefix = nativefn(prefs.beamProfileInitial.Prefix);
prefs.beamProfileFinal.Prefix = nativefn(prefs.beamProfileFinal.Prefix);
prefs.dark.Prefix = nativefn(prefs.dark.Prefix);
prefs.out.filePath = nativefn(prefs.out.filePath);
prefs.out.filePrefix = nativefn(prefs.out.filePrefix);
prefs.out.logFile = nativefn([prefs.out.filePath '/' prefs.out.logFile]);

% save the prefs into txt file
% writetable(struct2table(prefs), 'somefile.txt')
save 'test.mat' prefs
% save 'test.txt' prefs -ascii
% t = struct2table(prefs)
% writetable(t, 'station.xlsx');