tomoPre
============

Preprocessing python function before tomo reconstruction

Features
============



For this script, there are two ways to define prefers parameters. 

The first way is to define the parameters in Stitching_InitConv_TIFFtoHDF4_single.py and Stitching_InitConv_TIFFtoHDF4.py file.

The function are:

* prefInitSinglePy()

* prefInitPy()

The second way is to use the matlab file under matlab folder to generate a test.mat file, and use python to readin all the parameters.

The function is:

* prefInitMatlab()

The function used for reading the data from the image file:

* dataPrepare(prefs)

Usage
============
* Modify the prefs['filePath'] in Stitching_InitConv_TIFFtoHDF4_single.py and Stitching_InitConv_TIFFtoHDF4.py to point to your image path as the input
* Modify the prefs in Stitching_InitConv_TIFFtoHDF4_single.py and Stitching_InitConv_TIFFtoHDF4.py for all your other parameters.
* If you prefer to use matlab for parameter input, go to matlab folder, change correspondingly matlab file and run PrefsInitialize.m to generate the mat file having all the parameters.
* rec_1ID_example.py is example script for using the preprocessing function with tomopy.
* In rec_1ID_example.py, set the result path using function write_tiff_stack().
