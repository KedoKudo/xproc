from __future__ import print_function
import tomopy
import Stitching_Conv_TIFFtoHDF4
import timeit
import sys

if __name__ == '__main__':

    start1 = timeit.default_timer()
    prefs = Stitching_Conv_TIFFtoHDF4.prefInitMatlab()
    dataWhite1, dataProj, dataWhite2, dataDark = Stitching_Conv_TIFFtoHDF4.dataPrepare(prefs)
    stop1 = timeit.default_timer()
    print("end read data", (stop1 - start1))


    # proj = dataProj
    flat = dataWhite1
    dark = dataDark

    step = 100;

    index = dataProj.shape[1] / step+1

    for i in range(0, index):
        sliceStart = i*step;
        sliceEnd = (i+1)*step;
        if(sliceStart >= dataProj.shape[1]):
            sys.exit(0);
        if(sliceEnd > dataProj.shape[1]):
            sliceEnd = dataProj.shape[1];

        proj = dataProj[:,sliceStart:sliceEnd,:]

        start1 = timeit.default_timer()
        theta = tomopy.angles(proj.shape[0], 11, 168)

        stop1 = timeit.default_timer()
        print("end angles", (stop1 - start1))

        start2 = timeit.default_timer()
        # # Flat-field correction of raw data.
        proj = tomopy.normalize(proj, flat, dark)
        stop2 = timeit.default_timer()
        print("end normalize", (stop2 - start2))

        # start3 = timeit.default_timer()
        # # # Find rotation center.
        # rot_center = tomopy.find_center(proj, theta, ind=0, init=1024, tol=0.5)
        # stop3 = timeit.default_timer()
        # print("end find_center", (stop3 - start3))
        rot_center = 840
        #
        # # Reconstruct object using Gridrec algorithm.

        start4 = timeit.default_timer()
        rec = tomopy.recon(proj, theta, center=rot_center, algorithm='gridrec')

        stop4 = timeit.default_timer()
        print("end recon", (stop4 - start4))

        # #print("rec is ", rec)
        # # Mask each reconstructed slice with a circle.
        # rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)
        #
        # # Write data as stack of TIFs.
        start5 = timeit.default_timer()
        tomopy.write_tiff_stack(rec, fname='recon_dir/recon', start = sliceStart)
        stop5 = timeit.default_timer()
        print("end write_tiff_stack", (stop5 - start5))
