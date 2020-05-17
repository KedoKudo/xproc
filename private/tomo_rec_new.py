#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python script to reconstruct the APS 6-BM tomography data using Tomopy.

    To-do: (1) check if file/folder exist.
           (2) post-reconstruction crop

by Andrew Chuang, cchuang@aps.anl.gov
"""

# from __future__ import print_function
import sys

sys.path.append("/home/beams/CCHUANG/codezoo/python/img_util/")
import img_util as imu
import tomopy
import dxchange.reader as dxreader
import dxchange.writer as dxwriter
import numpy as np
import time
import os
import cv2
from argparse import ArgumentParser as parser

if __name__ == "__main__":

    # create parser
    par = parser(description="Tomography Image Reconstruction Script")
    # define input
    par.add_argument("exp_log", help="Exp_log file")
    par.add_argument("scan_id", help="ID of the scan (defined in Exp_log file)")
    par.add_argument("rec_type", help="'full' or 'sub' reconstruction")

    # define optional input
    par.add_argument("--rot-axis", type=float, help="specify rotation axis")
    par.add_argument("--rot-step", type=float, help="rotation axis step(s)")
    par.add_argument("--mblur", type=int, help="ksize of median filter")

    args = par.parse_args()  # returns data from the options specified (echo)

    # initialize parameters
    opt = imu.tomo_par_init(args.exp_log, args.scan_id)

    # assign value based on input files
    rec_type = args.rec_type
    show_rec_img = 0  # show rec img to check rot_center
    clean_up_folder = 1  # clean up destination folder for resconstructed files

    # Assign scan parameters based on exp_log file
    if args.rot_axis:
        rot_center = args.rot_axis
        print("rotation axis is specified by user as: ", rot_center)
    else:
        rot_center = opt["rot_axis"]

    outpath = opt["path_output"]
    fprefix = opt["prefix"]
    rot_angle = opt["rot_angle"]
    clim = opt["clim"]
    nstitch = opt["nstitch"]
    roi = opt["roi"]
    th_start = opt["theta"][0]
    th_end = opt["theta"][1]
    postrecroi = opt["post_recon_roi"]
    postrecrot = opt["post_recon_rot"]
    focochkroi = opt["focus_check_roi"]

    if args.rot_step:  # rec will be done +-10 steps around rot_center
        rc_step = args.rot_step
        print("rotation axis split step is specified by user as: ", args.rot_step)
    else:
        rc_step = 0.1

    if args.mblur:  # median filter to remove sand noise in raw images
        medfilter_size = int(abs(args.mblur))
        print("Apply noise reduction on raw images, ksize: ", args.mblur)
    else:
        medfilter_size = None

    c_scaling16 = 65536 / (clim[1] - clim[0] + 1)
    crop_x = (roi[0], roi[0] + roi[2])
    crop_y = (roi[1], roi[1] + roi[3])

    # prepare file name & index
    fname = opt["input_template"]

    if nstitch > 4:
        print("Do not support stitch > 4!!")
    elif nstitch == 3:
        # generate index of files
        ind_wf1 = opt["ind_w1"][2]
        ind_pj1 = opt["ind_pj"][0]
        ind_pj2 = opt["ind_pj"][1]
        ind_pj3 = opt["ind_pj"][2]
        ind_dk1 = opt["ind_dk"][0]

        tic = time.time()
        print("Start reading files....")
        wf1 = dxreader.read_tiff_stack(
            fname,
            ind=ind_wf1,
            digit=None,
            slc=(crop_y, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        # wf2 = dxreader.read_tiff_stack(fname, ind=ind_wf2, digit=None, slc=(crop_y, crop_x), angle=rot_angle, mblur=medfilter_size)
        pj1 = dxreader.read_tiff_stack(
            fname,
            ind=ind_pj1,
            digit=None,
            slc=(crop_y, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        pj2 = dxreader.read_tiff_stack(
            fname,
            ind=ind_pj2,
            digit=None,
            slc=(crop_y, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        pj3 = dxreader.read_tiff_stack(
            fname,
            ind=ind_pj3,
            digit=None,
            slc=(crop_y, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        dk1 = dxreader.read_tiff_stack(
            fname,
            ind=ind_dk1,
            digit=None,
            slc=(crop_y, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        print("Done. %.4f sec" % (time.time() - tic))

        # normalize image first before stitching
        projn = list()
        projn.append(tomopy.normalize(pj1, wf1, dk1))
        projn.append(tomopy.normalize(pj2, wf1, dk1))
        projn.append(tomopy.normalize(pj3, wf1, dk1))
        # print(projn[0].shape, projn[1].shape)

        shift = opt["shift"].astype(np.float32)
        stroi = opt["stroi"].astype(np.float32)
        psroi = opt["psroi"]

        # determine ROI for post stitching images
        if rec_type != "full":
            # sino = (50, roi[3] - 50 - psroi[2] - psroi[3] + 1, (roi[3] - 100 - - psroi[2] - psroi[3]) // 2)
            sino = (
                150,
                roi[3] - 150 - psroi[2] - psroi[3] + 1,
                (roi[3] - 300 - -psroi[2] - psroi[3]) // 2,
            )
        else:
            sino = None

        tic = time.time()
        print("Start stitching files....")
        # proj = imu.stitcher(projn, shift, stroi, axis=0, slc=(307, 308))
        proj = imu.stitcher(projn, shift, stroi, axis=0, slc=sino, psroi=psroi)
        # proj = imu.stitcher(projn, shift, stroi, axis=0)
        print("Done. %.4f sec" % (time.time() - tic))

    elif nstitch == 2:
        # generate index of files
        ind_wf1 = opt["ind_w1"][0]
        ind_wf2 = opt["ind_w1"][1]
        ind_pj1 = opt["ind_pj"][0]
        ind_pj2 = opt["ind_pj"][1]
        ind_dk1 = opt["ind_dk"][0]

        tic = time.time()
        print("Start reading files....")
        wf1 = dxreader.read_tiff_stack(
            fname,
            ind=ind_wf1,
            digit=None,
            slc=(crop_y, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        wf2 = dxreader.read_tiff_stack(
            fname,
            ind=ind_wf2,
            digit=None,
            slc=(crop_y, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        pj1 = dxreader.read_tiff_stack(
            fname,
            ind=ind_pj1,
            digit=None,
            slc=(crop_y, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        pj2 = dxreader.read_tiff_stack(
            fname,
            ind=ind_pj2,
            digit=None,
            slc=(crop_y, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        dk1 = dxreader.read_tiff_stack(
            fname,
            ind=ind_dk1,
            digit=None,
            slc=(crop_y, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        print("Done. %.4f sec" % (time.time() - tic))

        # check quality of the measurement
        print("Size of projection is", pj1.shape)

        # normalize image first before stitching
        projn = list()
        projn.append(tomopy.normalize(pj1, wf1, dk1))
        projn.append(tomopy.normalize(pj2, wf2, dk1))

        shift = opt["shift"].astype(np.float32)
        stroi = opt["stroi"].astype(np.float32)
        psroi = opt["psroi"]

        # determine ROI for post stitching images
        if rec_type != "full":
            # sino = (50, roi[3] - 50 - psroi[2] - psroi[3] + 1, (roi[3] - 100 - - psroi[2] - psroi[3]) // 2)
            sino = (
                100,
                roi[3] - 100 - psroi[2] - psroi[3] + 1,
                (roi[3] - 200 - -psroi[2] - psroi[3]) // 2,
            )
            # sino = (50+260, 50+281, 10)
        else:
            sino = None

        tic = time.time()
        print("Start stitching files....")
        proj = imu.stitcher(projn, shift, stroi, axis=0, slc=sino, psroi=psroi)
        print("Done. %.4f sec" % (time.time() - tic))

    else:
        # determine ROI for each projection
        if rec_type != "full":
            # sino = (roi[1] + 50, roi[1] + roi[3] - 50 + 1, (roi[3] - 100) // 2)
            sino = (roi[1] + 100, roi[1] + roi[3] - 100 + 1, (roi[3] - 200) // 2)
            # sino = (roi[1] + 161, roi[1] + 171 + 1, 5)
            # sino = (roi[1] + 100, roi[1] + 200 + 1, (200 - 100) // 2)
            # sino = (roi[1] + 66, roi[1] + 68 + 1, 1)
        else:
            sino = crop_y

        # print(type(scno_start), type(nwhite1))
        ind_white1 = opt["ind_w1"][0]
        ind_proj = opt["ind_pj"][0]
        ind_white2 = opt["ind_w2"][0]
        ind_dark = opt["ind_dk"][0]

        tic = time.time()
        print("Start reading files and rotate image %.3f degree...." % rot_angle)
        # white1 = dxreader.read_tiff_stack(fname, ind=ind_white1, digit=None, slc=(sino, crop_x), angle=rot_angle, mblur=medfilter_size)
        proj = dxreader.read_tiff_stack(
            fname,
            ind=ind_proj,
            digit=None,
            slc=(sino, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        white2 = dxreader.read_tiff_stack(
            fname,
            ind=ind_white2,
            digit=None,
            slc=(sino, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        dark = dxreader.read_tiff_stack(
            fname,
            ind=ind_dark,
            digit=None,
            slc=(sino, crop_x),
            angle=rot_angle,
            mblur=medfilter_size,
        )
        proj[proj == 0] = 1
        print("Done. %.4f sec" % (time.time() - tic))

        # check quality of the measurement
        print("Size of projection is", proj.shape)

        # Flat-field correction of raw data.
        tic = time.time()
        print("Normalization....")
        proj = tomopy.normalize(proj, white2, dark)
        print("Done. %.4f sec" % (time.time() - tic))

    # Set data collection angles as equally spaced between theta_start ~ theta_end (in degrees.)
    theta = tomopy.angles(proj.shape[0], ang1=th_start, ang2=th_end)

    # Ring removal.
    tic = time.time()
    print("Apply Ring removal filter...")
    # proj = tomopy.remove_stripe_fw(proj)      # pretty effective, bg distortion
    proj = tomopy.remove_stripe_ti(proj)  # pretty effective, bg distortion
    # proj = tomopy.remove_stripe_sf(proj)      # less useful, but won't distort background
    print("Done. %.4f sec" % (time.time() - tic))

    tic = time.time()
    print("Calculate minus_log of projection...")
    proj = tomopy.minus_log(proj)
    print("Done. %.4f sec" % (time.time() - tic))

    print(proj.shape, proj.dtype, proj.max(), proj.min())

    print("Create & Cleanup destination folder...")
    # create output folder if not exist
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    else:
        # clean folder if requested
        if clean_up_folder:
            filelist = os.listdir(outpath)
            for files in filelist:
                os.remove(os.path.join(outpath, files))

    # debug
    # dxwriter.write_tiff_stack(proj[:,:,:], fname= outpath + 'sinogram')

    if rec_type != "full":
        cen = tomopy.find_center_pc(proj[0], proj[900], tol=0.5)  # for theta step = 0.2
        # cen = tomopy.find_center_pc(proj[0], proj[720], tol=0.5)    # for theta step = 0.25
        print("Expected rotation center: ", cen - proj.shape[2] / 2)

        # cv2.namedWindow('Check Proj', cv2.WINDOW_NORMAL)
        # cv2.selectROI('Check Proj', proj[:,0,:], True)

        print(
            "Do 3 slices, ",
            list(range(sino[0], sino[1], sino[2])),
            ", to find Rot center",
        )
        print(postrecroi)
        if postrecroi != None:
            recdimx = postrecroi[2]
            recdimy = postrecroi[3]
        else:
            recdimx = abs(proj.shape[2])
            recdimy = abs(proj.shape[2])

        if show_rec_img:
            rectop = np.zeros((recdimy, recdimx, 21), np.float32)
            recmid = np.zeros((recdimy, recdimx, 21), np.float32)
            recbot = np.zeros((recdimy, recdimx, 21), np.float32)

        if focochkroi != None:
            xs = focochkroi[0]
            xe = focochkroi[0] + focochkroi[2]
            ys = focochkroi[1]
            ye = focochkroi[1] + focochkroi[3]
        else:
            xs = 0
            xe = recdimx
            ys = 0
            ye = recdimy

        _, nslice, projw = proj.shape
        focus_score = np.zeros((4, 21))
        i = 0
        tic = time.time()
        print("rotate reconstructed image", postrecrot, "degree")
        for rotcenter in np.linspace(
            rot_center - rc_step * 10, rot_center + rc_step * 10, 21
        ):
            # proj[:, :, int(projw/2 + rotcenter):] = 0       # block right of rot_ax
            rec = tomopy.recon(
                proj, theta, center=proj.shape[2] / 2 + rotcenter, algorithm="gridrec"
            )
            rec = rec * 10e7

            # rotate the image (unbounded, i.e image will trim outside original size)
            if not (postrecrot == None or postrecrot == 0):
                # grab the dimensions of the image and then determine the center
                (nimg, h, w) = rec.shape[:]
                (cX, cY) = (w // 2, h // 2)
                # grab the rotation matrix, then grab the sine and cosine
                M = cv2.getRotationMatrix2D((cX, cY), postrecrot, 1)
                # perform the actual rotation and return the image
                for k in range(0, nimg):
                    rec[k, :, :] = cv2.warpAffine(rec[k, :, :], M, (w, h))

            # crop the reconstructed image
            if postrecroi != None:
                rec = rec[
                    :,
                    postrecroi[1] : postrecroi[1] + postrecroi[3],
                    postrecroi[0] : postrecroi[0] + postrecroi[2],
                ]

            # calculate blurness of the image using "variation of the Laplacian"
            focus_score[0, i] = rotcenter
            focus_score[1, i] = cv2.Laplacian(rec[0, ys:ye, xs:xe], cv2.CV_32F).var()
            focus_score[2, i] = cv2.Laplacian(rec[1, ys:ye, xs:xe], cv2.CV_32F).var()
            focus_score[3, i] = cv2.Laplacian(rec[2, ys:ye, xs:xe], cv2.CV_32F).var()
            print(
                "Center of Rot[%2d]: %.2f, %.1f, %.1f, %.1f,"
                % (
                    i,
                    rotcenter,
                    focus_score[1, i],
                    focus_score[2, i],
                    focus_score[3, i],
                )
            )

            dxwriter.write_tiff_stack(rec, fname=outpath + fprefix)
            if show_rec_img:
                rectop[:, :, i] = rec[0, :, :]
                recmid[:, :, i] = rec[1, :, :]
                recbot[:, :, i] = rec[2, :, :]
            i = i + 1

        # plot blurness curve to check rotation axis location
        import matplotlib.pyplot as plt

        rotcen = focus_score[0, :]
        fscore = focus_score[1:, :].T
        fmax = fscore.max(axis=0, keepdims=True)

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        fig, ax = plt.subplots(3, sharex=True)
        ax[0].plot(
            rotcen, fscore[:, 0], "o-", color=colors[0], label="%d (+0)" % (sino[0])
        )
        ax[0].grid(True, ls="--")
        ax[0].legend(loc=0)
        ax[0].set_title("image rotation is %2.3f deg" % rot_angle)
        ax[1].plot(
            rotcen,
            fscore[:, 1],
            "o-",
            color=colors[1],
            label="%d (+%d)" % (sino[0] + sino[2], sino[2]),
        )
        ax[1].grid(True, ls="--")
        ax[1].legend(loc=0)
        ax[2].plot(
            rotcen,
            fscore[:, 2],
            "o-",
            color=colors[2],
            label="%d (+%d)" % (sino[0] + sino[2] * 2, sino[2] * 2),
        )
        ax[2].grid(True, ls="--")
        ax[2].legend(loc=0)
        ax[2].set_xlabel("rotation axis (px)")
        fig.suptitle("rotation axis offset", fontsize=17)

        fig2, ax2 = plt.subplots()
        ax2.plot(rotcen, fscore / fmax, "o-")
        ax2.grid(True, ls="--")
        ax2.legend(("top", "mid", "bot"), loc=0)
        ax2.set_xlabel("rotation axis (px)")
        ax2.set_ylabel("normalization")

        print("Done. %.4f sec" % (time.time() - tic))
        plt.show()

        if show_rec_img:
            # convert the data to uint16 and normalize to its MIN/MAX or clim
            rectop = cv2.normalize(
                rectop,
                None,
                alpha=-clim[0] * c_scaling16,
                beta=(65535 - clim[0]) * c_scaling16,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_16U,
            )
            recmid = cv2.normalize(
                recmid,
                None,
                alpha=-clim[0] * c_scaling16,
                beta=(65535 - clim[0]) * c_scaling16,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_16U,
            )
            recbot = cv2.normalize(
                recbot,
                None,
                alpha=-clim[0] * c_scaling16,
                beta=(65535 - clim[0]) * c_scaling16,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_16U,
            )

            # recmidn = np.einsum('ijk->kij',recmidn)
            # dxwriter.write_tiff_stack(recmid, fname=outpath + 'test')

            window1 = imu.PanZoomWindow(rectop, "top layer")
            window2 = imu.PanZoomWindow(recmid, "middle layer")
            window3 = imu.PanZoomWindow(recbot, "bottom layer")

            print("User 'q' or ESC to quite the window.")
            key = -1
            while key != ord("q") and key != 27:  # 27 = escape key
                key = cv2.waitKey(5)  # User can press 'q' or ESC to exit
            cv2.destroyAllWindows()

    else:
        print("Do full volume reconstruction with Rot_center = %f" % rot_center)

        # Reconstruct object using Gridrec algorithm.
        tic = time.time()
        print("Start reconstruction....")
        recs = tomopy.recon(
            proj, theta, center=proj.shape[2] / 2 + rot_center, algorithm="gridrec"
        )
        print("Done. %.4f sec" % (time.time() - tic))

        # rotate the image (unbounded, i.e image will trim outside original size)
        if not (postrecrot == None or postrecrot == 0):
            # grab the dimensions of the image and then determine the center
            (nimg, h, w) = recs.shape[:]
            (cX, cY) = (w // 2, h // 2)
            # grab the rotation matrix, then grab the sine and cosine
            M = cv2.getRotationMatrix2D((cX, cY), postrecrot, 1)
            # perform the actual rotation and return the image
            for k in range(0, nimg):
                recs[k, :, :] = cv2.warpAffine(recs[k, :, :], M, (w, h))

        # Mask each reconstructed slice with a circle.
        # recs = tomopy.circ_mask(recs, axis=0, ratio=0.95)

        # crop the reconstructed image
        if postrecroi != None:
            recs = recs[
                :,
                postrecroi[1] : postrecroi[1] + postrecroi[3],
                postrecroi[0] : postrecroi[0] + postrecroi[2],
            ]

        # recs = recs * 10E7

        # Write data as stack of TIFFs.
        tic = time.time()
        print("Start saving files....")
        dxwriter.write_tiff_stack(recs, fname=outpath + fprefix)
        print("Done. %.4f sec" % (time.time() - tic))
