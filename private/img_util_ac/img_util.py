#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modules related to image visualization & tomo image pre-processing
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import cv2
import numpy as np
import yaml as yml
import os
from argparse import ArgumentParser as parser

logger = logging.getLogger(__name__)

__author__ = "Andrew Chuang"
__copyright__ = "Copyright (c) 2018, Andrew Chuang"
__docformat__ = "restructuredtext en"
__all__ = [
    "PanZoomWindow",
    "PanAndZoomState",
    "rotate",
    "rotate_bound",
    "find_translation",
]


class PanZoomWindow(object):
    """
    https://stackoverflow.com/questions/28595958/creating-trackbars-to-scroll-large-image-in-opencv-python

    Controls an OpenCV window. Registers a mouse listener so that:
        1. right-dragging up/down zooms in/out
        2. right-clicking re-centers
        3. trackbars scroll vertically and horizontally
    You can open multiple windows at once if you specify different window names.
    You can pass in an onLeftClickFunction, and when the user left-clicks, this
    will call onLeftClickFunction(y,x), with y,x in original image coordinates.
    """

    def __init__(self, img, windowName="PanZoomWindow", onLeftClickFunction=None):
        self.WINDOW_NAME = windowName
        self.H_TRACKBAR_NAME = "x"
        self.V_TRACKBAR_NAME = "y"
        self.Z_TRACKBAR_NAME = "frame"
        self.img = img
        self.onLeftClickFunction = onLeftClickFunction
        self.TRACKBAR_TICKS = 1000
        self.panAndZoomState = PanAndZoomState(img.shape, self)
        self.lButtonDownLoc = None
        self.mButtonDownLoc = None
        self.rButtonDownLoc = None
        self.Z_selected = 0
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.redrawImage()
        cv2.setMouseCallback(self.WINDOW_NAME, self.onMouse)
        cv2.createTrackbar(
            self.H_TRACKBAR_NAME,
            self.WINDOW_NAME,
            0,
            self.TRACKBAR_TICKS,
            self.onHTrackbarMove,
        )
        cv2.createTrackbar(
            self.V_TRACKBAR_NAME,
            self.WINDOW_NAME,
            0,
            self.TRACKBAR_TICKS,
            self.onVTrackbarMove,
        )
        if img.ndim > 2:

            def nothing(x):
                pass

            self.TRACKBAR_ZTICKS = img.shape[2] - 1
            cv2.createTrackbar(
                self.Z_TRACKBAR_NAME,
                self.WINDOW_NAME,
                0,
                self.TRACKBAR_ZTICKS,
                self.onZTrackbarMove,
            )

    def onMouse(self, event, x, y, _ignore1, _ignore2):
        """ Responds to mouse events within the window.
        The x and y are pixel coordinates in the image currently being displayed.
        If the user has zoomed in, the image being displayed is a sub-region, so you'll need to
        add self.panAndZoomState.ul to get the coordinates in the full image."""
        if event == cv2.EVENT_MOUSEMOVE:
            return
        elif event == cv2.EVENT_RBUTTONDOWN:
            # record where the user started to right-drag
            self.mButtonDownLoc = np.array([y, x])
        elif event == cv2.EVENT_RBUTTONUP and self.mButtonDownLoc is not None:
            # the user just finished right-dragging
            dy = y - self.mButtonDownLoc[0]
            pixelsPerDoubling = 0.2 * self.panAndZoomState.shape[0]  # lower = zoom more
            changeFactor = 1.0 + abs(dy) / pixelsPerDoubling
            changeFactor = min(max(1.0, changeFactor), 5.0)
            if changeFactor < 1.05:
                dy = 0  # this was a click, not a draw. So don't zoom, just re-center.
            if dy > 0:  # moved down, so zoom out.
                zoomInFactor = 1.0 / changeFactor
            else:
                zoomInFactor = changeFactor
            #            print "zoomFactor:",zoomFactor
            self.panAndZoomState.zoom(
                self.mButtonDownLoc[0], self.mButtonDownLoc[1], zoomInFactor
            )
        elif event == cv2.EVENT_LBUTTONDOWN:
            # the user pressed the left button.
            coordsInDisplayedImage = np.array([y, x])
            if np.any(coordsInDisplayedImage < 0) or np.any(
                coordsInDisplayedImage > self.panAndZoomState.shape[:2]
            ):
                print("you clicked outside the image area")
            else:
                print(
                    "you clicked on",
                    coordsInDisplayedImage,
                    "within the zoomed rectangle",
                )
                coordsInFullImage = self.panAndZoomState.ul + coordsInDisplayedImage
                print("this is", coordsInFullImage, "in the actual image")
                if len(self.img.shape) == 2:
                    print(
                        "this pixel holds ",
                        self.img[int(coordsInFullImage[0]), int(coordsInFullImage[1])],
                    )
                else:
                    print(
                        "this pixel holds ",
                        self.img[
                            int(coordsInFullImage[0]),
                            int(coordsInFullImage[1]),
                            self.Z_selected,
                        ],
                    )

                if self.onLeftClickFunction is not None:
                    self.onLeftClickFunction(coordsInFullImage[0], coordsInFullImage[1])
        # you can handle other mouse click events here

    def onVTrackbarMove(self, tickPosition):
        self.panAndZoomState.setYFractionOffset(
            float(tickPosition) / self.TRACKBAR_TICKS
        )

    def onHTrackbarMove(self, tickPosition):
        self.panAndZoomState.setXFractionOffset(
            float(tickPosition) / self.TRACKBAR_TICKS
        )

    def onZTrackbarMove(self, tickPosition):
        self.Z_selected = tickPosition
        frame_selected = self.img[:, :, tickPosition]
        pzs = self.panAndZoomState
        cv2.imshow(
            self.WINDOW_NAME,
            frame_selected[
                int(pzs.ul[0]) : int(pzs.ul[0] + pzs.shape[0]),
                int(pzs.ul[1]) : int(pzs.ul[1] + pzs.shape[1]),
            ],
        )
        # cv2.resizeWindow(self.WINDOW_NAME, 600, 800)
        # print("select from: ", tickPosition)

    def redrawImage(self):
        if self.img.ndim == 2:
            pzs = self.panAndZoomState
            cv2.imshow(
                self.WINDOW_NAME,
                self.img[
                    int(pzs.ul[0]) : int(pzs.ul[0] + pzs.shape[0]),
                    int(pzs.ul[1]) : int(pzs.ul[1] + pzs.shape[1]),
                ],
            )
            cv2.resizeWindow(self.WINDOW_NAME, 600, 800)
        else:
            frame_no = self.Z_selected
            frame_selected = self.img[:, :, frame_no]
            img_ratio = frame_selected.shape[0] / frame_selected.shape[1]
            # print(frame_selected.shape, frame_selected.min(), frame_selected.max())

            pzs = self.panAndZoomState
            cv2.imshow(
                self.WINDOW_NAME,
                frame_selected[
                    int(pzs.ul[0]) : int(pzs.ul[0] + pzs.shape[0]),
                    int(pzs.ul[1]) : int(pzs.ul[1] + pzs.shape[1]),
                ],
            )
            cv2.resizeWindow(self.WINDOW_NAME, 700, int(700 * img_ratio + 150))


class PanAndZoomState(object):
    """ Tracks the currently-shown rectangle of the image.
    Does the math to adjust this rectangle to pan and zoom."""

    MIN_SHAPE = np.array([50, 50])

    def __init__(self, imShape, parentWindow):
        self.ul = np.array(
            [0, 0]
        )  # upper left of the zoomed rectangle (expressed as y,x)
        self.imShape = np.array(imShape[0:2])
        self.shape = self.imShape  # current dimensions of rectangle
        self.parentWindow = parentWindow

    def zoom(self, relativeCy, relativeCx, zoomInFactor):
        self.shape = (self.shape.astype(np.float) / zoomInFactor).astype(np.int)
        # expands the view to a square shape if possible. (I don't know how to get the actual window aspect ratio)
        self.shape[:] = np.max(self.shape)
        self.shape = np.maximum(
            PanAndZoomState.MIN_SHAPE, self.shape
        )  # prevent zooming in too far
        c = self.ul + np.array([relativeCy, relativeCx])
        self.ul = c - self.shape / 2
        self._fixBoundsAndDraw()

    def _fixBoundsAndDraw(self):
        """ Ensures we didn't scroll/zoom outside the image.
        Then draws the currently-shown rectangle of the image."""
        #        print "in self.ul:",self.ul, "shape:",self.shape
        self.ul = np.maximum(0, np.minimum(self.ul, self.imShape - self.shape))
        self.shape = np.minimum(
            np.maximum(PanAndZoomState.MIN_SHAPE, self.shape), self.imShape - self.ul
        )
        #        print "out self.ul:",self.ul, "shape:",self.shape
        yFraction = float(self.ul[0]) / max(1, self.imShape[0] - self.shape[0])
        xFraction = float(self.ul[1]) / max(1, self.imShape[1] - self.shape[1])
        cv2.setTrackbarPos(
            self.parentWindow.H_TRACKBAR_NAME,
            self.parentWindow.WINDOW_NAME,
            int(xFraction * self.parentWindow.TRACKBAR_TICKS),
        )
        cv2.setTrackbarPos(
            self.parentWindow.V_TRACKBAR_NAME,
            self.parentWindow.WINDOW_NAME,
            int(yFraction * self.parentWindow.TRACKBAR_TICKS),
        )
        self.parentWindow.redrawImage()

    def setYAbsoluteOffset(self, yPixel):
        self.ul[0] = min(max(0, yPixel), self.imShape[0] - self.shape[0])
        self._fixBoundsAndDraw()

    def setXAbsoluteOffset(self, xPixel):
        self.ul[1] = min(max(0, xPixel), self.imShape[1] - self.shape[1])
        self._fixBoundsAndDraw()

    def setYFractionOffset(self, fraction):
        """ pans so the upper-left zoomed rectange is "fraction" of the way down the image."""
        self.ul[0] = int(round((self.imShape[0] - self.shape[0]) * fraction))
        self._fixBoundsAndDraw()

    def setXFractionOffset(self, fraction):
        """ pans so the upper-left zoomed rectange is "fraction" of the way right on the image."""
        self.ul[1] = int(round((self.imShape[1] - self.shape[1]) * fraction))
        self._fixBoundsAndDraw()


def rotate(image, angle=None, scaling=None):
    """
    Rotate image (maintain the original dimension, will trim image due to rotation)

    :param img:
    :return:
    """
    # this is adapted from https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

    if scaling is None:
        scaling = 1.0

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix, then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, scaling)

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (w, h))


def rotate_bound(image, angle=None, scaling=None):
    """
    Rotate image (maintain the original shape without trimming due to rotation)

    :param img:
    :return:
    """
    if scaling is None:
        scaling = 1.0

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix, then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, scaling)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def find_translation(imgA, imgB, method=3, roiB=None, roiA=None, show=None):
    """
    Find the translation of ImageB with respect to ImageA based on ROI in imageB using cross correlation

    :param img:
        imgA: target
        imgB: template
        method: cross correlation method defined in OpenCV (default: 3 (cv2.TM_CCORR_NORMED))
        roiA: select ROI in imgA if want to limit search area
        roiB: select ROI in imgB to be used as template


    :return:
    """
    # grab size of image
    (ha, wa) = imgA.shape[:2]
    (hb, wb) = imgB.shape[:2]

    if roiA is None:
        (a_top_left_x, a_top_left_y, a_w, a_h) = (
            0,
            0,
            wa,
            ha,
        )  # (top_left_x, top_left_y, width, height)
    else:
        (a_top_left_x, a_top_left_y, a_w, a_h) = (roiA[0], roiA[1], roiA[2], roiA[3])

    if roiB is None:
        (b_top_left_x, b_top_left_y, b_w, b_h) = (
            0,
            0,
            wb,
            hb,
        )  # (top_left_x, top_left_y, width, height)
    else:
        (b_top_left_x, b_top_left_y, b_w, b_h) = (roiB[0], roiB[1], roiB[2], roiB[3])

    template = imgB[
        b_top_left_y : (b_top_left_y + b_h), b_top_left_x : (b_top_left_x + b_w)
    ]

    # All the 6 methods for comparison in a list
    methods = [
        "cv2.TM_CCOEFF",
        "cv2.TM_CCOEFF_NORMED",
        "cv2.TM_CCORR",
        "cv2.TM_CCORR_NORMED",
        "cv2.TM_SQDIFF",
        "cv2.TM_SQDIFF_NORMED",
    ]

    match_algorithm = eval(methods[method])

    # Apply template Matching
    res = cv2.matchTemplate(
        imgA[a_top_left_y : a_top_left_y + a_h, a_top_left_x : a_top_left_x + a_w],
        template,
        match_algorithm,
    )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        val = min_val
        res = abs(res - res.max())
    else:
        top_left = max_loc
        val = max_val

    # bottom_right = (top_left[0] + b_w, top_left[1] + b_h)

    weight_range = 5  # 5~8 seems to be a good range
    M = cv2.moments(
        res[
            top_left[1] - weight_range : top_left[1] + weight_range,
            top_left[0] - weight_range : top_left[0] + weight_range,
        ]
    )
    if M["m00"] == 0:
        cxf = 0
        cyf = 0
        print("M[m00] is zero. Possibility bad match")
    else:
        cxf = round(M["m10"] / M["m00"] - weight_range + 0.5, 3)
        cyf = round(M["m01"] / M["m00"] - weight_range + 0.5, 3)
    cx = top_left[0] + a_top_left_x
    cy = top_left[1] + a_top_left_y
    # shift is based on size of imgB (not size of roiB)
    shift = (cx - b_top_left_x, cy - b_top_left_y, cxf, cyf)

    if show:
        print("calculated shift (x, y) (coarse/fine): {}".format(shift))

        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Template", cv2.WINDOW_NORMAL)
        cv2.namedWindow(methods[method], cv2.WINDOW_NORMAL)

        # convert image to uint8 and normalize to MIN/MAX
        imgAn = cv2.normalize(imgA, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        imgBn = cv2.normalize(imgB, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        resn = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # draw roiA
        imgA1 = cv2.rectangle(
            imgAn,
            (a_top_left_x, a_top_left_y),
            (a_top_left_x + a_w, a_top_left_y + a_h),
            255,
            1,
        )
        imgA1 = cv2.putText(
            imgA1,
            "Searched Area",
            (a_top_left_x, a_top_left_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            255,
            1,
        )
        # draw matched location
        imgA1 = cv2.rectangle(imgA1, (cx, cy), (cx + b_w, cy + b_h), 255, 2)
        imgA1 = cv2.putText(
            imgA1, "Matched Area", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 1
        )
        # draw roiB
        imgB1 = cv2.rectangle(
            imgBn,
            (b_top_left_x, b_top_left_y),
            (b_top_left_x + b_w, b_top_left_y + b_h),
            255,
            1,
        )
        imgB1 = cv2.putText(
            imgB1,
            "Template",
            (b_top_left_x, b_top_left_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            255,
            1,
        )

        cv2.imshow("Original", imgA1)
        cv2.imshow("Template", imgB1)
        cv2.imshow(methods[method], resn)

        key = -1
        while key != ord("q") and key != 27:  # 27 = escape key
            key = cv2.waitKey(1)  # User can press 'q' or ESC to exit.
        cv2.destroyAllWindows()

    return shift


def get_roi(shift, dim, gap=(0, 0, 0, 0)):
    """
    find the over-lappying region of the image based on shift
    top-left (0, 0), x(+) to the right, y(+) goes down


    :param shift:
        (shift_x, shift_y) positive integer
    :param dim:
        (w , h) positive integer
    :param gap:
        (left, right, top, bottom) positive integer
    :return:

        ( x1, y1, x2, y2, w, h)

        x1, y1 top-left corner on original frame
        x2, y2 top-left corner on shifted frame
        w      width of ROI
        h      height of ROI
    """
    # dimansion of the image
    w0 = dim[1]
    h0 = dim[0]

    # shift of the image
    sx = int(shift[0])
    sy = int(shift[1])

    # gap of the ROI to the over-lapped area
    gl = abs(int(gap[0]))
    gr = abs(int(gap[1]))
    gt = abs(int(gap[2]))
    gd = abs(int(gap[3]))

    # coordinate and size of ROI
    w = w0 - abs(sx) - gl - gr
    h = h0 - abs(sy) - gt - gd

    w = max(0, w)
    h = max(0, h)

    if sx >= 0:
        x1 = gl + sx
        x2 = gl
    else:
        x1 = gl
        x2 = gl + abs(sx)

    if sy >= 0:
        y1 = gt + sy
        y2 = gt
    else:
        y1 = gt
        y2 = gt + abs(sy)

    output = (x1, y1, x2, y2, w, h)
    return output


def stitcher(imgstk, shifts, rois, axis=0, slc=None, psroi=[0, 0, 0, 0]):
    """
    stitch images together based on the information in shift

    :param imgstk:
        list of images,  imgstk = list(imgA, imgB,.., imgN)
    :param shifts:
        list of shifts,  shifts = list(shiftA, shiftB, ..., shiftN)
        shiftN = (x, y)
    :return:
    """
    # print("insid log start:")

    if axis not in (0, 2):
        print("axis can only be 0 or 2 !!")
        return

    # number of images to stitch
    (nvol, Null) = shifts.shape

    # image stack format should be [nframe, rows, cols]
    if axis == 2:
        for i in range(0, nvol):
            imgstk[i] = np.einsum("ijk->kij", imgstk[i])

    # determine stitched image size
    arr_sz = np.zeros((nvol, 3))
    for i in range(0, nvol):
        arr_sz[i, :] = imgstk[i].shape

    dim = arr_sz.max(axis=0)

    # print(shifts)
    # print(dim)

    fw = int(abs(shifts[:, 0]).max() * 2 + dim[2] + 20)
    fh = int(abs(shifts[:, 1]).max() * 2 + dim[1] + 20)
    nframe = int(dim[0])

    # construct array
    arr = np.zeros((nframe, fh, fw), dtype=np.float32)

    for i in range(0, nframe):  # loop over nframe
        for j in range(0, nvol):  # loop over nvol
            img = imgstk[j][i, :, :]

            # pad image with frame (increase size)
            # img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=1)
            # pad image with frame (maintain size)
            # img = cv2.rectangle(img, (0, 0), (0 + w, 0 + h), 1, 2)

            (h, w) = img.shape

            # shift the image based on original input size (before ROI)
            shiftx = (fw - w) // 2 + shifts[j][0]
            shifty = (fh - h) // 2 + shifts[j][1]

            # construct rotation matrix
            # M = cv2.getRotationMatrix2D((w //2, h // x), 0, 2)    # for rotation
            M = np.float32([[1, 0, shiftx], [0, 1, shifty]])

            # perform the actual translation(or rotation) and return the image
            img1 = cv2.warpAffine(img, M, (fw, fh), flags=cv2.INTER_AREA)

            # apply ROI
            roi_x = int(rois[j][0])
            roi_y = int(rois[j][1])
            roi_w = int(rois[j][2])
            roi_h = int(rois[j][3])

            if (roi_x + roi_w) > w:
                print("bad roi, roi_x larger than image_x")
                return

            if (roi_y + roi_h) > h:
                print("bad roi, roi_y larger than image_y")
                return

            # Now label the roi region
            img1 = cv2.rectangle(
                img1,
                (int(shiftx // 1 + roi_x - 1), int(shifty // 1 + roi_y - 1)),
                (
                    int(shiftx // 1 + roi_x + roi_w + 1),
                    int(shifty // 1 + roi_y + roi_h + 1),
                ),
                2,
                1,
            )
            img1 = cv2.putText(
                img1,
                "ROI",
                (int(shiftx // 1 + roi_x - 1), int(shifty // 1 + roi_y - 1)),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                1,
            )

            # Now copy the transformed image to final array
            arr[
                i,
                int(shifty // 1 + roi_y) : int(shifty // 1 + roi_y + roi_h),
                int(shiftx // 1 + roi_x) : int(shiftx // 1 + roi_x + roi_w),
            ] = img1[
                int(shifty // 1 + roi_y) : int(shifty // 1 + roi_y + roi_h),
                int(shiftx // 1 + roi_x) : int(shiftx // 1 + roi_x + roi_w),
            ]

    if slc is not None:
        if len(slc) == 3:
            _slc = list(np.arange(slc[0], slc[1], slc[2]) + 10 + psroi[2])
        else:
            _slc = list(np.arange(slc[0], slc[1]) + 10 + psroi[2])
        arr = arr[:, _slc, 10 + psroi[0] : -(10 + psroi[1])]
    else:
        arr = arr[:, 10 + psroi[2] : -(10 + psroi[3]), 10 + psroi[0] : -(10 + psroi[1])]

    # return arr, img1
    return arr


def tomo_par_init(cfg_name, scid):
    """

    Initialize parameters for reconstruction from experimental file/log

    :param:
        cfg_name: experiment log
            scid: Scan ID defined in experiment log

    :return:
        par: output parameter file
    """
    try:
        cfg = yml.load(open(cfg_name, "r"))[scid]

    except IOError:
        print('Parameter file "%s" is missing or cannot be read.' % (cfg_name))
        return None

    except KeyError:
        print('"%s" is not a correct Scan_ID.' % (scid))
        return None

    print("parameter file successfully loaded!!")

    # Copy necessary keys to output
    # par = {k: cfg[k] for k in ('Notes', 'm', 'n') if k in cfg}   # store only key with values
    par = {
        k: cfg.get(k, None)
        for k in (  # key without values will be assigned "None"
            "Notes",
            "prefix",
            "rot_axis",
            "nstitch",
            "nproj",
            "psroi",
            "roi",
            "post_recon_roi",
            "post_recon_rot",
            "focus_check_roi",
        )
    }

    # default assignment if not available
    par["path_output"] = cfg.get("path_output", "recon_dir/")
    par["rot_angle"] = cfg.get("rot_angle", 0)
    par["clim"] = cfg.get("clim", [0, 65535])

    # default value if not assigned (TO-DO)

    # create file template
    fname = (
        os.path.join(cfg["path_input"], cfg["prefix"])
        + "_"
        + str(1).zfill(cfg["digit"])
        + ".tif"
    )
    par["input_template"] = fname

    # par['roi'] = [cfg['roi_x'][0], cfg['roi_y'][0], cfg['roi_x'][1], cfg['roi_y'][1]]  # [top_left_x, top_left_y, width, height)
    par["theta"] = [
        float(cfg["theta_start"]),
        cfg["theta_start"] + cfg["theta_step"] * (cfg["nproj"] - 1),
    ]

    if par["nstitch"] > 1:
        par["stroi"] = np.asarray(cfg["stroi"])
        par["shift"] = np.asarray(cfg["shift"])
    else:
        par["stroi"] = None
        par["shift"] = None

    # calculate file index
    par["ind_w1"] = list()
    par["ind_pj"] = list()
    par["ind_w2"] = list()
    par["ind_dk"] = list()

    for i in range(0, cfg["nstitch"]):
        scno_start = cfg["scno"][i]
        par["ind_w1"].append(list(range(scno_start, scno_start + cfg["nwhite1"])))
        par["ind_pj"].append(
            list(
                range(
                    scno_start + cfg["nwhite1"],
                    scno_start + cfg["nwhite1"] + cfg["nproj"],
                )
            )
        )
        par["ind_w2"].append(
            list(
                range(
                    scno_start + cfg["nwhite1"] + cfg["nproj"],
                    scno_start + cfg["nwhite1"] + cfg["nproj"] + cfg["nwhite2"],
                )
            )
        )
        par["ind_dk"].append(
            list(
                range(
                    scno_start + cfg["nwhite1"] + cfg["nproj"] + cfg["nwhite2"],
                    scno_start
                    + cfg["nwhite1"]
                    + cfg["nproj"]
                    + cfg["nwhite2"]
                    + cfg["ndark"],
                )
            )
        )

    return par


def find_roi(opt):
    """
    Use bright field to determine ROI (crop region)

    :param opt:
    output of tomo_par_init

    :return roi:
    roi = (x, y, w, h)
    """
    import dxchange.reader as dxread

    ind_to_read = [opt["ind_w1"][0][0]]
    roi = opt["roi"]
    rot_angle = opt["rot_angle"]
    medfilter_size = None
    fname = opt["input_template"]

    imgw = dxread.read_tiff_stack(
        fname,
        ind=ind_to_read,
        digit=None,
        slc=None,
        angle=rot_angle,
        mblur=medfilter_size,
    )
    imgn = cv2.normalize(
        imgw[0, :, :],
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    [ch, cw] = imgn.shape

    if roi[2] < cw or roi[3] < ch or roi[0] != 0 or roi[1] != 0:
        cv2.rectangle(
            imgn,
            (roi[0], roi[1]),
            (roi[0] + roi[2], roi[1] + roi[3]),
            255,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            imgn,
            "current ROI(x,y,w,h): " + ",".join(map(str, roi)),
            (roi[0], roi[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            255,
            2,
            cv2.LINE_AA,
        )

    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    (x, y, w, h) = cv2.selectROI("Select ROI", imgn, True)
    if x + y + w + h == 0:
        print("ROI remains the same!!")
        return roi
    else:
        print("New ROI is ", x, y, w, h)
        return x, y, w, h
