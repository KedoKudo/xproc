#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Provide functions that operate on projections
"""

import itertools
import multiprocessing
import tomopy
import numpy                     as     np
import scipy                     as     sp
import concurrent.futures        as     cf

from   typing                    import Tuple
from   scipy.signal              import medfilt
from   scipy.signal              import medfilt2d
from   scipy.ndimage             import gaussian_filter
from   scipy.ndimage             import gaussian_filter1d
from   scipy.spatial.distance    import pdist
from   scipy.spatial.distance    import squareform
from   skimage                   import exposure
from   skimage.transform         import probabilistic_hough_line
from   skimage.feature           import canny
from   skimage.feature           import register_translation
from   sklearn.cluster           import KMeans
from   tifffile                  import imread
from   lmfit.models              import GaussianModel
from   tomopy                    import minus_log
from   tomopy                    import find_center_pc
from   tomoproc.util.npmath      import rescale_image
from   tomoproc.util.peakfitting import fit_sigmoid
from   tomoproc.util.npmath      import rescale_image
from   tomoproc.util.npmath      import binded_minus_log


def detect_sample_in_sinogram(
    sino: np.ndarray,
    kernel_size: int=3,
    sigma: int=50,
    minimum_distance_to_edge: int=5,
    ) -> Tuple[int, int]:
    """
    Description
    -----------
        Automatically detect the left and right edge of the sample region
        in a sinogram with median and gaussian filtering.

    Parameters
    ----------
    sino: np.ndarray
        Sinogram for evaluation
    kernel_size: int
        median filter (quick denoising) kernel size
    sigma: int
        gaussian filter kernel size
    minimum_distance_to_edge: int
        minimum amount of pixels to sinogram edge

    Returns
    -------
    (int, int)
        left and right edge of the sample region
    """
    # use median filter and gaussian filter to locate the sample region 
    # -- median filter is to counter impulse noise
    # -- gaussian filter is for estimating the sample location
    prf = np.gradient(
        np.sum(
            gaussian_filter(
                medfilt2d(sino, kernel_size=kernel_size), 
                sigma=sigma,
                ),
            axis=0,
            )
        )
    return  (
        max(prf.argmin(), minimum_distance_to_edge), 
        min(prf.argmax(), sino.shape[1]-minimum_distance_to_edge),
        )


def detect_corrupted_proj(
    projs: np.ndarray,
    omegas: np.ndarray,
    threshold: float=0.8,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Corrupted frames/projections will add a forgy layer (artifact) of random 
    noise to the final reconstruction.  These corrupted frames can be detected
    through 180 degree pair-wise checking.

    Parameters
    ----------
    projs: np.ndarray
        tomography image stack [axis_omega, axis_imgrow, axis_imgcol]
    omegas: np.ndarray
        angular position vector
    threshold: float
        Threshold for picking out the outliers

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
           idx_BAD     idx_GOOD
        Return the indices of BAD frames and GOOD frames/projections
    """
    # assume equal step, find the index range equals to 180 degree
    dn = int(np.pi/abs(omegas[1] - omegas[0]))

    # get the cnts from each 180 pairs
    # use the faster version instead
    with cf.ProcessPoolExecutor() as e:
        _jobs = [
            e.submit(
                tomopy.find_center_pc,
                rescale_image(binded_minus_log(projs[nimg,:,:])), 
                rescale_image(binded_minus_log(projs[nimg+dn,:,:])), 
            )
            for nimg in range(dn)
        ]

    cnts = [me.result() for me in _jobs]

    # 180 -> 360
    cnts = np.array(cnts + cnts)

    # locate outlier
    diff = np.absolute(cnts - medfilt(cnts))/cnts

    return np.where(diff>threshold)[0], np.where(diff<=threshold)[0]


def guess_slit_box(img: np.ndarray, boost: bool=True) -> dict:
    """
    Description
    -----------
    Auto detect/guess the four blades position (in pixels) for given image

    Parameters
    ----------
    img: np.ndarray
        2D tomography image with slit box

    Returns
    -------
    dict:
        dictionary contains the approximated position (in pixel) for the four
        slit blades
    
    NOTE
    ----
    For images without any slit blades, a random (probably useless) region
    will be returned.

    Relative fast:
    tested on MacBookPro13,3
    395 ms ± 14 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """    
    if boost:
        # Contrast stretching
        pl, ph = np.percentile(img, (2, 98))
        img = exposure.rescale_intensity(img, in_range=(pl, ph))
        
        # equilize hist
        img = exposure.equalize_adapthist(img)
    
    # map to log to reveal transition box
    img = np.log(medfilt2d(img.astype(float))+1)
    
    # get row and col profile gradient
    pdot_col = np.gradient(gaussian_filter1d(np.average(img, axis=0), sigma=11))
    pdot_row = np.gradient(gaussian_filter1d(np.average(img, axis=1), sigma=11))

    return {
        'left':  np.argmax(pdot_col),
        'right': np.argmin(pdot_col),
        'top':   np.argmax(pdot_row),
        'bot':   np.argmin(pdot_row),
    }


def detect_slit_corners(img: np.ndarray, r: float=50) -> list:
    """
    Description
    -----------
    Detect four corners (sub-pixel) formed by the four balde slits commonly 
    used at 1ID@APS.

    Parameters
    ----------
    img: np.ndarray
        input images, slit baldes must be visible within the image
    r: float
        domain size, will be automatically adjusted to avoid out-of-bound
        issue
    
    Returns
    -------
    list[upper_left, lower_left, lower_right, upper_right]
        List of the sub-pixel positions of the four corners in the
        counter-clock-wise order

    NOTE
    ----
    The location of the corner is affected by the size of the domain (r). A
    consistent value is recommended for quantitative analysis, such as detector
    drift correction.
    """
    # guess the rough location first
    # by default use boost contrast, if failed, use raw image
    img = _safe_read_img(img)
    try:
        edges = guess_slit_box(img, boost=True)
        le,re,te,be = edges['left'], edges['right'], edges['top'], edges['bot']
        
        r_row, r_col = min(r, be-te-1), min(r, re-le-1)
        
        safe_domain = lambda row, col: [(max(row - r_row, 0), min(row + r_row + 1, img.shape[0])), 
                                        (max(col - r_col, 0), min(col + r_col + 1, img.shape[1])),
                                    ]
        
        cnrs = [(te, le), (be, le), (be, re), (te, re)]  # (row, col)
        
        for i, cnr in enumerate(cnrs):
            
            rowrange, colrange = safe_domain(*cnr)
            domain = img[rowrange[0]:rowrange[1], colrange[0]:colrange[1]]
            
            horizontal_lp = np.average(domain, axis=0)
            vertical_lp   = np.average(domain, axis=1)
            
            popt, _ = fit_sigmoid(np.arange(len(vertical_lp)), vertical_lp)
            _row = popt[0]
            popt, _ = fit_sigmoid(np.arange(len(horizontal_lp)), horizontal_lp)
            _col = popt[0]
            
            cnrs[i] = (rowrange[0]+_row, colrange[0]+_col)
    except:
        print("boost contrast leads to error, use raw image instead")
        edges = guess_slit_box(img, boost=False)
        le,re,te,be = edges['left'], edges['right'], edges['top'], edges['bot']
        
        r_row, r_col = min(r, be-te-1), min(r, re-le-1)
        
        safe_domain = lambda row, col: [(max(row - r_row, 0), min(row + r_row + 1, img.shape[0])), 
                                        (max(col - r_col, 0), min(col + r_col + 1, img.shape[1])),
                                    ]
        
        cnrs = [(te, le), (be, le), (be, re), (te, re)]  # (row, col)
        
        for i, cnr in enumerate(cnrs):
            
            rowrange, colrange = safe_domain(*cnr)
            domain = img[rowrange[0]:rowrange[1], colrange[0]:colrange[1]]
            
            horizontal_lp = np.average(domain, axis=0)
            vertical_lp   = np.average(domain, axis=1)
            
            popt, _ = fit_sigmoid(np.arange(len(vertical_lp)), vertical_lp)
            _row = popt[0]
            popt, _ = fit_sigmoid(np.arange(len(horizontal_lp)), horizontal_lp)
            _col = popt[0]
            
            cnrs[i] = (rowrange[0]+_row, colrange[0]+_col)
    
    return cnrs


def detect_rotation_center(
    projs: np.ndarray, 
    omegas: np.ndarray,
    index_good: np.ndarray,
    do_minus_log: bool=True,
    ) -> float:
    """
    Description
    -----------
    Use the phase-contrast method provided in tomopy to detect the rotation
    center of given tomo image stack.

    Parameters
    ----------
    projs: np.ndarray
        Tomo imagestack with [axis_omega, axis_row, axis_col]
    omegas: np.ndarray
        rotary position array
    index_good: np.ndarray
        indices of uncorrupted frames

    Returns
    -------
    Rotation center horizontal position
    """
    # assume equal step, find the index range equals to 180 degree
    # in case the omeage is increasing in the negative direction
    dn = np.rint(np.pi/abs(omegas[1] - omegas[0])).astype(int)

    with cf.ProcessPoolExecutor() as e:
        if do_minus_log:
            _jobs = [
                e.submit(
                    tomopy.find_center_pc,
                    rescale_image(binded_minus_log(projs[nimg,:,:])), 
                    rescale_image(binded_minus_log(projs[nimg+dn,:,:])), 
                )
                for nimg in range(dn)
            ]
        else:
            _jobs = [
                e.submit(
                    tomopy.find_center_pc,
                    rescale_image(projs[nimg,:,:]), 
                    rescale_image(projs[nimg+dn,:,:]), 
                )
                for nimg in range(dn)
            ]
    rot_cnts = [me.result() for me in _jobs]
    rot_cnts = np.array(rot_cnts + rot_cnts)

    return np.average(rot_cnts[index_good])


def get_pin_outline(
    img_pin: np.ndarray, 
    incrop:  int=61, 
    adapthist_clip: float=0.01,
    upsampling: int=12,
    ) -> list:
    """
    Description
    -----------
        Using canny edge detection and Hough transformation to detect the
        outline of a pin, which is commonly used for alignment of rotation
        stages at MPE@APS.
    
    Parameters
    ----------
    img_pin: np.ndarray
        input image with pin in the FOV
    incrop: int
        number of pixels to cropped into FOV to avoid interference of the slit blades
    adapthist_clip: float
        decrease it to supporess artifacts from scintillators and cam
    upsampling: int
        repeat hough transform to get more line segments of the same feature
        
    Returns
    -------
    list
        line segments in image coordiantes for the pin outline
    """
    # use log to suppress scitilator artifacts 
    img_pin = np.log(img_pin)
    
    # get the slit corner
    cnrs = np.array(detect_slit_corners(img_pin))
    
    # get the cropping location
    _minrow, _maxrow = int(min(cnrs[:,0])), int(max(cnrs[:,0]))
    _mincol, _maxcol = int(min(cnrs[:,1])), int(max(cnrs[:,1]))
    
    # crop the img
    # NOTE: agressisve incropping to avoid the edge detection interference from slits
    _img = exposure.rescale_intensity(
        img_pin[_minrow+incrop : _maxrow-incrop, 
                _mincol+incrop : _maxcol-incrop]
    )
    
    # use canny + hough_line to get outline segment
    _img = medfilt2d(_img)
    _img = exposure.equalize_adapthist(_img, clip_limit=adapthist_clip)
    _edges = canny(_img, sigma=3)

    # use multiprocessing for upsampling
    _cpus = max(multiprocessing.cpu_count() - 2, 2)
    with cf.ProcessPoolExecutor(max_workers=_cpus) as e:
        # schedule
        _jobs = [e.submit(
            probabilistic_hough_line,
            _edges,
            threshold=10,  
            line_length=7,  # Increase the parameter to extract longer lines.
            line_gap=2,     # Decrease the number to allow more short segments
            ) for _ in range(upsampling)]
        # # execute
        _lines = list(itertools.chain(*[me.result() for me in _jobs]))
    
    return [[(pt[0]++_mincol+incrop, pt[1]+_minrow+incrop) for pt in line] for line in _lines] 


def get_pin_tip(
    img: np.ndarray, 
    niter: int=12,
    )->np.ndarray:
    """
    Description
    -----------
    Return a representative point as the tip of the pin

    Parameters
    ----------
    img: np.ndarray
        Input image with a pin in FOV
    niter: int
        Number of passes.  Increase this value can help counter occasional
        bad outline detection as well as bad clustering (which could happen)
    
    Returns
    -------
    tuple
        The 2D corrdinate of the pin in the raster scan frame
          (0,0) --> Y   // raster frame
          |
          v    (tip[0], tip[1])
          X
        
        To plot in matplotlib, we need to 
        plt.imshow(img)           <-- raster frame
        plt.plot(tip[1], tip[0])  <-- pyplot frame
          (0,0) --> X   //pyplot frame
          |
          v
          Y
    """
    # Get pin outline
    lines = get_pin_outline(img, upsampling=niter)

    # cluster line segments into 3 group
    # NOTE: better algorithm is needed to handle 1D data, using kmeans for now
    thetas = [np.degrees(np.arctan2(abs(p1[1]-p0[1]), abs(p1[0]-p0[0]))) for p0,p1 in lines]
    fv     = [(theta, theta) for theta in thetas]
    lns    = [(p1[1]-p0[1])**2+(p1[0]-p0[0])**2 for p0,p1 in lines]
    kmeans = KMeans(n_clusters=3, algorithm='full').fit(fv)
    
    # the tip should consist the shortest line collection
    _linelen = [sum([ln for n, ln in enumerate(lns) if kmeans.labels_[n]==i]) for i in range(3)]
    pts_tip  = list(itertools.chain(*[line for n, line in enumerate(lines) if kmeans.labels_[n]==_linelen.index(min(_linelen))]))
    pts_tip  = get_center(pts_tip)

    # cast the coordinate back to raster coordinate system
    return (pts_tip[1], pts_tip[0])


def get_center(points2d:np.ndarray) -> np.ndarray:
    """
    Description
    -----------
    Get center of 2D point cloud while minimizing effect of outliers using
    inverse distance weighting

    Parameters
    ----------
    img: np.ndarray
        List of points, might contain outliers

    Return
    ------
    np.ndarray
        Averaged center coordinates
    """
    points2d = np.array(points2d)  
    cnt = np.average(points2d, axis=0)
    return np.average(points2d, weights=(1/np.sqrt(np.sum((points2d-cnt)**2, axis=1)))**2, axis=0)


def get_pin_vertical_offset(
    img_0: np.ndarray, 
    img_180: np.ndarray,
    ) -> float:
    """
    Description
    -----------
    Calculate the vertical offset (related to wedge angle) between a 180 
    degree pair of pin during alingment, which can be used to calculate the 
    amount of additinal tilt adjustment needed to level the SMS.

    Parameters
    ----------
    img_0: np.ndarray
        img taken at omega=0
    img_180: np.ndarray
        img taken at omega=180
    
    Return
    ------
    Offset from img_180 to img_0.  For example               
        if offset > 0:  img_180 is higher than img_0
                        img_180
                img_0
        if offset < 0:  img_180 if lower than img_0
                img_0
                        img_180
    """
    img_0 = _safe_read_img(img_0)
    img_180 = _safe_read_img(img_180)
    shift, _, _ = register_translation(img_0, img_180, upsample_factor=100)
    return shift[0]


# def get_pin_rotation_center(
#     img_0: np.ndarray,
#     img_180: np.ndarray,
#     ) -> Tuple:
#     """
#     Description
#     -----------
#     Using simple curve fitting to locate the rotation center of a 180-pair 
#     image of pin during alignment

#     Parameters
#     ----------
#     img_0: np.ndarray
#         img taken at omega=0
#     img_180: np.ndarray
#         img taken at omega=180

#     Return
#     ------
#     rot_center: float
#         rotation center
#     p1_center: float
#         pixel position of the first peak center
#     p2_center: float
#         pixel position of the second peak center

#     NOTE
#     Deprecated due to decreasing accuracy with pin closing to rotation axis
#     """
#     # only work for horizontal pin for now, should be easy to adapt to vertical pin
#     try:
#         _prof = np.average(img_0, axis=0) - np.average(match_histograms(img_180, img_0), axis=0)
#     except:
#         _prof = np.average(img_0, axis=0) - np.average(img_180, axis=0)
#         print("Cannot invoke match_histogram, please check scikit-image version")

#     # fit a two peak profile
#     mod = GaussianModel(prefix='p1_') + GaussianModel(prefix='p2_')
#     out = mod.fit(_prof, x=np.arange(_prof.shape[0]), 
#                   p1_center=np.argmax(_prof),
#                   p2_center=np.argmin(_prof),
#                 )

#     return (
#             (out.best_values['p1_center']+out.best_values['p2_center'])/2,     # rotation center
#             out.best_values['p1_center'],                                      # first peak center
#             out.best_values['p2_center'],                                      # second peak center
#     )


def get_beam_origin(
    img:np.ndarray, 
    slit_cnrs:np.ndarray=None, 
    size:Tuple=(500, 500),
    ) -> Tuple:
    """
    Description
    -----------
    Return the image coordinate (row, col) of the supposed beamcenter that provides the most homogeneous 
    beam proflie
    
    Parameters
    ----------
    img: np.ndarray
        Input image with only slits
    slit_cnrs: np.ndarray
        Image coordinates of the four corners defined by the slits
    size: (int, int)
        (row, col) size of the desired FOV.  A 500x500 FOV is commonly used to located the beamcenter.
        NOTE: smaller size often helps, but it should be depending on the actual FOV intended for the experiment
        
    Returns
    -------
    Tuple
    The image coordinates (row, col) of the beam center.  In ImageJ, this coordinate is displayed as (col, row).
    """
    img = _safe_read_img(img)

    # sanity check to make sure the FOV is not too large
    if size[0] > img.shape[0]:
        raise ValueError("FOV is way too large in vertical direction")
    if size[1] > img.shape[1]:
        raise ValueError("FOV is way too large in horizontal direction")
    # get the domain size
    _srow, _scol = size
    
    # detect slit corner is not provided
    slit_cnrs = np.array(detect_slit_corners(img)) if slit_cnrs is None else np.array(slit_cnrs)
    slit_top, slit_bot = int(min(slit_cnrs[:,0])), int(max(slit_cnrs[:,0]))
    slit_lft, slit_rgt = int(min(slit_cnrs[:,1])), int(max(slit_cnrs[:,1]))
        
    # avoid impact of noisy pixels (defects in detector)
    img = medfilt2d(img.astype(float))
    
    # find the brightest spot in the image, use it as a starting point
    _beamcenter = np.unravel_index(np.argmax(img, axis=None), img.shape)  # print(x_b, y_b, img[y_b, x_b])
    
    # form bounds as contraints
    def _row_in_range(beamcenter):
        return -1*(beamcenter[0]-slit_top+_srow/2)*(beamcenter[0]-slit_bot+_srow/2)
    def _col_in_range(beamcenter):
        return -1*(beamcenter[1]-slit_lft+_scol/2)*(beamcenter[1]-slit_rgt+_scol/2)
    
    # define objective function
    def _obj(beamcenter):
        _r, _c = beamcenter.astype(int)
        _data = img[_r-_srow:_r+_srow, _c-_scol:_c+_scol]
        
        _hp = np.average(_data, axis=0)
        _hmod = GaussianModel(prefix='hp_')
        _hfit = _hmod.fit(_hp, x=np.arange(_hp.shape[0]), hp_center=len(_hp)/2)
        
        _vp = np.average(_data, axis=1)
        _vmod = GaussianModel(prefix='vp_')
        _vfit = _vmod.fit(_vp, x=np.arange(_vp.shape[0]), vp_center=len(_vp)/2)
        
        # rms
        # minimizing the assymetry of the beam proflie in both directions to
        # the best we can.
        # NOTE: 
        #   The beam is not always symmetric, and we need (kind of) symmetric
        #   beam for ff-HEDM and nf-HEDM scan.
        return np.sqrt(
              0.5*(_hfit.best_values['hp_center']- len(_hp)/2)**2 \
            + 0.5*(_vfit.best_values['vp_center']- len(_vp)/2)**2
        )
    
    _rst = sp.optimize.minimize(_obj, _beamcenter,
                               constraints=({'type': 'ineq', 'fun':  _row_in_range },
                                            {'type': 'ineq', 'fun':  _col_in_range },
                                           ),
                               method='COBYLA',
                              )
    return _rst.x


def fit_pin(
    img_pin:    np.ndarray, 
    img_white:  np.ndarray, 
    side_mount: bool=False,
    ) -> float:
    """
    Description
    -----------
    Use Gaussian peak fit to locate the center of the peak.

    Parameters
    ----------
    img_pin: np.ndarray
        Tomo image containing a pin
    img_white: np.ndarray
        White field image, which should have the same FOV of img_pin minus the
        pin
    side_mount: bool
        If the pin is mounted sideway (very rare due to statibility issue),
        toggle this option to True.

    Returns
    -------
    float
        The sub-pixel position of the center of the Gaussian peak that best
        fit the profile of the pin
    """
    _ax = 1 if side_mount else 0
    img_pin = _safe_read_img(img_pin)
    img_white = _safe_read_img(img_white)
    _pf = np.average(img_white.astype(float)-img_pin.astype(float), axis=_ax)
    _mod = GaussianModel(prefix='pin_')
    _fit = _mod.fit(_pf, x=np.arange(len(_pf)), pin_center= len(_pf)/2)
    return _fit.best_values['pin_center']


def get_rotation_center(
    img_pin_0: np.ndarray, 
    img_pin_180: np.ndarray, 
    img_white:np.ndarray, 
    side_mount=False,
    ) -> float:
    """
    Description
    -----------
    Return the rotation center (in pixel) for given 180 degree pairs

    Parameters
    ----------
    img_pin_0: np.ndarray
        pin image at 0 deg
    img_pin_180: np.ndarray
        pin image at 180 deg
    img_white: np.ndarray
        white field image
    side_mount: bool
        If the pin is mounted horizontally (very rare)

    Returns
    -------
    Rotation center in pixels 
    """
    return 0.5*(fit_pin(img_pin_0, img_white, side_mount)+fit_pin(img_pin_180, img_white, side_mount))


def _safe_read_img(img):
    """
    Read in tiff image if a path is given instead of np object.
    """
    img = imread(img) if isinstance(img, str) else np.array(img)
    return np.nan_to_num(img)


if __name__ == "__main__":
    projs = np.random.random((360,60,60))
    thetas = np.linspace(0, np.pi*2, 360)
    detect_corrupted_proj(projs, thetas)
