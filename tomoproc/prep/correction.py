#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Contain functions operating on tomography sinograsm

NOTE:
    Different experiment requires different type of correction, the choice of
    which should be established via trial and error.
"""
import tomopy
import numpy                   as     np
import concurrent.futures      as     cf

from   typing                  import Optional
from   typing                  import Tuple
from   scipy.signal            import medfilt2d
from   scipy.ndimage           import gaussian_filter
from   scipy.ndimage           import shift
from   scipy.ndimage           import affine_transform
from   tomoproc.prep.detection import detect_sample_in_sinogram
from   tomoproc.prep.detection import detect_corrupted_proj
from   tomoproc.prep.detection import detect_slit_corners
from   tomoproc.prep.detection import detect_rotation_center
from   tomoproc.util.npmath    import calc_affine_transform
from   tomoproc.util.npmath    import rescale_image
from   tomoproc.util.npmath    import binded_minus_log

def denoise(
        sino: np.ndarray,
        method: str='smd',
        config: dict={
            'kernel_size': 3,
            'threshold': 0.1,
            },
    ) -> np.ndarray:
    """
    Description
    -----------
        Use selected method 
            [median, selective median, SVD] 
        to reduce the impulse noise in the sinogram.  All methods used here are
        lossy, especially the SVD method.
    
    Parameters
    ----------
    sino: np.ndarray
        single sinograms for denoising
    method: str, [median, smd, SVD]
        method for denoising includes
        **median**
            simple 2D median filtering using scipy.signal.medfilt2d
            config = {'kernel_size': 3}
        **selective median filtering**
            only replace the noisy pixels with the corresponding median value
            config = {'kernel_size': 3, 'threshold': 0.05}
        **singular value decomposition**
            retain upto given threshold level of details in the corresponding
            eigen space
            config = {'threshold': 0.95}
    config: dict
        parameters for different methods
    
    Returns
    -------
    np.ndarray
        denoised sinogram
    """
    if method.lower() in ['median', 'simple']:
        return medfilt2d(sino, kernel_size=config['kernel_size'])
    elif method.lower() in ['selective median', 'smd']:
        sino_md = medfilt2d(sino, kernel_size=config['kernel_size'])
        diff = np.absolute(sino_md - sino)/(sino+1)
        return np.where(diff>config['threshold'], sino_md, sino)
    elif method.lower() in ['svd']:
        U, S, V = np.linalg.svd(sino, full_matrices=True)
        eigen_cut = int(min(U.shape[1], V.shape[0])*config['threshold'])
        return (U[:,:eigen_cut]*S[:eigen_cut])@V[:eigen_cut,:]
    else:
        raise NotImplementedError


def beam_intensity_fluctuation_correction(
        sino: np.ndarray,
        detect_bg: bool=True,
        left_bg:  int=5,        # in pixels
        right_bg: int=5,        # in pixels
        interpolate: bool=True,
    ) -> np.ndarray:
    """
    Description
    -----------
    The beam intensity always varies during an experiment, leading to varying
    background (non-sample) region.  This artifacts will leads to strong linear
    artifacts in the final reconstruction, therefore need to be corrected by
    forcing all non-sample region (background) to be one (zero attenuation).
    By default, the sample bound (left and right edge) will be automatically 
    detected with 
        tomoproc.prep.detection.detect_sample_in_sinogram
    which can be bypassed (for speed) if the sample limit is known.

    Parameters
    ----------
    sino: np.ndarray
        sinogram as attenuation map (befor the minus-log step)
    detect_bg: bool
        whether to use automated background pixel detection
    left_bg, right_bg: int
        designated background pixels, superceeded by detect_bg
    interpolate: bool
        whether to interpolate background or not
        ========
        | NOTE |
        ========
        linear interpolation is recommended as the beam are not alawys stable,
        which could lead to intensity shift that cannot be correct through
        background removal.

    Returns
    -------
    np.ndarray
        sinogram with non-sample region (background) normalized to one
            one -> zero attenuation

    NOTE
    ----
    * This method will not work if the sample occupy the entire filed of view.
    * Sometimes two or four iterations are necessary for unstable beam.
    """
    sino = np.sqrt(sino)  # for better auto background detection
    
    # get sample location
    if detect_bg:
        ledge, redge = detect_sample_in_sinogram(sino)
    else:
        ledge, redge = left_bg, sino.shape[1]-right_bg

    # locate the left and right background
    # NOTE:
    #   Due to hardware issue, the first and last pixels are not always
    #   reliable, therefore throw them out...
    lbg = np.average(sino[:, 1:ledge],   axis=1)
    rbg = np.average(sino[:, redge:-1],  axis=1)

    # calculate the correction matrix alpha
    alpha = np.ones(sino.shape)
    if interpolate:
        for n in range(sino.shape[0]):
            alpha[n,:] = np.linspace(lbg[n], rbg[n], sino.shape[1])
    else:
        alpha *= ((lbg+rbg)/2)[:,None]
    
    # apply the correction
    return (sino/alpha)**2


def remove_corrupted_projs(
    projs: np.ndarray,
    omegas: np.ndarray,
    idx_good: Optional[np.ndarray]=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Remove corrupted prjections/frames in a given tomo image stack

    Parameters
    ----------
    projs: np.ndarray
        Tomo image stack [axis_omega, axis_imgrow, axis_imgcol]
    idx_good: np.ndarray|None
        index (along omega axis) for good frames 

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Pruned tomo image stack and corresponding angular position (omegas)
    """
    # get the index of good frames if not provided
    idx_good = detect_corrupted_proj(projs, omegas)[1] if idx_good is None else idx_good

    return projs[idx_good,:,:], omegas[idx_good]


def correct_horizontal_jittering(
    projs: np.ndarray,
    omegas: np.ndarray,
    remove_bad_frames: bool=True,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Correct the horizontal jittering of sample by shifting all projections
    such that the rotation center is alawys at the center of each projection.

    Parameters
    ----------
    projs: np.ndarray
        tomogaphy image stacks [axis_omega, axis_imgrow, axis_imgcol]
    omegas: np.ndarray
        rotary position array
    remove_bad_frames: bool
        remove corrupted frames from the projs

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        corrected (and pruned) projs ready for tomography reconstruction and
        corresponding omegas
    """
    # assume equal step, find the index range equals to 180 degree
    dn = int(np.pi/(omegas[1] - omegas[0]))

    # identify bad frames if necesary
    if remove_bad_frames:
        _, idx_good = detect_corrupted_proj(projs, omegas)

    # get the cnts from each 180 pairs
    with cf.ProcessPoolExecutor() as e:
        _jobs = [
            e.submit(
                tomopy.find_center_pc,
                rescale_image(binded_minus_log(projs[nimg,:,:])), 
                rescale_image(binded_minus_log(projs[nimg+dn,:,:])), 
            )
            for nimg in range(dn)
        ]
    
    cnts = [me.results() for me in _jobs]

    # 180 -> 360
    cnts = cnts + cnts
    shift_vals = [
        np.array([0, projs.shape[2]/2 - val])
        for val in cnts
    ]

    # shift each proj so that the rotation center is the central col
    for n in range(len(shift_vals)):
        projs[n,:,:] = shift(
            projs[n,:,:],
            shift_vals[n],
            mode='constant', cval=0, order=1,
            )

    # remove the corrupted frames if requested
    if remove_bad_frames:
        projs = projs[idx_good,:,:]
        omegas = omegas[idx_good]
    
    return projs, omegas


def correct_detector_drifting(
    projs: np.ndarray,
    ) -> Tuple[np.ndarray, list]:
    """
    Description
    -----------
    Systematic shifting and rotation could happen to a tomo imagestack
    due to detetor stage settling.

    Parameters
    ----------
    projs: np.ndarray
        Tomo imagestack with [axis_omega, axis_row, axis_col]

    Returns
    -------
    (np.ndarray, list)
        Tomo imagestack with detector drifting corrected and the list of the
        affine transformation matrix used for correction.

    NOTE
    ----
    Use ALL available resource possible by default (aggresive approach)
    """
    # -- detect the four corners
    with cf.ProcessPoolExecutor() as e:
        _jobs = [
            e.submit(
                detect_slit_corners, 
                projs[n_omega,:,:],
                )
            for n_omega in range(projs.shape[0])
        ]
    
    cnrs_all = [me.result() for me in _jobs]

    # -- calculate the transformation matrix using the first frames as the
    # -- reference frame
    with cf.ProcessPoolExecutor() as e:
        _jobs = [
            e.submit(
                calc_affine_transform, 
                np.array(me),            # source
                np.array(cnrs_all[0]),   # reference/target
                )
            for me in cnrs_all
        ]
    
    correction_matrix = [me.result() for me in _jobs]

    # -- apply the affine transformation to each frame
    with cf.ProcessPoolExecutor() as e:
        _jobs = [
            e.submit(
                affine_transform,
                projs[n_omega,:,:],
                correction_matrix[n_omega][0:2,0:2],      # rotation
                offset=correction_matrix[n_omega][0:2,2]  # translation
            )
            for n_omega in range(projs.shape[0])
        ]
    projs = np.stack([me.result() for me in _jobs], axis=0)

    return projs, correction_matrix


def correct_detector_tilt(
    projs: np.ndarray, 
    omegas: np.ndarray,
    tor: int=1, 
    nchunk: int=4,
    ) -> np.ndarray:
    """
    Description
    -----------
    Due to detector mounting process, the vertical axis of the detector (hence
    the image) might not be parallel to the actual rotation axis.  Therefore,
    the projections need to be slighly rotated until the rotation axis is
    parallel to the vertial axis of the image.

    Parameters
    ----------
    projs: np.ndarray
        Tomo imagestack with [axis_omega, axis_row, axis_col]
    omegas: np.ndarray
        rotary position array
    tor: int
        tolerance for horizontal shift in pixels
    nchunk: int
        number of subdivisions used to identify the rotation axis tilt
    
    Returns
    -------
    np.ndarray
        Correct projection images.
    """
    # calculate slab thickness (allow overlap)
    _st = int(np.ceil(projs.shape[1]/nchunk))

    _err = 10  #
    _cnt = 0   #
    while(_err > tor):
        cnt_cols = [
            detect_rotation_center(projs[:,n*_st:min((n+1)*_st, projs.shape[2]),:], omegas)
            for n in range(nchunk)
        ]
        
        cnt_col = np.average(cnt_cols)

        # update the error
        _err = np.max([abs(me-cnt_col) for me in cnt_cols])
        _cnt = _cnt + 1

        # safe guard and lazy update
        if _cnt > 10000: break
        if _err < tor: break

        # calcualte the correction matrix
        pts_src = np.array([[(n+0.5)*_st, cnt_cols[n]] for n in range(nchunk)])
        pts_tgt = np.array([[(n+0.5)*_st, cnt_col    ] for n in range(nchunk)])
        _afm = calc_affine_transform(pts_src, pts_tgt)

        # -- apply the affine transformation to each frame
        with cf.ProcessPoolExecutor() as e:
            _jobs = [
                e.submit(
                    affine_transform,
                    projs[n_omega,:,:],
                    _afm[0:2,0:2],      # rotation
                    offset=_afm[0:2,2]  # translation
                )
                for n_omega in range(projs.shape[0])
            ]
        projs = np.stack([me.result() for me in _jobs], axis=0)
    
    return projs


if __name__ == "__main__":
    testimg = np.random.random((500,500))
    sino = denoise(testimg, method='svd')
