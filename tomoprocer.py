#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""tomoprocer
    morph: legacy data support
        convert tiff|binary to HDF5 archive for subsequent analysis
    prep:  preprocessing tomography data
                            ----------------
                            |avaiable modes|
                            ----------------
        | ==========| === | === | === | === | ==== | ==== | === |
        |           | inr | bgn | cdd | dcp | bifc | crop | dtc |
        | --------- | --- | --- | --- | --- | ---- | ---- | --- |
        | express   |  y  |  y  |  n  |  n  |  n   |  n   |  n  |
        | lite      |  y  |  y  |  y  |  y  |  y   |  y   |  n  |
        | royal     |  y  |  y  |  y  |  y  |  y   |  y   |  y  |
        | ==========| === | === | === | === | ==== | ==== | === |
        
        * [inr]  impulse_noise_removal
        * [bgn]  background_normalization               # sinogram method
        * [cdd]  correct_detector_drifting              # through slit detection
        * [dcp]  detect_corrupted_proj                  # through 180 deg pair matching
        * [bifc] beam_intensity_fluctuation_correction  # through sinogram
        # [crop] data reduction (corpping)
        * [dtc]  correct_detector_tilt                  # rotation axis tilt correction

        NOTE: some processing steps are only available in interactive session

    recon: perform tomography reconstruction using external engine specified
           in configuration file
        * tomopy
        * tomoMPI
        * MIDAS (upcoming)

    analyze: perform specified analysis on reconstruction volume
        * porosity characterization
        * phase boundary detection
        * crack network visualization (vtk)

Usage:
    tomoprocer.py morph    <CONFIGFILE> [-v|--verbose]
    tomoprocer.py prep     <CONFIGFILE> [-v|--verbose]
    tomoprocer.py recon    <CONFIGFILE> [-v|--verbose]
    tomoprocer.py analyze  <CONFIGFILE> [-v|--verbose]
    tomoprocer.py -h | --help
    tomoprocer.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  -v --verbose  Verbose output
"""

from docopt                   import docopt
from graphviz                 import Digraph
from tomoproc.util.file       import load_h5
from tomoproc.util.file       import load_yaml


def build_graph(graph_root, nodes, edges, fn='processing_graph.gv'):
    """Build processing graph with given nodes and edges"""
    graph = Digraph()
    _nodes = [graph_root] + nodes

    for i, node in enumerate(_nodes):
        graph.node(str(i), node)
    
    for i, edge in enumerate(edges):
        graph.edge(str(i), str(i+1), label=edge)

    graph.render(fn)


def get_h5_file_name(cfg):
    """Return the HDF5 file name based on the configuration file"""
    from os.path import join
    return join(cfg['output']['filepath'],  f"{cfg['output']['fileprefix']}.{cfg['output']['type']}")


def tomo_prep(cfg, verbose_output=False, write_to_disk=True):
    """Pre-processing tomography data with given tomography configurations"""
    # NOTE:
    # The current implementation of processing graph is not very clean, might
    # need to redo it later...
    import tomopy
    import multiprocessing
    import h5py
    import numpy              as np
    import concurrent.futures as cf
    from tomoproc.prep.detection  import detect_slit_corners
    from tomoproc.prep.detection  import detect_corrupted_proj
    from tomoproc.prep.detection  import detect_rotation_center
    from tomoproc.prep.correction import correct_detector_drifting
    from tomoproc.prep.correction import correct_detector_tilt
    from tomoproc.prep.correction import denoise
    from tomoproc.prep.correction import beam_intensity_fluctuation_correction as bifc
    from tqdm                     import tqdm
    # -- 
    _cpus = max(multiprocessing.cpu_count() - 3, 2)

    # --
    mode = cfg['reconstruction']['mode']
    _nodes = []
    _edges = []

    # --
    if verbose_output: print("loading H5 to memory")
    h5fn = get_h5_file_name(cfg)
    with h5py.File(h5fn, 'r') as _h5f:
        wfbg = _h5f['exchange']['data_white_pre'][()]
        proj = _h5f['exchange']['data'][()]
        wbbg = _h5f['exchange']['data_white_post'][()]
        dbbg = _h5f['exchange']['data_dark'][()]
    # h5f = load_h5(h5fn)
    # wfbg = h5f['exchange']['data_white_pre']
    # proj = h5f['exchange']['data']
    # wbbg = h5f['exchange']['data_white_post']
    # dbbg = h5f['exchange']['data_dark']

    # --
    if verbose_output: print("extracting omegas")

    # try:
    #     with h5py.File(h5fn, 'r') as _h5f:
    #         omegas = _h5f['/omegas'][()]  # use omega list if possible
    #         delta_omega = omegas[1] - omegas[0]
    # except:
    delta_omega = (cfg['omega_end']-cfg['omega_start'])/(proj.shape[0]-1)
    omegas = np.arange(cfg['omega_start'], cfg['omega_end']+delta_omega, delta_omega)
    if verbose_output:
        print(f"Omega range:{omegas[0]} ~ {omegas[-1]} with step size of {delta_omega}")
    omegas = np.radians(omegas)

    # -- noise reduction
    e = cf.ProcessPoolExecutor(max_workers=_cpus)
    _jobs = [e.submit(denoise, proj[n,:,:].astype(float)) for n in range(proj.shape[0])]
    # execute
    _proj = [me.result() for me in _jobs]
    # map back
    for n in range(proj.shape[0]):
        proj[n,:,:] = _proj[n]
    _nodes.append('proj')
    _edges.append('noise reduction')

    # -- correct detector drifting and crop data
    if mode in ['lite', 'royal']:
        if verbose_output: print("correct detector drifting")
        proj, m_corr_drift = correct_detector_drifting(proj)
        _nodes.append('proj, m_corr_drift')
        _edges.append('correct_detector_drifting')

        if verbose_output: print("crop out slits")
        cnrs = np.array(detect_slit_corners(proj[1,:,:]))
        _minrow, _maxrow = int(min(cnrs[:,0])), int(max(cnrs[:,0]))
        _mincol, _maxcol = int(min(cnrs[:,1])), int(max(cnrs[:,1]))
        _shape_before = proj.shape
        wfbg = wfbg[:, _minrow:_maxrow, _mincol:_maxcol]
        proj = proj[:, _minrow:_maxrow, _mincol:_maxcol]
        wbbg = wbbg[:, _minrow:_maxrow, _mincol:_maxcol]
        dbbg = dbbg[:, _minrow:_maxrow, _mincol:_maxcol]
        _shape_after = proj.shape
        _nodes.append(f'proj:{_shape_before}->{_shape_after}')
        _edges.append('detect_slit_corners')

        if verbose_output: print("remove corrupted frames")
        idx_bad, idx_good = detect_corrupted_proj(proj, omegas)
        _shape_before = proj.shape
        proj = proj[idx_good,:,:]
        omegas = omegas[idx_good]
        _shape_after = proj.shape
        _nodes.append(f'proj:{_shape_before}->{_shape_after}')
        _edges.append('detect_corrupted_proj')
        if verbose_output: print(f"corrupted frames ind:{idx_bad}")
    
    # --
    if verbose_output: print("remove background")
    wflat = 0.5*(np.median(wfbg, axis=0) + np.median(wbbg, axis=0))
    dflat = np.median(dbbg, axis=0)
    proj = (proj-dflat)/(wflat-dflat)
    _nodes.append('proj=(proj-dark)/(white-dark)')
    _edges.append('remove background')

    # --
    if mode in ['royal']:
        if verbose_output: print("correct detector tilt")
        proj = correct_detector_tilt(proj, omegas)
        _nodes.append('proj')
        _edges.append('correct_detector_tilt')
    
    # --
    # NOTE:
    # TODO:
    # For some unknown reason, the multiprocessing approach does not work here.
    # Will investigate later.
    if mode in ['lite', 'royal']:
        if verbose_output: print("normalize sinograms with multiprocessing")
        # for n in tqdm(range(proj.shape[1])):
        #     proj[:,n,:] = denoise(bifc(proj[:,n,:]))
        def _bgadjust(img):
            return denoise(bifc(img))
        # use multi-processing to speed up
        e = cf.ProcessPoolExecutor(max_workers=_cpus)
        _jobs = [ e.submit(_bgadjust, proj[:,n,:]) for n in range(proj.shape[1])]
        # execute
        _proj = [me.result() for me in _jobs]
        # map back
        for n in tqdm(range(proj.shape[1])):
            proj[:,n,:] = _proj[n]
        _nodes.append('proj')
        _edges.append('bg normalize')

    # -log
    if verbose_output: print("-log")
    proj = tomopy.minus_log(proj, ncore=max(1, multiprocessing.cpu_count()-1))
    proj[np.isnan(proj)] = 0
    proj[np.isinf(proj)] = 0
    proj[proj<0] = 0
    _nodes.append('proj')
    _edges.append('-log')

    # either 
    #   - write data back to HDF5 archive
    #   - return the intermedia results
    if write_to_disk:
        if verbose_output: print(f"writing data back to {h5fn}")
        with h5py.File(h5fn, 'a') as _h5f:
            _dst_omegas = _h5f.create_dataset('/tomoproc/omegas', data=omegas)
            _dst_corrm  = _h5f.create_dataset('/tomoproc/m_corr_drift', data=m_corr_drift)
            _dst_proj   = _h5f.create_dataset('/tomoproc/proj', data=proj, chunks=True, compression="gzip", compression_opts=9, shuffle=True)
        if verbose_output: print(f"Building processing graph")
        build_graph(h5fn, _nodes, _edges, fn=h5fn.split('.')[:-1]+"_prep.gv")
    else:
        return proj, omegas, _nodes, _edges



def tomo_recon(cfg, verbose_output=False):
    """Perform reconstruction using specified eigine"""
    import tomopy
    import h5py
    from tomoproc.prep.detection  import detect_rotation_center
    # -- read sinograms into memory
    h5fn = get_h5_file_name(cfg)
    h5f = h5py.File(h5fn, 'a')
    try:
        if verbose_output: print("Try to located pre-processed sinogram...")
        omegas = h5f['/tomoproc/omegas'][:]
        proj   = h5f['/tomoproc/proj'][:]
        _nodes = []
        _edges = []
    except:
        if verbose_output: 
            print("cannot find pre-processed sinogram.")
            print("start pre-processing now")
        proj, omegas, _nodes, _edges= tomo_prep(cfg, verbose_output=verbose_output, write_to_disk=False)
    # --
    if verbose_output: print("Locate rotation center...")
    rot_cnt = detect_rotation_center(proj, omegas)
    if verbose_output:
        print(f"proj.shape = {proj.shape}")
        print(f"omegas.shape = {omegas.shape}")
        print(f"rotation center = {rot_cnt}")
    _nodes.append('proj,rot={rot_cnt}')
    _edges.append('detect_rotation_center')
    
    # --
    recon = tomopy.recon(proj, omegas, center=rot_cnt, algorithm='gridrec', filter_name='hann')
    if verbose_output: print(f"reconstruction shape = {recon.shape}")
    _nodes.append('recon')
    _edges.append('tomopy_gridrec_hann')

    # --
    if verbose_output: print("write to HDF5 archive")
    _dst_recon = h5f.create_dataset("/tomoproc/recon_auto", data=recon, chunks=True, compression="gzip", compression_opts=9, shuffle=True)
    _dst_recon.attrs['engien'] = "tomopy"
    _dst_recon.attrs['algorithm'] = "gridrec"
    _dst_recon.attrs['filter_name'] = "hann"
    _dst_recon.attrs['rotation_center'] = rot_cnt
    h5f.close()

    # -- 
    build_graph(h5fn, _nodes, _edges, fn=h5fn.split('.')[:-1]+"_recon.gv")
    

if __name__ == "__main__":
    argvs = docopt(__doc__, argv=None, help=True, version="tomoprocer v0.0.2")
    verbose_output = argvs['--verbose']
    if verbose_output:
        print(argvs)

    if argvs['morph']:
        # lazy import
        from tomoproc.morph.tiff2h5 import pack_tiff_to_hdf5
        try:
            pack_tiff_to_hdf5(argvs['<CONFIGFILE>'])
        except:
            raise FileExistsError('remove previous generated H5 archive')
    elif argvs['prep']:
        # NOTE:
        # Standard experiment at 6-ID-D should have a config file (yml).
        # However, a temp config file can be generate from the H5 archive if 
        # necessary.
        file_ext = argvs['<CONFIGFILE>'].split(".")[-1].lower()
        if file_ext in ['hdf5', 'h5', 'hdf']:
            print("Generate config file for given HDF5 archive...")
            # extract the config information from HDF5 archive
            fn_config = "".join(argvs['<CONFIGFILE>'].split('.')[:-1]+[".yml"])
            config_dict = {
                'tomo': {
                    'omega_start': -180,
                    'omega_end': 180,
                    'output':   {
                        'filepath':    './',    
                        'fileprefix':  "".join(argvs['<CONFIGFILE>'].split('.')[:-1]),
                        'type':        argvs['<CONFIGFILE>'].split(".")[-1].lower(),
                    },
                    'reconstruction': {
                        'mode': 'lite',
                    }
                }
            }
            # write config file to disk
            from tomoproc.util.file import write_yaml
            write_yaml(fn_config, config_dict)
            print(f"Please double check the generated config file: {fn_config}")
            print("Then start the prep with:")
            print(f">> tomoprocer prep {fn_config}")
        elif file_ext in ['yml', 'yaml']:
            cfg_all =load_yaml(argvs['<CONFIGFILE>'])
            tomo_prep(cfg_all['tomo'], verbose_output=verbose_output)
        else:
            raise ValueError("Please use config(yml) or h5 archive.")
    elif argvs['recon']:
        # perform reconstruction
        cfg_all =load_yaml(argvs['<CONFIGFILE>'])
        tomo_recon(cfg_all['tomo'], verbose_output=verbose_output)
    elif argvs['analyze']:
        pass
    else:
        raise ValueError('Please use --help to check available optoins')
