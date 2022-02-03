import numpy as np
import eos.sar
from eos.products.sentinel1.proj_model import primary_project_and_correct, secondary_project_and_correct

def get_burst_resampling_matrices(swath_model_no_correc, swath_model_correc, 
                                  rows_no_correc_global, cols_no_correc_global,
                                  rows_correc_global, cols_correc_global, 
                                  burst_ids, global_rows_fit=False): 
    """
    Compute the resampling matrix of two swath models burst by burst. This is\ 
    typically used between the ideal model (called no_correc) and the imperfect\ 
    model where corrections should be applied to coordinates when projecting for\
    e.g. (called correc).

    Parameters
    ----------
    swath_model_no_correc : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Swath model in ideal (primary img) coordinate system.
    swath_model_correc : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Swath model in imperfect coordinate system (primary or secondary img).
    rows_no_correc_global : list
        Each element is an array of row coords without corrections inside a burst.
    cols_no_correc_global : list
        Each element is an array of col coords without corrections inside a burst.
    rows_correc_global : list
        Each element is an array of row coords with corrections inside a burst.
    cols_correc_global : list
        Each element is an array of col coords with corrections inside a burst.
    burst_ids : list
        Burst ids in the swath (0 based) where we want the registration matrices.
    global_rows_fit : boolean, optional
        If set to True, a global fitting is used for the row affine translation.
        This is a useful feature if ESD algorithm is run afterwards. 
        The default is False.

    Returns
    -------
    burst_resampling_matrices : dict
        Dict where the key is the burst id and the value is a 3x3 affine inverse
        resampling matrix of the burst.

    """
    assert len(rows_no_correc_global) == len(cols_no_correc_global)\
        == len(rows_correc_global) == len(cols_correc_global)\
            ==len(burst_ids), "List len mismatch"
    
    if global_rows_fit: 
        pts_no_correc_global = np.column_stack([np.hstack(rows_no_correc_global), 
                                               np.hstack(cols_no_correc_global)]
                                               )
        pts_correc_global = np.column_stack([np.hstack(rows_correc_global), 
                                               np.hstack(cols_correc_global)]
                                               )
        
        A_global = eos.sar.regist.affine_transformation(pts_no_correc_global, pts_correc_global)
    
    burst_resampling_matrices = {}
    
    for j, bid in enumerate(burst_ids):
        
        col_dst, row_dst = swath_model_no_correc.burst_orig_in_swath(
            bid)
        col_src, row_src = swath_model_correc.burst_orig_in_swath(
            bid)
        
        pts_no_correc_local = np.column_stack([rows_no_correc_global[j] - row_dst, 
                                               cols_no_correc_global[j] - col_dst]
                                              )
        pts_correc_local = np.column_stack([rows_correc_global[j] - row_src, 
                                            cols_correc_global[j] - col_src])
        
        A_local = eos.sar.regist.affine_transformation(pts_no_correc_local, 
                                                       pts_correc_local)
        
        if global_rows_fit:
            # Adapt global matrix to burst origins
            A_burst_adapt = eos.sar.regist.change_resamp_mat_orig(
                row_dst, col_dst, row_src, col_src, A_global)
            # row from global fitting
            A_local[0] = A_burst_adapt[0]
        
        burst_resampling_matrices[bid] = A_local
    
    return burst_resampling_matrices

def primary_registration_estimation(primary_swath_model, primary_burst_models,
                                    x, y, alt, crs, burst_ids):
    """
    Estimate the resampling matrices for a primary image so that we can resample\
    later into a ideal frame where no projection correction needs to be applied. 

    Parameters
    ----------
    primary_swath_model : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Swath model for primary img.
    primary_burst_models : eos.products.sentinel1.proj_model.Sentinel1BurstModel
        List of burst models where the resampling matrices should be estimated.
    x : array
        x coordinate of dem points.
    y : array
        y coordinate of dem points.
    alt : array
        Altitude of dem points.
    crs : any crs type accepted by pyproj
        CRS of the dem points.
    burst_ids : Iterable
        Ids in the swath(0 based) of the bursts where we wish to have resampling matrices.

    Returns
    -------
    rows_no_correc_global : list
        Each element is an array of row coords without corrections inside a burst.
    cols_no_correc_global : list
        Each element is an array of col coords without corrections inside a burst.
    rows_correc_global : list
        Each element is an array of row coords with corrections inside a burst.
    cols_correc_global : list
        Each element is an array of col coords with corrections inside a burst.
    pts_in_burst_mask : list
        Each element is a mask defining which points from the initial x, y, alt arrays
        were projected in the different bursts.
    burst_resampling_matrices : dict
        Dict where the key is the burst id and the value is a 3x3 affine inverse
        resampling matrix of the burst.

    """
    rows_no_correc_global, cols_no_correc_global, rows_correc_global, cols_correc_global, pts_in_burst_mask = primary_project_and_correct(
        primary_swath_model, x, y, alt, crs, 
        burst_ids, primary_burst_models)
    
    if np.all([b.corrections_deactivated() for b in primary_burst_models]): 
        burst_resampling_matrices = {key: None for key in burst_ids}
    else: 
        burst_resampling_matrices = get_burst_resampling_matrices(
            primary_swath_model, primary_swath_model, rows_no_correc_global, cols_no_correc_global, 
            rows_correc_global, cols_correc_global, burst_ids)
    
    return rows_no_correc_global, cols_no_correc_global,\
     rows_correc_global, cols_correc_global, pts_in_burst_mask,\
         burst_resampling_matrices

def secondary_registration_estimation(
        secondary_swath_model, secondary_burst_models,  x, y, alt, crs,
        burst_ids, pts_in_burst_mask, primary_swath_model,  rows_no_correc_global, 
        cols_no_correc_global, global_rows_fit=True ):
    """
    Estimate the resampling matrices for a secondary img w.r.t. the ideal primary frame. 

    Parameters
    ----------
    secondary_swath_model : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Swath model for the secondary img.
    secondary_burst_models : eos.products.sentinel1.proj_model.Sentinel1BurstModel
        List of burst models where the resampling matrices should be estimated.
    x : array
        x coordinate of dem points.
    y : array
        y coordinate of dem points.
    alt : array
        Altitude of dem points.
    crs : any crs type accepted by pyproj
        CRS of the dem points.
    burst_ids : Iterable
        Ids in the swath(0 based) of the bursts where we wish to have resampling matrices.
    pts_in_burst_mask : list
        Each element is a mask defining which points from the initial x, y, alt arrays
        should be projected in the different bursts.
    primary_swath_model : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Swath model for primary img.
    rows_no_correc_global : list
        Each element is an array of row coords in the primary swath without corrections inside a burst.
    cols_no_correc_global : list
        Each element is an array of col coords in the primary swath without corrections inside a burst.
    global_rows_fit : boolean, optional
        If set to True, a global fitting is used for the row affine translation.
        This is a useful feature if ESD algorithm is run afterwards. 
        The default is True.

    Returns
    -------
    burst_resampling_matrices : dict
        Dict where the key is the burst id and the value is a 3x3 affine inverse
        resampling matrix of the burst.

    """
    _, _, rows_correc_global_sec, cols_correc_global_sec = secondary_project_and_correct(
        secondary_swath_model, x, y, alt, crs,
        burst_ids, secondary_burst_models , pts_in_burst_mask)
    
    burst_resampling_matrices = get_burst_resampling_matrices(
        primary_swath_model, secondary_swath_model, rows_no_correc_global, cols_no_correc_global, 
        rows_correc_global_sec, cols_correc_global_sec, burst_ids, global_rows_fit=global_rows_fit)
    
    return burst_resampling_matrices

