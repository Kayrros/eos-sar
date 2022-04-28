import numpy as np
import eos.sar
from eos.products.sentinel1.proj_model import primary_project_and_correct, secondary_project_and_correct


def get_burst_resampling_matrices(primary_cutter, secondary_cutter,
                                  rows_no_correc_global, cols_no_correc_global,
                                  rows_correc_global, cols_correc_global,
                                  bsids):
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
    rows_no_correc_global : dict bsid -> list of coords
        Each element is an array of row coords without corrections inside a burst.
    cols_no_correc_global : dict bsid -> list of coords
        Each element is an array of col coords without corrections inside a burst.
    rows_correc_global : dict bsid -> list of coords
        Each element is an array of row coords with corrections inside a burst.
    cols_correc_global : dict bsid -> list of coords
        Each element is an array of col coords with corrections inside a burst.
    bsids : list
        List of BSIDs of interest for the registration.

    Returns
    -------
    burst_resampling_matrices : dict bsid -> matrix
        Dict where the key is the bsid and the value is a 3x3 affine inverse
        resampling matrix of the burst.

    """
    assert len(rows_no_correc_global) == len(cols_no_correc_global)
    assert len(rows_correc_global) == len(cols_correc_global)
    assert len(bsids) <= len(rows_correc_global) <= len(rows_no_correc_global)

    burst_resampling_matrices = {}
    for bsid in bsids:
        azt, rng = primary_cutter.to_azt_rng(rows_no_correc_global[bsid], cols_no_correc_global[bsid])
        rows_primary_local, cols_primary_local = primary_cutter.to_row_col_in_burst(azt, rng, bsid)

        azt, rng = secondary_cutter.to_azt_rng(rows_correc_global[bsid], cols_correc_global[bsid])
        rows_secondary_local, cols_secondary_local = secondary_cutter.to_row_col_in_burst(azt, rng, bsid)

        pts_no_correc_local = np.column_stack([rows_primary_local, cols_primary_local])
        pts_correc_local = np.column_stack([rows_secondary_local, cols_secondary_local])

        A_local = eos.sar.regist.affine_transformation(pts_no_correc_local, pts_correc_local)
        burst_resampling_matrices[bsid] = A_local

    return burst_resampling_matrices


def primary_registration_estimation(primary_swath_model, primary_cutter, primary_burst_models,
                                    x, y, alt, crs, bsids):
    """
    Estimate the resampling matrices for a primary image so that we can resample\
    later into a ideal frame where no projection correction needs to be applied.

    Parameters
    ----------
    primary_swath_model : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Swath model for primary img.
    primary_cutter : eos.products.sentinel1.acquisition.Sentinel1AcquisitionCutter
        Primary acquisition cutter.
    primary_burst_models : dict bsid -> eos.products.sentinel1.proj_model.Sentinel1BurstModel
        Burst model per bsid, where the resampling matrices should be estimated.
    x : array
        x coordinate of dem points.
    y : array
        y coordinate of dem points.
    alt : array
        Altitude of dem points.
    crs : any crs type accepted by pyproj
        CRS of the dem points.
    bsids : list
        List of BSIDs of interest for the registration.

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
        bsids, primary_burst_models)

    if all(b.corrections_deactivated() for b in primary_burst_models.values()):
        burst_resampling_matrices = {bsid: None for bsid in bsids}
    else:
        burst_resampling_matrices = get_burst_resampling_matrices(
            primary_cutter, primary_cutter, rows_no_correc_global, cols_no_correc_global,
            rows_correc_global, cols_correc_global, bsids)

    return rows_no_correc_global, cols_no_correc_global,\
        rows_correc_global, cols_correc_global, pts_in_burst_mask,\
        burst_resampling_matrices


def secondary_registration_estimation(
        secondary_swath_model, secondary_cutter, secondary_burst_models, x, y, alt, crs,
        bsids, pts_in_burst_mask, primary_cutter, rows_no_correc_global, cols_no_correc_global):
    """
    Estimate the resampling matrices for a secondary img w.r.t. the ideal primary frame.

    Parameters
    ----------
    secondary_swath_model : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Swath model for the secondary img.
    secondary_cutter : eos.products.sentinel1.acquisition.Sentinel1AcquisitionCutter
        Secondary acquisition cutter.
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
    bsids : list
        List of BSIDs of interest for the registration.
    pts_in_burst_mask : list
        Each element is a mask defining which points from the initial x, y, alt arrays
        should be projected in the different bursts.
    primary_cutter : eos.products.sentinel1.acquisition.Sentinel1AcquisitionCutter
        Primary acquisition cutter.
    rows_no_correc_global : list
        Each element is an array of row coords in the primary swath without corrections inside a burst.
    cols_no_correc_global : list
        Each element is an array of col coords in the primary swath without corrections inside a burst.

    Returns
    -------
    burst_resampling_matrices : dict
        Dict where the key is the burst id and the value is a 3x3 affine inverse
        resampling matrix of the burst.

    """
    _, _, rows_correc_global_sec, cols_correc_global_sec = secondary_project_and_correct(
        secondary_swath_model, x, y, alt, crs,
        bsids, secondary_burst_models, pts_in_burst_mask)

    # from swath coordinate system to the cutter coordinate system
    rows_correc_mosaic_sec = {}
    cols_correc_mosaic_sec = {}
    for bsid in bsids:
        azt, rng = secondary_swath_model.to_azt_rng(rows_correc_global_sec[bsid], cols_correc_global_sec[bsid])
        row, col = secondary_cutter.to_row_col(azt, rng)
        rows_correc_mosaic_sec[bsid], cols_correc_mosaic_sec[bsid] = row, col

    burst_resampling_matrices = get_burst_resampling_matrices(
        primary_cutter, secondary_cutter, rows_no_correc_global, cols_no_correc_global,
        rows_correc_mosaic_sec, cols_correc_mosaic_sec, bsids)

    return burst_resampling_matrices
