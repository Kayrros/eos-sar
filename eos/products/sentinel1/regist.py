import numpy as np
import eos.sar
from eos.products.sentinel1.proj_model import primary_project_and_correct, secondary_project_and_correct


def get_burst_resampling_matrices(primary_cutter, secondary_cutter,
                                  azt_no_correc, rng_no_correc,
                                  azt_correc, rng_correc,
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
    azt_no_correc : dict bsid -> array
        Each element is an array of azimuth times without corrections.
    rng_no_correc : dict bsid -> array
        Each element is an array of ranges without corrections.
    azt_correc : dict bsid -> array
        Each element is an array of azimuth times with corrections.
    rng_correc : dict bsid -> array
        Each element is an array of ranges with corrections.
    bsids : list
        List of BSIDs of interest for the registration.

    Returns
    -------
    burst_resampling_matrices : dict bsid -> matrix
        Dict where the key is the bsid and the value is a 3x3 affine inverse
        resampling matrix of the burst.

    """
    assert len(azt_no_correc) == len(rng_no_correc)
    assert len(azt_correc) == len(rng_correc)
    assert len(bsids) <= len(azt_correc) <= len(azt_no_correc)

    burst_resampling_matrices = {}
    for bsid in bsids:
        rows_primary, cols_primary = \
            primary_cutter.to_row_col_in_burst(azt_no_correc[bsid], rng_no_correc[bsid], bsid)
        rows_secondary, cols_secondary = \
            secondary_cutter.to_row_col_in_burst(azt_correc[bsid], rng_correc[bsid], bsid)

        pts_no_correc = np.column_stack([rows_primary, cols_primary])
        pts_correc = np.column_stack([rows_secondary, cols_secondary])

        A = eos.sar.regist.affine_transformation(pts_no_correc, pts_correc)
        burst_resampling_matrices[bsid] = A

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
    azt_no_correc : dict bsid -> array
       Each element is an array of azimuth times without corrections.
    rng_no_correc : dict bsid -> array
       Each element is an array of ranges without corrections.
    azt_correc : dict bsid -> array
       Each element is an array of azimuth times with corrections.
    rng_correc : dict bsid -> array
        Each element is an array of ranges with corrections.
    pts_in_burst_mask : list
        Each element is a mask defining which points from the initial x, y, alt arrays
        were projected in the different bursts.
    burst_resampling_matrices : dict
        Dict where the key is the burst id and the value is a 3x3 affine inverse
        resampling matrix of the burst.

    """
    azt_no_correc, rng_no_correc, azt_correc, rng_correc, pts_in_burst_mask = primary_project_and_correct(
        primary_swath_model, x, y, alt, crs,
        bsids, primary_burst_models)

    if all(b.corrections_deactivated() for b in primary_burst_models.values()):
        burst_resampling_matrices = {bsid: None for bsid in bsids}
    else:
        burst_resampling_matrices = get_burst_resampling_matrices(
            primary_cutter, primary_cutter, azt_no_correc, rng_no_correc,
            azt_correc, rng_correc, bsids)

    return azt_no_correc, rng_no_correc, azt_correc, rng_correc, \
        pts_in_burst_mask, burst_resampling_matrices


def secondary_registration_estimation(
        secondary_swath_model, secondary_cutter, secondary_burst_models, x, y, alt, crs,
        bsids, pts_in_burst_mask, primary_cutter, azt_no_correc, rng_no_correc):
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
    azt_no_correc : dict bsid -> array
        Each element is an array of azimuth times in the primary swath without corrections.
    rng_no_correc : dict bsid -> array
        Each element is an array of ranges in the primary swath without corrections.

    Returns
    -------
    burst_resampling_matrices : dict
        Dict where the key is the burst id and the value is a 3x3 affine inverse
        resampling matrix of the burst.

    """
    _, _, azt_correc, rng_correc = secondary_project_and_correct(
        secondary_swath_model, x, y, alt, crs,
        bsids, secondary_burst_models, pts_in_burst_mask)

    burst_resampling_matrices = get_burst_resampling_matrices(
        primary_cutter, secondary_cutter, azt_no_correc, rng_no_correc,
        azt_correc, rng_correc, bsids)

    return burst_resampling_matrices
