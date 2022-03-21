import numpy as np
from eos.sar import utils
from eos.products.sentinel1 import burst_resamp


def warp_rois_read_resample_deburst(read_rois_no_correc, burst_ids, primary_swath_model,
                                    secondary_swath_model, burst_resampling_matrices,
                                    secondary_bursts_metas, image_readers,
                                    write_rois, out_shape, out=None,
                                    get_complex=True, margin=5):
    """
    Warp the rois, read then resample, and deburst.

    Parameters
    ----------
    read_rois_no_correc : List of eos.sar.roi.Roi
        Each element is an roi in the ideal primary frame.
    burst_ids : Iterable
        Burst ids in the swath (0 based) associated with each roi.
    primary_swath_model : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Primary swath model.
    secondary_swath_model : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Secondary swath model.
    burst_resampling_matrices : dict
        Dict where the key is the burst id and the value is a 3x3 affine inverse
        resampling matrix of the burst.
    secondary_bursts_metas : List of dicts
        List of metadata of all bursts in a swath (even the ones we are not considering).
    image_readers : List of rasterio.DatasetReader
        Opened rasterio datasets.
    write_rois : List of eos.sar.roi.Roi
        Each element defines the roi to write the data in the output array.
    out_shape : tuple
        Output array shape.
    out : ndarray, optional
        Output array (shape should be `out_shape` and dtype should be compatible with `get_complex`).
    get_complex : boolean, optional
        If set to True, get the complex array. Otherwise, all the processing is conducted
        on the amplitude from the start. The default is True.
    margin : int, optional
        Pixel safety margin to be applied after warping read_rois_no_correc. The default is 5.

    Returns
    -------
    debursted_crop : ndarray
        The debursted crop.
    read_rois_correc : List of eos.sar.roi.Roi
        Each element is an roi in the imperfect (primary or secondary) frame.
        It is obtained by warping the input roi and adding a padding within the
        valid image boundaries.
    resamplers : List of eos.products.sentinel1.Sentinel1BurstResample
        Each resampler can be applied directly on the read array with read_rois_correc.

    """
    read_rois_correc = []
    resamplers = []

    def gen():
        for read_roi_dst, bid in zip(read_rois_no_correc, burst_ids):
            burst_roi_dst = primary_swath_model.bursts_rois[bid]
            burst_roi_src = secondary_swath_model.bursts_rois[bid]

            burst_array_resamp, read_roi_src, resampler = burst_resamp.warp_roi_read_resample(
                burst_resampling_matrices[bid], burst_roi_dst, read_roi_dst, burst_roi_src,
                secondary_bursts_metas[bid], image_readers[bid], get_complex, margin)

            read_rois_correc.append(read_roi_src)
            resamplers.append(resampler)
            yield burst_array_resamp, write_rois[bid]

    debursted_crop = utils.stitch_arrays(gen(), out_shape, out=out)
    return debursted_crop, read_rois_correc, resamplers


def deburst_primary(roi_in_swath_no_correc, primary_swath_model,
                    burst_resampling_matrices, bursts_metas, image_readers,
                    get_complex=True):
    """
    Deburst an roi in the primary image.

    Parameters
    ----------
    roi_in_swath_no_correc : eos.sar.roi.Roi
        Region of interest to be deburst defined in the swath.
    primary_swath_model : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Primary swath model.
    burst_resampling_matrices : dict
        Dict where the key is the burst id and the value is a 3x3 affine inverse
        resampling matrix of the burst.
    bursts_metas : List of dicts
        List of metadata of all bursts in a swath (even the ones we are not considering).
    image_readers : List of rasterio.DatasetReader
        Opened rasterio datasets.
    get_complex : boolean, optional
        If set to True, get the complex array. Otherwise, all the processing is conducted
        on the amplitude from the start. The default is True.

    Returns
    -------
    burst_ids : List of ints
        Burst ids in the swath (0 based) associated with each roi\
            of each different burst intersected by the input roi.
    read_rois_no_correc : List of eos.sar.roi.Roi
        Each element is a "read" roi in the ideal primary frame.
    write_rois_no_correc : List of eos.sar.roi.Roi
        Each element is a roi were the resampled data was written.
    debursted_crop : ndarray
        The debursted crop.
    read_rois_correc : List of eos.sar.roi.Roi
        Each element is an roi in the imperfect (primary or secondary) frame.
        It is obtained by warping the input roi and adding a padding within the
        valid image boundaries.
    resamplers : List of eos.products.sentinel1.Sentinel1BurstResample
        Each resampler can be applied directly on the read array with read_rois_correc.


    """
    burst_ids, read_rois_no_correc, write_rois_no_correc, out_shape = primary_swath_model.get_read_write_rois(
        roi_in_swath_no_correc)
    debursted_crop, read_rois_correc, resamplers = warp_rois_read_resample_deburst(read_rois_no_correc, burst_ids, primary_swath_model,
                                                                                   primary_swath_model, burst_resampling_matrices,
                                                                                   bursts_metas, image_readers,
                                                                                   write_rois_no_correc, out_shape,
                                                                                   get_complex)
    return burst_ids, read_rois_no_correc,\
        write_rois_no_correc, debursted_crop, read_rois_correc, resamplers


# filter bursts common to all acquisitions in time series
def get_bursts_intersection(num_bursts, burst_rel_ids):
    """
    Compute the burst id intersection of two swaths containing multiple bursts.

    Parameters
    ----------
    num_bursts : list of int
        List of number of bursts in the swath in time series.
    burst_rel_ids : list of int
        List of relative spatial id of the first burst in the swath in the time series.

    Returns
    -------
    (Nswath, Ncommonbursts) ndarray
        ids of common bursts in the swath per swath
    """
    b_rel_ids = np.array(burst_rel_ids).reshape(-1, 1)
    n_bursts = np.array(num_bursts).reshape(-1, 1)
    rel_min = np.amax(burst_rel_ids)
    rel_max = np.amin(b_rel_ids + n_bursts - 1)
    if rel_min > rel_max:
        print('no intersection', rel_min, rel_max)
        return []
    else:
        list_rel_ids = np.arange(rel_min, rel_max + 1).reshape(1, -1)
        return list_rel_ids - b_rel_ids
