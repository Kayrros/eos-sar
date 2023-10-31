import numpy as np

from eos.products.sentinel1 import burst_resamp
from eos.sar import utils


def warp_rois_read_resample_deburst(
    bsids,
    burst_resamplers,
    within_burst_rois_no_correc,
    secondary_cutter,
    image_readers,
    write_rois,
    out_shape,
    out=None,
    get_complex=True,
    margin=5,
    reramp=True,
):
    """
    Warp the rois, read then resample, and deburst.

    Parameters
    ----------
    bsids : Iterable
        List of BSID of interest.
    burst_resamplers : Dict bsid -> eos.products.sentinel1.burst_resamp.Sentinel1BurstResample
        Eeach element is a resampler pre-set on a primary-secondary burst couple.
    within_burst_rois_no_correc : Dict bsid -> eos.sar.roi.Roi
        Each element is an roi in the ideal primary frame within a burst (referenced to burst outer origin).
    secondary_cutter : eos.products.sentinel1.acquisition.Sentinel1AcquisitionCutter
        Secondary acquisition cutter.
    image_readers : Dict bsid -> rasterio.DatasetReader
        Opened rasterio datasets.
    write_rois : dict bsid -> eos.sar.roi.Roi
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
    reramp : bool
        Set to False to avoid reramping after resampling.

    Returns
    -------
    debursted_crop : ndarray
        The debursted crop.
    read_rois_correc : Dict bsid -> eos.sar.roi.Roi
        Each element is an roi in the imperfect (primary or secondary) frame.
        It is obtained by warping the input roi and adding a padding within the
        valid image boundaries.
    resamplers_on_rois : Dict bsid -> eos.products.sentinel1.Sentinel1BurstResample
        Each resampler was applied on the read array with read_rois_correc.

    """
    read_rois_correc = {}
    resamplers_on_rois = {}

    def gen():
        for bsid in bsids:
            dst_roi_in_burst = within_burst_rois_no_correc[bsid]

            burst_orig_src_in_tiff = secondary_cutter.get_burst_outer_roi_in_tiff(
                bsid
            ).get_origin()

            (
                burst_array_resamp,
                read_roi_src,
                resampler_on_roi,
            ) = burst_resamp.warp_roi_read_resample(
                burst_resamplers[bsid],
                dst_roi_in_burst,
                burst_orig_src_in_tiff,
                image_readers[bsid],
                get_complex,
                margin,
                reramp,
            )

            read_rois_correc[bsid] = read_roi_src
            resamplers_on_rois[bsid] = resampler_on_roi

            yield burst_array_resamp, write_rois[bsid]

    debursted_crop = utils.stitch_arrays(gen(), out_shape, out=out)
    return debursted_crop, read_rois_correc, resamplers_on_rois


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
        print("no intersection", rel_min, rel_max)
        return []
    else:
        list_rel_ids = np.arange(rel_min, rel_max + 1).reshape(1, -1)
        return list_rel_ids - b_rel_ids
