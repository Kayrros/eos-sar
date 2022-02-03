from eos.sar import roi, utils
from eos.products.sentinel1 import burst_resamp


def get_overlaps(swath_model, ovl_ids):
    """
    Get the overlap info (between consecutive bursts) in a swath model for a\
        set of overlap ids.

    Parameters
    ----------
    swath_model : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Swath were burst overlaps need to be computed.
    ovl_ids : Iterable of int
        overlap ids.

    Returns
    -------
    ovl_burst_ids : list
        Burst id associated with the overlap.
    read_rois : list
        Read roi of the overlap in the swath.
    write_rois : list
        Write roi in the final overlap array
    out_shapes : list
        Each element is a (overalp_height, swath_width) tuple.

    """
    read_rois = []
    ovl_burst_ids = []
    out_shapes = []
    write_rois = []
    swath_width = swath_model.w
    assert (min(ovl_ids) > -1) and (max(ovl_ids) < len(swath_model.bursts_times) - 1),\
        "Overlap id out of range"
    for ovl_id in ovl_ids:
        # forward looking overlap
        prev_roi, next_roi = swath_model.overlap_roi(ovl_id)
        for bid, ovl_roi in zip([ovl_id, ovl_id + 1], [prev_roi, next_roi]):
            ovl_burst_ids.append(bid)
            col, row, _, _ = swath_model.bursts_rois[bid].to_roi()
            ovl_roi.translate_roi(col, row, inplace=True)
            read_rois.append(ovl_roi)
            ovl_h, ovl_w = ovl_roi.get_shape()
            out_shapes.append((ovl_h, swath_width))
            write_rois.append(
                roi.Roi(col - swath_model.col_min, 0, ovl_w, ovl_h))
    return ovl_burst_ids, read_rois, write_rois, out_shapes


def warp_rois_read_resample_ovl(primary_swath_model, secondary_swath_model,
                                secondary_bursts_metas, burst_resampling_matrices,
                                ovl_burst_ids, read_rois_no_correc,
                                write_rois,  out_shapes, image_reader,
                                get_complex=True, margin=5):
    """
    Warp overlap rois, read, resample.

    Parameters
    ----------
    primary_swath_model : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Primary swath model.
    secondary_swath_model : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Secondary swath model.
    secondary_bursts_metas : List of dicts
        List of metadata of all bursts in a swath (even the ones we are not considering).
    burst_resampling_matrices : dict
        Dict where the key is the burst id and the value is a 3x3 affine inverse
        resampling matrix of the burst.
    ovl_burst_ids : list
        Burst id associated with the overlap..
    read_rois_no_correc : List of eos.sar.roi.Roi
        Each elem is an roi of an overlap in the ideal frame.
    write_rois : list of eos.sar.roi.Roi
        Roi to write the ovl in the final array.
    out_shapes : list of tuple
        Output shapes of the overlaps.
    image_reader : rasterio.DatasetReader
        Opened rasterio dataset.
    get_complex : boolean, optional
        If set to True, get the complex array. Otherwise, all the processing is conducted
        on the amplitude from the start. The default is True.
    margin : int, optional
        Pixel safety margin to be applied after warping read_rois_no_correc. The default is 5.

    Returns
    -------
    burst_arrays_resamp : List
        Each element is a resampled overlap img.
    read_rois_correc : List of eos.sar.roi.Roi
        Each element is an roi in the imperfect (primary or secondary) frame.
        It is obtained by warping the input roi and adding a padding within the
        valid image boundaries.
    resamplers : List of eos.products.sentinel1.Sentinel1BurstResample
        Each resampler can be applied directly on the read array with read_rois_correc.

    """
    burst_arrays_resamp = []
    read_rois_correc = []
    resamplers = []

    for read_roi_dst, bid in zip(read_rois_no_correc, ovl_burst_ids):
        burst_roi_dst = primary_swath_model.bursts_rois[bid]
        burst_roi_src = secondary_swath_model.bursts_rois[bid]

        burst_array_resamp, read_roi_src, resampler = burst_resamp.warp_roi_read_resample(
            burst_resampling_matrices[bid], burst_roi_dst, read_roi_dst, burst_roi_src,
            secondary_bursts_metas[bid], image_reader, get_complex, margin)

        burst_arrays_resamp.append(burst_array_resamp)
        read_rois_correc.append(read_roi_src)
        resamplers.append(resampler)

    burst_arrays_resamp = [utils.write_array(arr, write_roi, out_shape)
                           for arr, write_roi, out_shape in zip(
        burst_arrays_resamp, write_rois, out_shapes)]
    return burst_arrays_resamp, read_rois_correc, resamplers


def warp_rois_read_resample_ovl_primary(primary_swath_model, burst_resampling_matrices,
                                        primary_burst_metas, image_reader, ovl_ids, get_complex=True):
    """
    Determine rois of overalps, warp, read and resample.

    Parameters
    ----------
    primary_swath_model : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Primary swath model.
    burst_resampling_matrices : dict
        Dict where the key is the burst id and the value is a 3x3 affine inverse
        resampling matrix of the burst.
    primary_burst_metas : List of dicts
        List of metadata of all bursts in a swath (even the ones we are not considering).
    image_reader : rasterio.DatasetReader
        Opened rasterio dataset.
    ovl_ids : Iterable of ints
        Overlap ids.
    get_complex : boolean, optional
        If set to True, get the complex array. Otherwise, all the processing is conducted
        on the amplitude from the start. The default is True.

    Returns
    -------
    ovl_burst_ids : list
        Burst id associated with the overlap.
    read_rois : list
        Read roi of the overlap in the swath.
    write_rois : list
        write rois for the overlaps.
    out_shapes : list
        Each element is a (overalp_height, swath_width) tuple.
    burst_arrays_resamp_prim : List of arrays
        Each elem is a resampled overlap img.
    read_rois_correc : List of eos.sar.roi.Roi
        Each element is an roi in the imperfect (primary or secondary) frame.
        It is obtained by warping the input roi and adding a padding within the
        valid image boundaries.
    resamplers : List of eos.products.sentinel1.Sentinel1BurstResample
        Each resampler can be applied directly on the read array with read_rois_correc.

    """

    ovl_burst_ids, read_rois_no_correc, write_rois, out_shapes = get_overlaps(
        primary_swath_model, ovl_ids)

    burst_arrays_resamp_prim, read_rois_correc, resamplers = warp_rois_read_resample_ovl(
        primary_swath_model, primary_swath_model,
        primary_burst_metas, burst_resampling_matrices,
        ovl_burst_ids, read_rois_no_correc,
        write_rois, out_shapes, image_reader, get_complex)

    return ovl_burst_ids, read_rois_no_correc, write_rois, out_shapes,\
        burst_arrays_resamp_prim, read_rois_correc, resamplers
