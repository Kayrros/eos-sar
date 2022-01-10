import numpy as np
from eos.sar import io, roi
from eos.products.sentinel1 import burst_resamp

def stitch_arrays(rect_arrays, write_rois, out_shape):
    """
    Stitch individual rectangular arrays into an image of known shape, by 
    writing the arrays into the given locations. 

    Parameters
    ----------
    rect_arrays : list of ndarray
        Each element is a rectangular imagette to be used in the mosaic.
    write_rois : list of eos.sar.roi.Roi
        Each instance indicates where we should write the 
        rectangular array in the output image.
    out_shape : tuple
        (h, w) Shape of output image.

    Returns
    -------
    out_img : ndarray
        Output mosaic.

    """
    out_img = np.empty(out_shape, dtype=rect_arrays[0].dtype)
    out_img[:] = np.nan
    for arr, write_roi in zip(rect_arrays, write_rois):
        assert arr.shape == write_roi.get_shape(), "array shape must match write roi shape"
        write_roi.assert_valid(out_shape)
        col_min, row_min, w, h = write_roi.to_roi()
        out_img[row_min:row_min+h, col_min:col_min+w] = arr
    return out_img

def write_array(arr, write_roi, out_shape):
    """
    Write an array using the roi location. 

    Parameters
    ----------
    arr : ndarray
        Array to be written.
    write_roi : eos.sar.roi.Roi
        Roi where array needs to be written.
    out_shape : tuple
        Output array shape.

    Returns
    -------
    out_arr : ndarray
        Shifted output array.

    """
    return stitch_arrays([arr], [write_roi], out_shape)

def get_read_rois_correc_and_resamplers(
        burst_ids, read_rois_no_correc, swath_model_no_correc,
        swath_model_correc, burst_resampling_matrices, bursts_metas_correc):
    """
    Use the burst resampling matrices to get the rois where we need to read\
    and the associated resamplers. 

    Parameters
    ----------
    burst_ids : Iterable
        Burst ids in the swath (0 based) associated with each roi.
    read_rois_no_correc : List of eos.sar.roi.Roi
        Each element is an roi in the ideal primary frame.
    swath_model_no_correc : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Swath model in ideal (primary img) coordinate system.
    swath_model_correc : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Swath model in imperfect coordinate system (primary or secondary img).
    burst_resampling_matrices : dict
        Dict where the key is the burst id and the value is a 3x3 affine inverse
        resampling matrix of the burst.
    bursts_metas_correc : List of dicts
        List of metadata of all bursts in a swath (even the ones we are not considering).

    Returns
    -------
    read_rois_correc : List of eos.sar.roi.Roi
        Each element is an roi in the imperfect (primary or secondary) frame. 
        It is obtained by warping the input roi and adding a padding within the
        valid image boundaries. 
    resamplers : List of eos.products.sentinel1.Sentinel1BurstResample
        Each resampler can be applied directly on the read array with read_rois_correc.

    """
    
    resamplers = []
    read_rois_correc = []
    for j, bid in enumerate(burst_ids): 
        A_resamp = burst_resampling_matrices[bid]
        # get the roi w.r.t. burst origin
        col_dst, row_dst, w_dst, h_dst = swath_model_no_correc.bursts_rois[bid].to_roi()
        dst_roi_in_burst = read_rois_no_correc[j].translate_roi(-col_dst, -row_dst)
        
        # warp the roi
        col_src, row_src, w_src, h_src = swath_model_correc.bursts_rois[bid].to_roi()
        src_roi_in_burst = dst_roi_in_burst.warp_valid_roi(
            (h_dst, w_dst), (h_src, w_src), A_resamp, margin=5)
        
        # burst resampler
        resampler =  burst_resamp.burst_resample_from_meta(
                                                          bursts_metas_correc[bid],
                                                          dst_burst_shape=(
                                                              h_dst, w_dst),
                                                          matrix=A_resamp)
        # set to resample within the burst
        resampler.set_inside_burst(dst_roi_in_burst, src_roi_in_burst)
        
        resamplers.append(resampler)
        
        # Secondary read rois
        read_rois_correc.append(src_roi_in_burst.translate_roi(col_src, row_src))
    return read_rois_correc, resamplers

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
    assert (min(ovl_ids) > -1) and (max(ovl_ids)<len(swath_model.bursts_times) - 1),\
    "Overlap id out of range"
    for ovl_id in ovl_ids: 
        # forward looking overlap
        prev_roi, next_roi = swath_model.overlap_roi(ovl_id)
        ovl_h, ovl_w = prev_roi.get_shape()
        for bid, ovl_roi in zip([ovl_id, ovl_id + 1], [prev_roi, next_roi]): 
            ovl_burst_ids.append(bid)
            col, row, _, _ = swath_model.bursts_rois[bid].to_roi()
            ovl_roi.translate_roi(col, row, inplace=True)
            read_rois.append(ovl_roi)
            out_shapes.append((ovl_h, swath_width))
            write_rois.append(roi.Roi(col - swath_model.col_min, 0, ovl_w, ovl_h))
    return ovl_burst_ids, read_rois, write_rois, out_shapes 



def warp_rois_read_resample(read_rois_no_correc, burst_ids, swath_model_no_correc,
                            swath_model_correc, burst_resampling_matrices,
                            bursts_metas_correc, image_reader,
                            get_complex=True):
    """
    Warp the rois to the imperfect frame, read then resample.

    Parameters
    ----------
    read_rois_no_correc : List of eos.sar.roi.Roi
        Each element is an roi in the ideal primary frame.
    burst_ids : Iterable
        Burst ids in the swath (0 based) associated with each roi.
    swath_model_no_correc : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Swath model in ideal (primary img) coordinate system.
    swath_model_correc : eos.products.sentinel1.proj_model.Sentinel1SwathModel
        Swath model in imperfect coordinate system (primary or secondary img).
    burst_resampling_matrices : dict
        Dict where the key is the burst id and the value is a 3x3 affine inverse
        resampling matrix of the burst.
    bursts_metas_correc : List of dicts
        List of metadata of all bursts in a swath (even the ones we are not considering).
    image_reader : rasterio.DatasetReader
        Opened rasterio dataset.
    get_complex : boolean, optional
        If true, the complex images are read and resampled. The default is True.

    Returns
    -------
    List of arrays (imgs)
        Each element is a roi that has been warped, read, resampled.
    read_rois_correc : List of eos.sar.roi.Roi
        Each element is an roi in the imperfect (primary or secondary) frame. 
        It is obtained by warping the input roi and adding a padding within the
        valid image boundaries. 
    resamplers : List of eos.products.sentinel1.Sentinel1BurstResample
        Each resampler can be applied directly on the read array with read_rois_correc.

    """
    # this is in the only case you just need to read the rois 
    if burst_resampling_matrices is None: 
        return io.read_windows(image_reader, read_rois_no_correc, get_complex),\
            read_rois_no_correc, None
    read_rois_correc, resamplers = get_read_rois_correc_and_resamplers(
        burst_ids, read_rois_no_correc, swath_model_no_correc, swath_model_correc, 
        burst_resampling_matrices, bursts_metas_correc )
    
    padded_burst_arrays = io.read_windows(image_reader, read_rois_correc, get_complex)
    
    burst_arrays_resamp = [resamp.resample(arr) for arr, resamp in zip(padded_burst_arrays, resamplers)]
    
    return burst_arrays_resamp, read_rois_correc, resamplers

def warp_rois_read_resample_deburst(read_rois_no_correc, burst_ids, primary_swath_model,
                            secondary_swath_model, burst_resampling_matrices,
                            secondary_bursts_metas, image_reader,
                            write_rois, out_shape,
                            get_complex=True): 
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
    image_reader : rasterio.DatasetReader
        Opened rasterio dataset.
    write_rois : List of eos.sar.roi.Roi
        Each element defines the roi to write the data in the output array.
    out_shape : tuple
        Output array shape.
    get_complex : boolean, optional
        If set to True, get the complex array. Otherwise, all the processing is conducted
        on the amplitude from the start. The default is True.

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
    burst_arrays_resamp, read_rois_correc, resamplers = warp_rois_read_resample(read_rois_no_correc, burst_ids, primary_swath_model,
                                secondary_swath_model, burst_resampling_matrices,
                                secondary_bursts_metas, image_reader,
                                get_complex)
    debursted_crop = stitch_arrays(burst_arrays_resamp, write_rois, out_shape)
    return debursted_crop, read_rois_correc, resamplers

def deburst_primary(roi_in_swath_no_correc, primary_swath_model,
                    burst_resampling_matrices, bursts_metas, image_reader,
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
    image_reader : rasterio.DatasetReader
        Opened rasterio dataset.
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
                                bursts_metas, image_reader,
                                write_rois_no_correc, out_shape,
                                get_complex)
    return burst_ids, read_rois_no_correc,\
 write_rois_no_correc, debursted_crop, read_rois_correc, resamplers
    
def warp_rois_read_resample_ovl(primary_swath_model, secondary_swath_model,
                                secondary_bursts_metas, burst_resampling_matrices, 
                                ovl_burst_ids, read_rois_no_correc,
                                write_rois,  out_shapes, image_reader, get_complex=True): 
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
    burst_arrays_resamp, read_rois_correc, resamplers =  warp_rois_read_resample(
        read_rois_no_correc, ovl_burst_ids, primary_swath_model,
        secondary_swath_model, burst_resampling_matrices,
        secondary_bursts_metas, image_reader,
        get_complex)
    burst_arrays_resamp = [write_array(arr, write_roi, out_shape)\
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
    
    ovl_burst_ids, read_rois_no_correc, write_rois, out_shapes = get_overlaps(primary_swath_model, ovl_ids)
    
    burst_arrays_resamp_prim, read_rois_correc, resamplers = warp_rois_read_resample_ovl(
        primary_swath_model, primary_swath_model,
        primary_burst_metas, burst_resampling_matrices, 
        ovl_burst_ids, read_rois_no_correc,
        write_rois, out_shapes, image_reader, get_complex)
    
    return ovl_burst_ids, read_rois_no_correc, write_rois, out_shapes,\
        burst_arrays_resamp_prim, read_rois_correc, resamplers

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
    n_bursts =  np.array(num_bursts).reshape(-1, 1)
    rel_min = np.amax(burst_rel_ids)
    rel_max = np.amin(b_rel_ids + n_bursts -1)
    if rel_min > rel_max:
        print('no intersection', rel_min, rel_max)
        return []
    else:
        list_rel_ids = np.arange(rel_min, rel_max + 1).reshape(1, -1)
        return list_rel_ids - b_rel_ids
