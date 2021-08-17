import numpy as np
from eos.sar import io, regist, roi
from eos.products.sentinel1 import burst_resamp


def stitch_arrays(rect_arrays, write_rois, out_shape):
    """
    Stitch individual rectangular arrays into an image of known shape, by 
    writing the arrays into the given locations. 

    Parameters
    ----------
    rect_arrays : list of ndarray
        Each element is a rectangular imagette to be used in the mosaic.
    write_rois : list of tuples
        Each tuple (col, row, w, h) indicates where we should write the 
        rectangular array in the output image.
    out_shape : tuple
        (h, w) Shape of output image.

    Returns
    -------
    out_img : ndarray
        Output mosaic.

    """
    out_img = np.zeros(out_shape, dtype=rect_arrays[0].dtype)
    for arr, w_roi in zip(rect_arrays, write_rois):
        col_min, row_min, w, h = w_roi
        out_img[row_min:row_min+h, col_min:col_min+w] = arr
    return out_img


def deburst_in_primary_swath(primary_swath_model, image_reader, roi_in_swath=None):
    """
    Compute debursted crop inside the swath/of the whole swath of a primary image. 

    Parameters
    ----------
    primary_swath_model : Sentinel1SwathModel
        SwathModel of the image. It is supposed to be the primary frame in the 
        context of interferometry, since no resampling is performed. 
    image_reader : rasterio.DatasetReader
            opened image
    roi_in_swath : tuple
        (col, row, w, h) Region to deburst inside a swath in swath coordinates.
        If None, the whole swath is taken. The default is None. 
    Returns
    -------
    debursted_crop : ndarray
        Debursted crop containing a mosaic of arrays extracted from the different 
        bursts in the swath.
    burst_ids : list of int
            Ids of the bursts intersected by the roi.
    rois_read : list of tuples
        Each tuple (col, row, w, h) corresponds to the region to be read from 
        the tiff file.
    rois_write : list of tuples
        Each tuple (col, row, w, h) corresponds to the region where the output
        data should be written in the output image.
    """
    burst_ids, rois_read, rois_write, out_shape = primary_swath_model.get_read_write_rois(
        roi_in_swath)
    burst_arrays = io.read_windows(image_reader, rois_read)
    debursted_crop = stitch_arrays(burst_arrays, rois_write, out_shape)
    return debursted_crop, burst_ids, rois_read, rois_write


def secondary_rois_and_resamplers(primary_swath_model, rois_read, burst_ids,
                                   secondary_swath_model, secondary_bursts_meta,
                                   matrix):
    """
    Compute the regions of interest in the secondary image and the resamplers 
    associated from given regions in the primary image (each one being inside a burst). 
    The registration matrix is used, along with the secondary image metdata for resampling. 
    
    Parameters
    ----------
    primary_swath_model : Sentinel1SwathModel
        Model for the swath of the primary acquisition.
    rois_read : list of tuples
        Each tuple (col, row, w, h) is a location to be read from the primary 
        tiff within the burst of corresponding id.
    burst_ids : list of int
        Ids of bursts associated with rois_read.
    secondary_swath_model : Sentinel1SwathModel
        Model for the swath of the secondary acquisition.
    secondary_bursts_meta : List of dict
        Each dict contains the metadata of a burst. The list covers all the bursts
        of the swath, even those outside the scope of the given burst_ids. Indexing
        should be consistent with burst_ids. 
    matrix : ndarray (3,3)
        Inverse resampling matrix between the primary swath to the secondary swath.

    Returns
    -------
    Secondary_rois_read: list of tuples
        Each tuple (col, row, w, h) is a location to be read from the secondary image.
    Resamplers: list of Sentinel1BurstResample
        Each resampler is associated with the corresponding roi in Secondary_read_roi

    """
    resamplers = []
    secondary_rois_read = []

    for j in range(len(burst_ids)):

        # adapt the resampling matrix
        col_dst, row_dst = primary_swath_model.burst_orig_in_swath(
            burst_ids[j])
        col_src, row_src = secondary_swath_model.burst_orig_in_swath(
            burst_ids[j])
        A_resamp = regist.change_resamp_mat_orig(
            row_dst, col_dst, row_src, col_src, matrix)

        # get the roi w.r.t. burst origin
        col_dst, row_dst, w_dst, h_dst = primary_swath_model.bursts_rois[burst_ids[j]]
        dst_roi_in_burst = roi.translate_roi(rois_read[j], -col_dst, -row_dst)

        # warp the roi
        col_src, row_src, w_src, h_src = secondary_swath_model.bursts_rois[burst_ids[j]]
        src_roi_in_burst = roi.warp_valid_rois(dst_roi_in_burst, (h_dst, w_dst),
                                               (h_src, w_src),
                                               A_resamp, margin=5)

        # burst resampler
        resampler = burst_resamp.burst_resample_from_meta(secondary_bursts_meta[burst_ids[j]],
                                                          dst_burst_shape=(
                                                              h_dst, w_dst),
                                                          matrix=A_resamp, degree=11)
        # set to resample within the burst
        resampler.set_inside_burst(dst_roi_in_burst, src_roi_in_burst)

        resamplers.append(resampler)

        # Secondary read rois
        secondary_rois_read.append(roi.translate_roi(
            src_roi_in_burst, col_src, row_src))
        
    return secondary_rois_read, resamplers

def read_resample_and_deburst(secondary_image_reader, secondary_rois_read, 
                              resamplers, rois_write, out_shape): 
    """
    Read rois from secondary, resample the complex images with deramping/reramping, 
    and deburst into a final stitched image. 

    Parameters
    ----------
    secondary_image_reader : rasterio.DatasetReader
        Opened secondary image
    secondary_rois_read : list of tuples
        Each tuple (col, row, w, h) is a location to read from in the secondary tiff.
    resamplers : list of Sentinel1BurstResample
        Each resampler is associated with the corresponding roi in Secondary_read_roi.
    rois_write : list of tuples
        Each tuple (col, row, w, h) is a location to write at in the output image.
    out_shape : tuple
        (h, w) output image shape.

    Returns
    -------
    secondary_debursted_crop : ndarray
        Debursted secondary image.

    """    
    
    burst_arrays = io.read_windows(secondary_image_reader, secondary_rois_read)
    burst_arrays = [resamp.resample(arr) for arr,resamp in zip(burst_arrays, resamplers)]
    secondary_debursted_crop = stitch_arrays(burst_arrays, rois_write, out_shape)
    return secondary_debursted_crop

def get_bursts_intersection(num_bursts1, burst_rel_id1, num_bursts2, burst_rel_id2): 
    """
    Compute the burst id intersection of two swaths containing multiple bursts. 

    Parameters
    ----------
    num_bursts1 : int
        Number of bursts in the first swath.
    burst_rel_id1 : int
        Relative spatial id of the first burst in the first swath.
    num_bursts2 : int
        Number of bursts in the second swath. 
    burst_rel_id2 : int
        Relative spatial id of the first burst in the second swath.

    Returns
    -------
    Iterable
        Each element is a burst id in the first swath in the intersection.
    Iterable
        Each element is a burst id in the second swath in the intersection.
        
    Notes
    -----
     If the metadata of the bursts are stored in a list of dict burst_metas, 
     num_bursts can be retrieved as len(burst_metas)
     burst_rel_id can be retrieved as burst_metas[0]['relative_burst_id']
    """
    
    rel_min = max(burst_rel_id1, burst_rel_id2)
    rel_max = min(burst_rel_id1 + num_bursts1 - 1, burst_rel_id2 + num_bursts2 - 1)
    if rel_min > rel_max:
        print('no intersection')
        return [], []
    else: 
        list_rel_ids = np.arange(rel_min, rel_max + 1)
        return list_rel_ids - burst_rel_id1, list_rel_ids - burst_rel_id2
        