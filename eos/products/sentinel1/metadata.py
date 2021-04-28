def fill_meta(model, bid):
    """
    Parameters
    ----------
    model : Sentinel1Model instance
        Instance created using s1m module on a subswath of a product.
    bid : integer
        Burst index in the swath that corresponds to model.

    Returns
    -------
    burst_metadata : dict
        Metadata necessary for further burst processing.

    """
    burst_metadata = {}
    burst_metadata['state_vectors'] = model.state_vectors
    burst_metadata['burst_times'] = model.burst_times[bid]
    burst_metadata['slant_range_time'] = model.slant_range_time
    burst_metadata['azimuth_frequency'] = model.azimuth_frequency
    burst_metadata['range_frequency'] = model.range_frequency
    burst_metadata['burst_roi'] = model.burst_rois[bid]
    burst_metadata['lines_per_burst'] = model.lines_per_burst
    burst_metadata['samples_per_burst'] = model.samples_per_burst
    burst_metadata['azimuth_anx_time'] = model.burst_azimuth_anx_times[bid]
    burst_metadata['approx_geom'] = model.burst_lon_lat_bboxes[bid]
    return burst_metadata
