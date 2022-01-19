import os
import json
import numpy as np
from eos.sar import io, dem_to_radar, regist, roi
from eos.products import sentinel1

def main(roi_path, margin=100):
    roiinfo = json.load(open(roi_path))

    output = roi_path.replace('.json', '_dem.tiff')
    product_id = roiinfo['product_id']
    mid = roiinfo['measurement_id']
    xml_path = os.path.join(f"{product_id}.SAFE", "annotation", f"{mid}.xml")
    burst_id = roiinfo['burst_id']
    window = roiinfo['window']

    # extract the burst metadata
    xml_content = io.read_xml_file(xml_path)
    burst_meta = sentinel1.metadata.extract_burst_metadata(xml_content, burst_id)

    # create a Sentinel1BurstModel
    bmod = sentinel1.proj_model.burst_model_from_burst_meta(burst_meta, intra_pulse_correction=True)

    # get a good approximation of the geometry of the burst with a margin
    refined_geom, _, _ = bmod.get_approx_geom(margin=margin)

    # get a dem on the previously estimated geometry
    _, _, raster, transform, _ = regist.dem_points(refined_geom, outfile=output)

    return
    # the rest is radar coding, might be useful

    # define a region of interest where geocoding should occur
    crop_roi = roi.Roi(*window)

    # estimate altitude only on roi
    # the approximate geometry is implicitly re-estimated (since not passed as param)
    crop_alt = dem_to_radar.dem_radarcoding(raster, transform, bmod,
            roi=crop_roi,
            margin=margin,
            get_xy=False)
    assert not np.any(np.isnan(crop_alt)), 'NaN detected, perhaps increase the margin ?'
    assert crop_alt.shape == crop_roi.get_shape()

if __name__ == '__main__':
    import fire
    fire.Fire(main)

