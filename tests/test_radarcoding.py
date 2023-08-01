import os
import numpy as np
from eos.sar import io, dem_to_radar, regist, roi
from eos.products import sentinel1
from eos.sar.orbit import Orbit, StateVector


def test_radar_coding():
    remote_test = True

    if remote_test:
        xml_folder = 's3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/annotation'
    else:
        xml_folder = '../tests/data'

    basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml'
    xml_path = os.path.join(xml_folder, basename)

    # read xml
    xml_content = io.read_xml_file(xml_path)

    burst_id = 1

    # extract the burst metadata
    burst_meta = sentinel1.metadata.extract_burst_metadata(
        xml_content, burst_id)

    # create an orbit
    orbit = Orbit(burst_meta.state_vectors)
    # create a doppler
    doppler = sentinel1.doppler_info.doppler_from_meta(burst_meta, orbit)
    # create a corrector
    corrector = sentinel1.coordinate_correction.s1_corrector_from_meta(
        burst_meta, orbit, doppler, apd=True, bistatic=True, intra_pulse=True,
        alt_fm_mismatch=True)

    # create a Sentinel1BurstModel
    bmod = sentinel1.proj_model.burst_model_from_burst_meta(
        burst_meta, orbit, corrector)

    margin = 10

    # get a good approximation of the geometry of the burst
    # with a margin of 10 pixels
    refined_geom, alts, mask = bmod.get_approx_geom(margin=margin)

    # get a dem on the previously estimated geometry
    x, y, raster, transform, crs = regist.dem_points(refined_geom)

    # define a region of interest where geocoding should occur
    crop_roi = roi.Roi(6000, 80, 500, 200)
    # estimate altitude only on roi
    # the approximate geometry is implicitly re-estimated (since not passed as param)
    crop_alt = dem_to_radar.dem_radarcoding(raster, transform, bmod,
                                            roi=crop_roi,
                                            margin=margin,
                                            get_xy=True)
    assert not np.any(np.isnan(crop_alt)), 'NaN detected, perhaps increase the margin ?'
    assert crop_alt.shape == (*crop_roi.get_shape(), 3)
