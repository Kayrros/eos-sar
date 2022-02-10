import numpy as np
import os
import shapely
from eos.sar import model, io, roi
from eos.products import sentinel1


def test_localize_without_alt():
    xml_folder = 's3://dev-satellite-test-data/sentinel-1/eos_test_data/annotation'
    basename = 's1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml'
    xml_path = os.path.join(xml_folder, basename)

    # read xml
    xml_content = io.read_xml_file(xml_path)

    burst_id = 1

    # extract the burst metadata
    burst_meta = sentinel1.metadata.extract_burst_metadata(
        xml_content, burst_id)

    # create a Sentinel1BurstModel
    bmod = sentinel1.proj_model.burst_model_from_burst_meta(burst_meta)
    rows = np.round(np.random.rand(5) * 1000)
    cols = np.round(np.random.rand(5) * 20000)
    # test recursively shrinking the interval on a single point
    precision = 1e-1
    amin, amax, adiff_srtm_low, adiff_srtm_high, masks = model.recursive_shrink_interval(
        sensor_model=bmod, row=0, col=0, alt_min=-10000, alt_max=10000,
        num_alt=50, max_iter=10, eps=1e-1, verbosity=False)
    assert (amax > amin) and (amax - amin) < precision
    assert adiff_srtm_low < precision
    assert adiff_srtm_high < precision
    assert masks["converged"].sum() == 1

    # test recursively shrinking the interval on a set of points
    amin, amax, adiff_srtm_low, adiff_srtm_high, masks = model.recursive_shrink_interval(
        sensor_model=bmod, row=rows, col=cols, alt_min=-10000, alt_max=10000,
        num_alt=50, max_iter=10, eps=precision, verbosity=False)
    assert np.all(amax > amin)
    assert np.all((amax - amin) < precision)
    assert np.all(adiff_srtm_low < precision)
    assert np.all(adiff_srtm_high < precision)
    assert masks["converged"].sum() == len(rows)

    # test on a list of points
    lon, lat, alt, masks = bmod.localize_without_alt(
        rows, cols, max_iter=5, eps=precision,
        alt_min=-1000, alt_max=9000, num_alt=100,
        verbosity=False)
    assert len(lon) == len(rows)
    assert masks["converged"].sum() == len(rows)

    rows_pred, cols_pred, _ = bmod.projection(lon, lat, alt)
    np.testing.assert_allclose(rows_pred, rows, atol=1e-2)
    np.testing.assert_allclose(cols_pred, cols, atol=1e-2)

    # test localizing an roi
    roi_geom = roi.Roi(50, 10, 800, 1000)
    # here this is just to get the ground truth bounding points
    rows_roi, cols_roi = roi_geom.to_bounding_points()
    # Localize to get the geometry, altitudes, validity masks
    approx_geom, alts, masks = bmod.get_approx_geom(roi_geom)
    assert len(approx_geom) == 4
    # reproject and compare with the ground truth
    projected = [bmod.projection(*a, alt) for (a, alt) in zip(approx_geom, alts)]

    np.testing.assert_allclose(rows_roi, [p[0] for p in projected], atol=1e-2)
    np.testing.assert_allclose(cols_roi, [p[1] for p in projected], atol=1e-2)

    initial_geom = shapely.geometry.Polygon(bmod.approx_geom)
    refined_geom, alts, mask = bmod.get_approx_geom()
    refined_geom_shp = shapely.geometry.Polygon(refined_geom)
    assert not refined_geom_shp.equals(initial_geom),\
        "Refined geometry is exactly the same as the initial geometry!"
    inter = refined_geom_shp.intersection(initial_geom).area
    union = refined_geom_shp.union(initial_geom).area
    IoU = inter / union
    assert IoU > 0.5, "Refined geometry is too different from initial geometry!"
