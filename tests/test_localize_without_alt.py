import os

import numpy as np
import shapely.geometry

import eos.dem
from eos.products import sentinel1
from eos.sar import io, model, roi
from eos.sar.orbit import Orbit


def test_localize_without_alt(s3_client):
    xml_folder = (
        "s3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/annotation"
    )
    basename = "s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml"
    xml_path = os.path.join(xml_folder, basename)

    # read xml
    xml_content = io.read_xml_file(xml_path, s3_client)

    burst_id = 1

    # extract the burst metadata
    burst_meta = sentinel1.metadata.extract_burst_metadata(xml_content, burst_id)

    # create an orbit
    orbit = Orbit(burst_meta.state_vectors)
    # create a Sentinel1BurstModel
    bmod = sentinel1.proj_model.burst_model_from_burst_meta(burst_meta, orbit)
    rows = np.round(np.random.rand(5) * 1000)
    cols = np.round(np.random.rand(5) * 20000)
    # test recursively shrinking the interval on a single point
    precision = 1e-1
    dem_source = eos.dem.get_any_source()
    alt_min = -10000
    alt_max = 10000
    dem = bmod.fetch_dem(dem_source, alt_min=alt_min, alt_max=alt_max)
    (
        amin,
        amax,
        adiff_srtm_low,
        adiff_srtm_high,
        masks,
    ) = model.recursive_shrink_interval(
        sensor_model=bmod,
        row=0,
        col=0,
        alt_min=alt_min,
        alt_max=alt_max,
        num_alt=50,
        max_iter=10,
        eps=1e-1,
        verbosity=False,
        dem=dem,
    )
    assert (amax > amin) and (amax - amin) < precision
    assert adiff_srtm_low < precision
    assert adiff_srtm_high < precision
    assert masks["converged"].sum() == 1

    # test recursively shrinking the interval on a set of points
    (
        amin,
        amax,
        adiff_srtm_low,
        adiff_srtm_high,
        masks,
    ) = model.recursive_shrink_interval(
        sensor_model=bmod,
        row=rows,
        col=cols,
        alt_min=-10000,
        alt_max=10000,
        num_alt=50,
        max_iter=10,
        eps=precision,
        verbosity=False,
        dem=dem,
    )
    assert np.all(amax > amin)
    assert np.all((amax - amin) < precision)
    assert np.all(adiff_srtm_low < precision)
    assert np.all(adiff_srtm_high < precision)
    assert masks["converged"].sum() == len(rows)

    # test on a list of points
    lon, lat, alt, masks = bmod.localize_without_alt(
        rows,
        cols,
        max_iter=5,
        eps=precision,
        alt_min=-1000,
        alt_max=9000,
        num_alt=100,
        verbosity=False,
        dem=dem,
    )
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
    approx_geom, alts, masks = bmod.get_approx_geom(roi_geom, dem=dem)
    assert len(approx_geom) == 4
    # reproject and compare with the ground truth
    projected = [
        bmod.projection(lon, lat, alt) for ((lon, lat), alt) in zip(approx_geom, alts)
    ]

    np.testing.assert_allclose(rows_roi, [p[0] for p in projected], atol=1e-2)
    np.testing.assert_allclose(cols_roi, [p[1] for p in projected], atol=1e-2)

    # compare get_approx_geom and get_buffered_geom
    # the two geometries should be quite similar
    approx_geom, _, _ = bmod.get_approx_geom(dem=dem)
    buffered_geom = bmod.get_buffered_geom(dem=dem)
    approx_geom_shp = shapely.geometry.Polygon(approx_geom)
    buffered_geom_shp = shapely.geometry.Polygon(buffered_geom)
    inter = approx_geom_shp.intersection(buffered_geom_shp).area
    union = approx_geom_shp.union(buffered_geom_shp).area
    IoU = inter / union
    assert IoU > 0.9, "Buffered geometry is too different from the approx geometry"

    # compare get_coarse_approx_geom and get_approx_geom
    # the approx geom should be included in the coarse approx geom
    coarse_approx_geom = bmod.get_coarse_approx_geom(
        margin=10, alt_min=-1000.0, alt_max=9000.0
    )
    coarse_approx_geom_shp = shapely.geometry.Polygon(coarse_approx_geom)
    assert coarse_approx_geom_shp.contains(
        approx_geom_shp
    ), "coarse_approx_geom does not contain the approx_geom"
