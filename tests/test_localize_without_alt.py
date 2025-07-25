import logging
import os

import numpy as np
import pytest
import shapely.geometry

import eos.dem
from eos.products import sentinel1
from eos.sar import io, model, roi
from eos.sar.orbit import Orbit


@pytest.fixture(scope="module")
def bmod_dem():
    xml_folder = "./tests/data"
    basename = "s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml"
    xml_path = os.path.join(xml_folder, basename)

    # read xml
    xml_content = io.read_xml_file(xml_path)

    burst_id = 1

    # extract the burst metadata
    burst_meta = sentinel1.metadata.extract_burst_metadata(xml_content, burst_id)

    # create an orbit
    orbit = Orbit(burst_meta.state_vectors)
    # create a Sentinel1BurstModel
    bmod = sentinel1.proj_model.burst_model_from_burst_meta(burst_meta, orbit)
    dem_source = eos.dem.get_any_source()
    alt_min = -10000
    alt_max = 10000
    dem = bmod.fetch_dem(dem_source, alt_min=alt_min, alt_max=alt_max)

    return bmod, dem


def test_localize_without_alt(bmod_dem):
    bmod, dem = bmod_dem

    rows = np.round(np.random.rand(5) * 1000)
    cols = np.round(np.random.rand(5) * 20000)
    # test recursively shrinking the interval on a single point
    precision = 1e-1

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
        alt_min=-10000,
        alt_max=10000,
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
    assert coarse_approx_geom_shp.contains(approx_geom_shp), (
        "coarse_approx_geom does not contain the approx_geom"
    )


def test_invalid_localize_without_alt(bmod_dem, caplog):
    bmod, dem = bmod_dem

    # create failure cases

    # Here we expect the point to be to far outside of the burst
    # that it would not intersect the dem that was download on the burst approximate geometry
    # We expect nans in the result, since nans appear when the points fall outside the DEM
    row = col = -1e5
    lon, lat, alt, masks = bmod.localize_without_alt(row, col, dem=dem)
    assert np.isnan(lon)
    assert np.isnan(lat)
    assert np.isnan(alt)
    assert masks["invalid"]

    # Here we expect a failure just because the search interval is too small
    # So the result will not be nan, but will be invalid nonetheless
    lon, lat, alt, masks = bmod.localize_without_alt(
        0, 0, dem=dem, alt_min=-1, alt_max=1
    )
    assert not np.isnan(lon)
    assert not np.isnan(lat)
    assert not np.isnan(alt)
    assert masks["invalid"]

    # take point near boundary of dem
    # Here we take near the lower right point because upper left is nan(water)
    # !!!!!! The test depends on the input image/DEM !!!!!!!
    h, w = dem.array.shape

    # integer offset of lower right point
    offset = 3
    lon, lat = dem.transform * (w - offset - 0.5, h - offset - 0.5)
    alt = dem.array[-offset, -offset]

    # project it
    row, col, _ = bmod.projection(lon, lat, alt)
    # then try localize without alt
    # Decrease the sampling of points to provoke a failure
    lon, lat, alt, masks = bmod.localize_without_alt(row, col, dem=dem, num_alt=10)
    # The result should contain nan because the failure is due to exceeding DEM bounds
    assert np.isnan(lon)
    assert np.isnan(lat)
    assert np.isnan(alt)
    assert masks["invalid"]

    h, w = bmod.h, bmod.w
    # localize some points in the middle
    # select a region in the middle of the burst to work with
    size = 200  # 200x200 Roi
    roi_mid = roi.Roi(w // 2 - size // 2, h // 2 - size // 2, size, size)

    alt_min = np.nanmin(dem.array)
    alt_max = np.nanmax(dem.array)

    geometry = shapely.geometry.Polygon(
        bmod.get_buffered_geom(dem, roi_mid, margin=0, alt_min=alt_min, alt_max=alt_max)
    )

    # subset the dem
    dem_subset = dem.subset(geometry.bounds)

    with caplog.at_level(logging.WARNING):
        geometry_with_subset = shapely.geometry.Polygon(
            bmod.get_buffered_geom(
                dem_subset, roi_mid, margin=0, alt_min=alt_min, alt_max=alt_max
            )
        )
        assert "some points may be invalid" in caplog.text

    # Here you might fail to localize some points but you get away
    IoU = (
        geometry.intersection(geometry_with_subset).area
        / geometry.union(geometry_with_subset).area
    )
    assert IoU > 0.5, (
        "roi geometry with subseted dem is too different from the on with full dem"
    )

    with pytest.raises(AssertionError):
        # When you add a margin, you should not be able to localize any point
        geometry_with_subset = shapely.geometry.Polygon(
            bmod.get_buffered_geom(
                dem_subset, roi_mid, margin=200, alt_min=alt_min, alt_max=alt_max
            )
        )

    size = 100  # 100x100 roi inside mid_roi
    inner_roi = roi.Roi(w // 2 - size // 2, h // 2 - size // 2, size, size)

    # For inner_roi, should not have a problem for both dems, and should get the same result
    geometry = shapely.geometry.Polygon(
        bmod.get_buffered_geom(
            dem, inner_roi, margin=0, alt_min=alt_min, alt_max=alt_max
        )
    )

    # reset logs
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        geometry_with_subset = shapely.geometry.Polygon(
            bmod.get_buffered_geom(
                dem_subset, inner_roi, margin=0, alt_min=alt_min, alt_max=alt_max
            )
        )
        assert "some points may be invalid" not in caplog.text

    IoU = (
        geometry.intersection(geometry_with_subset).area
        / geometry.union(geometry_with_subset).area
    )

    assert IoU == 1, "localization of same roi with two dems gave different result"

    # Test approx geom
    approx_geom, alts, masks = bmod.get_approx_geom(roi_mid, margin=0, dem=dem)
    assert not np.any(masks["invalid"])

    # Test approx geom with dem_subset
    # should fail because DEM too small and localizing corner points is risky
    with pytest.raises(AssertionError):
        bmod.get_approx_geom(roi_mid, margin=0, dem=dem_subset)

    # Test approx geom with dem_subset and inner_roi
    approx_geom, alts, masks = bmod.get_approx_geom(inner_roi, margin=0, dem=dem_subset)
    assert not np.any(masks["invalid"])
