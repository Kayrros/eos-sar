import os

import numpy as np
import pytest

import eos.dem
import eos.products.sentinel1 as s1
import eos.sar
from eos.sar.geoconfig import (
    LOSPredictor,
    convert_arrays_to_enu,
    get_geom_config,
    get_geom_config_from_grid_coords,
    get_grid,
    get_los_on_ellipsoid,
)
from eos.sar.roi import Roi

REF_GEOCONFIG = {
    "pts": np.array(
        [
            [0.0, 0.0],
            [11862.0, 0.0],
            [23724.0, 0.0],
            [0.0, 6097.0],
            [11862.0, 6097.0],
            [23724.0, 6097.0],
            [0.0, 12194.0],
            [11862.0, 12194.0],
            [23724.0, 12194.0],
        ]
    ),
    "inc": np.array(
        [
            0.72536266,
            0.76440833,
            0.79997491,
            0.72508394,
            0.76414932,
            0.79973245,
            0.72480184,
            0.76388718,
            0.79948711,
        ]
    ),
    "bperp": np.array(
        [
            [
                -173.47752162,
                -169.96362793,
                -166.64266185,
                -173.46479529,
                -169.94359065,
                -166.61601424,
                -173.42261988,
                -169.89471236,
                -166.56109973,
            ]
        ]
    ),
    "delta_r": np.array(
        [
            [
                -104.82583265,
                -110.43190335,
                -115.38188407,
                -104.99048797,
                -110.59903964,
                -115.5508141,
                -105.13601113,
                -110.74613258,
                -115.6988936,
            ]
        ]
    ),
}


@pytest.fixture(scope="module")
def models() -> tuple[
    s1.proj_model.Sentinel1SwathModel, s1.proj_model.Sentinel1SwathModel
]:
    xml_folder = "./tests/data"
    xml_basenames = [
        "s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml",
        "s1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml",
    ]
    # list of our xmls
    xml_paths = [os.path.join(xml_folder, p) for p in xml_basenames]
    # read the xmls as strings
    xml_content = []
    for xml_path in xml_paths:
        xml_content.append(eos.sar.io.read_xml_file(xml_path))

    # Now extract the needed metadata
    primary_bursts_meta = s1.metadata.extract_bursts_metadata(xml_content[0])
    secondary_bursts_meta = s1.metadata.extract_bursts_metadata(xml_content[1])

    # get the indices of the common bursts
    prim_burst_ids, sec_burst_ids = s1.deburst.get_bursts_intersection(
        [len(primary_bursts_meta), len(secondary_bursts_meta)],
        [
            primary_bursts_meta[0].relative_burst_id,
            secondary_bursts_meta[0].relative_burst_id,
        ],
    )

    # keep only the bursts intersecting
    primary_bursts_meta = eos.sar.utils.filter_list(primary_bursts_meta, prim_burst_ids)
    secondary_bursts_meta = eos.sar.utils.filter_list(
        secondary_bursts_meta, sec_burst_ids
    )

    primary_orbit = eos.sar.orbit.Orbit(
        s1.metadata.unique_sv_from_bursts_meta(primary_bursts_meta)
    )
    primary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
        primary_bursts_meta, primary_orbit
    )

    secondary_orbit = eos.sar.orbit.Orbit(
        s1.metadata.unique_sv_from_bursts_meta(secondary_bursts_meta)
    )
    secondary_swath_model = s1.proj_model.swath_model_from_bursts_meta(
        secondary_bursts_meta, secondary_orbit
    )

    return primary_swath_model, secondary_swath_model


def test_geoconfig(models):
    primary_swath_model, secondary_swath_model = models
    # Test geoconfig against reference result
    pts, inc, bperp, delta_r = get_geom_config(
        primary_swath_model,
        [
            secondary_swath_model,
        ],
        grid_size_col=3,
        grid_size_row=3,
    )
    np.testing.assert_allclose(pts, REF_GEOCONFIG["pts"])
    np.testing.assert_allclose(inc, REF_GEOCONFIG["inc"])
    np.testing.assert_allclose(bperp, REF_GEOCONFIG["bperp"])
    np.testing.assert_allclose(delta_r, REF_GEOCONFIG["delta_r"])


def test_geom_phase_prediction(models):
    primary_swath_model, secondary_swath_model = models

    primary_swath_roi = Roi(10000, 785, 50, 100)

    dem_source = eos.dem.get_any_source()
    dem = primary_swath_model.fetch_dem(dem_source, primary_swath_roi)

    topo = eos.sar.geom_phase.TopoCorrection(
        primary_swath_model,
        [secondary_swath_model],
        grid_size=50,
        degree=7,
    )

    # predict flat earth
    flat_earth = topo.flat_earth_image(primary_swath_roi, wrapped=True)
    assert flat_earth.shape == (1,) + primary_swath_roi.get_shape(), (
        "flat earth shape mismatch"
    )

    margin = 10
    approx_geom, alts, mask = primary_swath_model.get_approx_geom(
        primary_swath_roi, dem=dem, margin=margin
    )
    # Dem projection in radar coordinates
    heights = eos.sar.dem_to_radar.dem_radarcoding(
        dem,
        primary_swath_model,
        roi=primary_swath_roi,
        approx_geometry=approx_geom,
        margin=margin,
    )

    # predict topographic phase
    topo_phase = topo.topo_phase_image(
        heights, primary_roi=primary_swath_roi, wrapped=False
    )
    assert topo_phase.shape == (1,) + primary_swath_roi.get_shape(), (
        "topo phase shape mismatch"
    )

    # test sparse prediction

    # generate random set of sparse points
    col_orig, row_orig, w, h = primary_swath_roi.to_roi()
    num_pts = 200
    rows = np.random.randint(h, size=num_pts)
    cols = np.random.randint(w, size=num_pts)

    # test sparse flat earth prediction
    sparse_flat = topo.sparse_flat_earth(rows + row_orig, cols + col_orig, wrapped=True)
    np.testing.assert_almost_equal(sparse_flat[0], flat_earth[0][rows, cols])

    # test sparse topo prediction
    sparse_topo = topo.sparse_topo_phase(
        heights[rows, cols], rows + row_orig, cols + col_orig, wrapped=False
    )
    np.testing.assert_almost_equal(sparse_topo[0], topo_phase[0][rows, cols])

    # test on multiple secondary imgs
    list_topo = eos.sar.geom_phase.TopoCorrection(
        primary_swath_model,
        [secondary_swath_model, secondary_swath_model],
        grid_size=50,
        degree=7,
    )

    def check_list_func(list_func, ref_raster, assertion_msg, *args, **kwargs):
        """Check if a function operating on a list of secondary ids yields the same result
        if given different secondary id params assuming the same secondary model is repeated in the
        secondary models list"""
        # force to None if passed
        sec_ids = [None, [0]]
        for i in range(2):
            kwargs["secondary_ids"] = sec_ids[i]
            raster_list = list_func(*args, **kwargs)
            if i:
                assert len(raster_list) == 1, assertion_msg
            else:
                assert len(raster_list) > 1, assertion_msg
            for raster in raster_list:
                # assert np.all(raster==ref_raster), assertion_msg
                np.testing.assert_almost_equal(raster, ref_raster)

    check_list_func(
        list_topo.flat_earth_image,
        flat_earth[0],
        "Error list flat earth img",
        primary_swath_roi,
        wrapped=True,
    )

    check_list_func(
        list_topo.topo_phase_image,
        topo_phase[0],
        "Error list topo phase img",
        heights,
        primary_swath_roi,
        wrapped=False,
    )

    check_list_func(
        list_topo.sparse_flat_earth,
        sparse_flat[0],
        "Error list sparse flat earth",
        rows + row_orig,
        cols + col_orig,
        wrapped=True,
    )

    check_list_func(
        list_topo.sparse_topo_phase,
        sparse_topo[0],
        "Error list sparse topo phase",
        heights[rows, cols],
        rows + row_orig,
        cols + col_orig,
        wrapped=False,
    )


def test_los(models):
    primary_swath_model, _ = models

    los_pred = LOSPredictor.from_proj_model_grid_size(
        primary_swath_model,
        primary_swath_model.w // 100,
        primary_swath_model.h // 100,
        degree=7,
        alt=0.0,
        normalized=True,
        estimate_in_enu=False,
    )
    roi = Roi(2000, 1000, 100, 50)

    # evaluate the polynom
    # get a (50 * 100, 3) array
    evaluated = los_pred.predict_los(
        np.arange(roi.row, roi.row + roi.h),
        np.arange(roi.col, roi.col + roi.w),
        grid_eval=True,
    )

    # to compare, evaluate in a dense fashion with localization
    cols_grid, rows_grid = roi.get_meshgrid()
    # get a (50 * 100, 3) array
    los_precise, _ = get_los_on_ellipsoid(
        primary_swath_model,
        rows_grid.ravel(),
        cols_grid.ravel(),
        alt=0.0,
        normalized=True,
    )

    # compare
    np.testing.assert_allclose(evaluated, los_precise, atol=1e-5)

    # also test normalization on a sparse grid
    cols_grid, rows_grid = get_grid(roi.w, roi.h, grid_size_col=10, grid_size_row=10)

    for ell_alt in [0.0, 2000.0]:
        los_normalized, points_3D = get_los_on_ellipsoid(
            primary_swath_model,
            rows_grid.ravel(),
            cols_grid.ravel(),
            alt=ell_alt,
            normalized=True,
        )
        los_normalized_enu = convert_arrays_to_enu(
            los_normalized, points_3D, ell_alt == 0
        )

        los, points_3D_bis = get_los_on_ellipsoid(
            primary_swath_model,
            rows_grid.ravel(),
            cols_grid.ravel(),
            alt=ell_alt,
            normalized=False,
        )

        assert np.all(points_3D_bis == points_3D)
        los_enu = convert_arrays_to_enu(los, points_3D_bis, ell_alt == 0)

        norm = np.linalg.norm(los, axis=1)
        assert np.all(norm > 0)

        # Check normalization
        np.testing.assert_allclose(los_normalized, los / norm[:, None])

        np.testing.assert_allclose(np.linalg.norm(los_normalized, axis=1), 1)

        # Check that enu conversion preserves the norm
        np.testing.assert_allclose(
            np.linalg.norm(los_normalized, axis=1),
            np.linalg.norm(los_normalized_enu, axis=1),
        )

        np.testing.assert_allclose(
            np.linalg.norm(los, axis=1), np.linalg.norm(los_enu, axis=1)
        )


def test_geom_config_from_grid_coords(models):
    primary_swath_model, secondary_swath_model = models
    pts = REF_GEOCONFIG["pts"]
    cols = pts[:, 0]
    rows = pts[:, 1]
    inc, bperp, delta_r = get_geom_config_from_grid_coords(
        primary_swath_model, [secondary_swath_model], rows, cols
    )

    np.testing.assert_allclose(inc, REF_GEOCONFIG["inc"])
    np.testing.assert_allclose(bperp, REF_GEOCONFIG["bperp"])
    np.testing.assert_allclose(delta_r, REF_GEOCONFIG["delta_r"])

    # test scalar input
    inc_scl, bperp_scl, delta_r_scl = get_geom_config_from_grid_coords(
        primary_swath_model, [secondary_swath_model], rows[0], cols[0]
    )

    assert inc_scl.shape == (1,)
    assert bperp_scl.shape == (1, 1)
    assert delta_r_scl.shape == (1, 1)

    assert inc_scl[0] == inc[0]
    assert bperp_scl[0, 0] == bperp[0, 0]
    assert delta_r_scl[0, 0] == delta_r_scl[0, 0]
