import os

import numpy as np

import eos.dem
import eos.products.sentinel1 as s1
import eos.sar
from eos.sar.roi import Roi


def test_geom_phase_prediction(s3_client):
    xml_folder = (
        "s3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/annotation"
    )
    xml_basenames = [
        "s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml",
        "s1a-iw3-slc-vv-20190809t164050-20190809t164115-028495-033896-006.xml",
    ]
    # list of our xmls
    xml_paths = [os.path.join(xml_folder, p) for p in xml_basenames]
    # read the xmls as strings
    xml_content = []
    for xml_path in xml_paths:
        xml_content.append(eos.sar.io.read_xml_file(xml_path, s3_client))

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
    # %%
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
