from math import ceil, floor

import boto3
import numpy as np
import pandas as pd
import pytest

from eos.products.nisar import metadata
from eos.products.nisar.proj_model import NisarModel
from eos.sar.fourier_zoom import fourier_zoom
from eos.sar.io import open_netcdf_osio
from eos.sar.max_finding import sub_pixel_maxima
from eos.sar.orbit import Orbit
from eos.sar.roi import Roi

RSLC_SAMPLE_PATHS = [
    "s3://kayrros-dev-satellite-test-data/NISAR/simulated_samples/l1_rslc/sample1/NISAR_L1_PR_RSLC_001_030_A_019_2000_SHNA_A_20081012T060910_20081012T060926_D00402_N_F_J_001.h5",
    "s3://kayrros-dev-satellite-test-data/NISAR/simulated_samples/l1_rslc/sample2/NISAR_L1_PR_RSLC_002_030_A_019_2800_SHNA_A_20081127T061000_20081127T061014_D00404_N_F_J_001.h5",
]

# Expected position error statistics (in pixels) for the Rosamond corner reflectors
# documented in the Quality Assurance (QA) report provided for the two RSLC samples.
RSLC_SAMPLE_CORNER_REFLECTOR_STATS = {
    "RSLC_001": {
        "azimuth_offset_mean": -0.6294785577830715,
        "azimuth_offset_std": 0.016246651901043817,
        "azimuth_offset_min": -0.659733530075755,
        "azimuth_offset_max": -0.6012110629235394,
        "range_offset_mean": -1.3966073395058278,
        "range_offset_std": 0.015166019942838688,
        "range_offset_min": -1.4279259177578751,
        "range_offset_max": -1.3647166922087308,
    },
    "RSLC_002": {
        "azimuth_offset_mean": -0.637960597514533,
        "azimuth_offset_std": 0.014700779748502129,
        "azimuth_offset_min": -0.6716816127300262,
        "azimuth_offset_max": -0.6071704878704622,
        "range_offset_mean": -1.5213690557541912,
        "range_offset_std": 0.022480444274138648,
        "range_offset_min": -1.56365141891456,
        "range_offset_max": -1.4809904055978222,
    },
}


def rslc_meta_from_h5(h5_s3_path: str) -> metadata.NisarRSLCMetadata:
    osio_options = {"session": boto3.session.Session()}
    with open_netcdf_osio(h5_s3_path, **osio_options) as ds:
        meta = metadata.NisarRSLCMetadata.parse_metadata(ds)
    return meta


def dataset_exists_in_h5(h5_s3_path: str, dataset: str) -> bool:
    osio_options = {"session": boto3.session.Session()}
    with open_netcdf_osio(h5_s3_path, **osio_options) as ds:
        return dataset in ds.keys()


def read_dataset_from_h5(
    h5_s3_path: str, dataset: str, row: int, col: int, width: int, height: int
) -> np.ndarray:
    osio_options = {"session": boto3.session.Session()}
    with open_netcdf_osio(h5_s3_path, **osio_options) as ds:
        data = ds[dataset][row : row + height, col : col + width]
    return data


@pytest.mark.parametrize("rslc_sample_h5_path", RSLC_SAMPLE_PATHS)
def test_gcps(rslc_sample_h5_path: str):
    meta = rslc_meta_from_h5(rslc_sample_h5_path)
    assert isinstance(meta, metadata.NisarRSLCMetadata)
    assert meta.frequency_a is not None

    orbit = Orbit(sv=meta.state_vectors, degree=11)
    assert isinstance(orbit, Orbit)

    proj_model = NisarModel.from_metadata(meta, frequency="A", orbit=orbit)
    assert isinstance(proj_model, NisarModel)

    gcps_x, gcps_y, gcps_incidence_angle = (
        np.array(meta.gcps_x).flatten(),
        np.array(meta.gcps_y).flatten(),
        np.array(meta.gcps_incidence_angle).flatten(),
    )
    target_3d_shape = gcps_x.shape
    gcps_height, gcps_azimuth_time, gcps_slant_range = np.meshgrid(
        meta.gcps_height, meta.gcps_azimuth_time, meta.gcps_slant_range, indexing="ij"
    )
    gcps_height = gcps_height.flatten()
    gcps_azimuth_time = gcps_azimuth_time.flatten()
    gcps_slant_range = gcps_slant_range.flatten()

    assert gcps_x.shape == target_3d_shape
    assert gcps_y.shape == target_3d_shape
    assert gcps_incidence_angle.shape == target_3d_shape
    assert gcps_height.shape == target_3d_shape
    assert gcps_azimuth_time.shape == target_3d_shape
    assert gcps_slant_range.shape == target_3d_shape

    # Projection
    rows, cols = proj_model.to_row_col(gcps_azimuth_time, gcps_slant_range)
    rows_pred, cols_pred, incidence_angles_pred = proj_model.projection(
        gcps_x, gcps_y, gcps_height, as_azt_rng=False
    )
    incidence_angles_pred = np.rad2deg(incidence_angles_pred)
    np.testing.assert_allclose(
        rows_pred,
        rows,
        atol=1e-3,
        err_msg="More than 1e-3 pixel projection error in azimuth",
    )
    np.testing.assert_allclose(
        cols_pred,
        cols,
        atol=1e-4,
        err_msg="More than 1e-4 pixel projection error in range",
    )
    np.testing.assert_allclose(
        incidence_angles_pred,
        gcps_incidence_angle,
        atol=2e-1,
        err_msg="More than 2e-1 degree error in incidence angle",
    )

    # Localization
    proj_x, proj_y, proj_height = proj_model.localization(
        row=rows, col=cols, alt=gcps_height
    )
    rows_pred, cols_pred, incidence_angles_pred = proj_model.projection(
        proj_x, proj_y, proj_height
    )
    incidence_angles_pred = np.rad2deg(incidence_angles_pred)

    np.testing.assert_allclose(
        rows_pred,
        rows,
        atol=1e-6,
        err_msg="More than 1e-6 pixel localization error in azimuth",
    )
    np.testing.assert_allclose(
        cols_pred,
        cols,
        atol=1e-6,
        err_msg="More than 1e-6 pixel localization error in range",
    )
    np.testing.assert_allclose(
        incidence_angles_pred,
        gcps_incidence_angle,
        atol=2e-1,
        err_msg="More than 2e-1 degree error in incidence angle",
    )


@pytest.mark.parametrize("rslc_sample_h5_path", RSLC_SAMPLE_PATHS)
def test_corner_reflectors_rosamond(rslc_sample_h5_path: str):
    meta = rslc_meta_from_h5(rslc_sample_h5_path)
    assert isinstance(meta, metadata.NisarRSLCMetadata)
    assert meta.frequency_a is not None

    orbit = Orbit(sv=meta.state_vectors, degree=11)
    assert isinstance(orbit, Orbit)

    proj_model = NisarModel.from_metadata(meta, frequency="A", orbit=orbit)
    assert isinstance(proj_model, NisarModel)

    corner_reflectors = pd.read_csv("./tests/data/California_2020-02-29_crdat.csv")
    assert isinstance(corner_reflectors, pd.DataFrame)
    assert set(
        [
            '   "Corner ID"',
            "Height Above Ellipsoid (m)",
            "Azimuth (deg)",
            "Tilt / Elevation angle (deg)",
            "Side Length (m)",
            "Latitude (deg)",
            "Longitude (deg)",
        ]
    ).issubset(set(corner_reflectors.columns))
    corner_reflectors.rename(
        columns={
            '   "Corner ID"': "identifier",
            "Height Above Ellipsoid (m)": "height_above_ellips",
            "Azimuth (deg)": "azimuth",
            "Tilt / Elevation angle (deg)": "elevation_angle",
            "Side Length (m)": "side_length",
            "Latitude (deg)": "latitude",
            "Longitude (deg)": "longitude",
        },
        inplace=True,
    )

    sensor_facing_east = (
        meta.look_side == "right" and meta.orbit_direction == "ascending"
    ) or (meta.look_side == "left" and meta.orbit_direction == "descending")
    azimuth_condition = (
        corner_reflectors["azimuth"].astype(float) < 180
        if sensor_facing_east
        else corner_reflectors["azimuth"].astype(float) >= 180
    )

    # Exclude corner reflectors that were repositioned between data acquisition dates.
    # The NISAR sample data is from 2008, but the corner reflector survey (CSV) is from 2020-02-29.
    # Reflectors 1, 5, and 12 were moved during this 12-year gap, causing position mismatches.
    accurate_coordinates_condition = ~corner_reflectors["identifier"].isin([1, 5, 12])

    bright_corner_reflectors = corner_reflectors[
        azimuth_condition & accurate_coordinates_condition
    ]
    for _, corner_reflector in bright_corner_reflectors.iterrows():
        r, c, _ = proj_model.projection(
            x=corner_reflector.longitude,
            y=corner_reflector.latitude,
            alt=corner_reflector.height_above_ellips,
        )
        assert r >= 0 and r <= proj_model.h - 1 and c >= 0 and c <= proj_model.w - 1, (
            "CR outside image"
        )

        crop_size = 32
        zoom_factor = 8

        # do a crop around CR
        col = round(c) - crop_size // 2
        row = round(r) - crop_size // 2

        # take roi around the prediction
        roi = Roi(col, row, crop_size, crop_size).make_valid(
            (proj_model.h, proj_model.w)
        )
        assert roi != Roi(0, 0, 0, 0), "Roi outside image"
        col_pred, row_pred = c - col, r - row

        # compute intensity image
        dataset = "science/LSAR/RSLC/swaths/frequencyA/HH"
        assert dataset_exists_in_h5(rslc_sample_h5_path, dataset), (
            f"Dataset {dataset} not found in {rslc_sample_h5_path}"
        )
        hh = read_dataset_from_h5(
            rslc_sample_h5_path, dataset, roi.row, roi.col, roi.w, roi.h
        )
        amplitude = np.abs(hh)
        amplitude_zoomed = fourier_zoom(amplitude, z=zoom_factor)
        search_roi = Roi.from_bounds_tuple(
            (
                floor(col_pred - 3),
                floor(row_pred - 3),
                ceil(col_pred + 3),
                ceil(row_pred + 3),
            )
        )

        subpix_max_measured, _ = sub_pixel_maxima(
            amplitude_zoomed, search_roi, zoom_factor=zoom_factor
        )

        assert len(subpix_max_measured), "No local max found in search region"

        # then just take the most significant maximum
        (row_measured, col_measured), _ = subpix_max_measured[0]
        assert row_measured is not None, (
            "Quadratic polynomial fitting failed around prediction"
        )

        # in pixels
        az_ale_pixels = float(row_measured - row_pred)
        rng_ale_pixels = float(col_measured - col_pred)

        sample_id = "RSLC_001" if "RSLC_001" in rslc_sample_h5_path else "RSLC_002"
        expected_stats = RSLC_SAMPLE_CORNER_REFLECTOR_STATS[sample_id]
        computed_stats = {
            "azimuth_offset_mean": np.mean(az_ale_pixels),
            "azimuth_offset_std": np.std(az_ale_pixels),
            "azimuth_offset_min": np.min(az_ale_pixels),
            "azimuth_offset_max": np.max(az_ale_pixels),
            "range_offset_mean": np.mean(rng_ale_pixels),
            "range_offset_std": np.std(rng_ale_pixels),
            "range_offset_min": np.min(rng_ale_pixels),
            "range_offset_max": np.max(rng_ale_pixels),
        }
        for stat_name, expected_value in expected_stats.items():
            computed_value = computed_stats[stat_name]
            assert abs(computed_value - expected_value) < 0.2, (
                f"Computed {stat_name}={computed_value} differs from expected {expected_value} by more than 0.2"
            )

        # in meters
        az_ale_meters = az_ale_pixels * meta.frequency_a.azimuth_spacing
        rng_ale_meters = rng_ale_pixels * meta.frequency_a.slant_range_spacing

        assert abs(az_ale_meters) < 3.5, "More than 3.5 meters error in azimuth"
        assert abs(rng_ale_meters) < 8, "More than 8 meters error in range"
