"""Projection comparison with s1m (master branch). For this test, you need to\
have s1m installed on your system."""


import numpy as np
import s1m
from eos.products import sentinel1
from eos.sar.orbit import Orbit


def test_projection_vs_s1m():
    xml_path =\
        './tests/data/s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml'
    with open(xml_path) as f:
        xml_content = f.read()
    burst_meta = sentinel1.metadata.extract_burst_metadata(xml_content, burst_id=1)
    # create an orbit
    orbit = Orbit(burst_meta["state_vectors"])
    # create a doppler
    doppler = sentinel1.doppler_info.doppler_from_meta(burst_meta, orbit)
    # create a corrector
    corrector = sentinel1.coordinate_correction.s1_corrector_from_meta(
        burst_meta, orbit, doppler, apd=True, bistatic=True)
    # create a Sentinel1BurstModel
    bmod = sentinel1.proj_model.burst_model_from_burst_meta(
        burst_meta, orbit, corrector
    )

    # create a grid of points
    cols_grid, rows_grid = np.meshgrid(np.linspace(0, bmod.w - 1, 10), np.linspace(0, bmod.h - 1, 10))
    cols, rows = cols_grid.ravel(), rows_grid.ravel()
    alts = np.zeros_like(cols)

    # localize the points
    lon, lat, alt = bmod.localization(rows, cols, alts)

    # check if localized points are at alt = 0
    np.testing.assert_allclose(alts, alt, atol=1e-5)

    # now project these points back in the burst
    rows_pred, cols_pred, i_pred = bmod.projection(lon, lat, alt)

    # verify projection vs s1m projection
    s1model = s1m.Sentinel1Model(xml_path)
    s1_cols_pred, s1_row_pred, s1_i_pred = s1m.main_projection(
        s1model, lon, lat,
        alt, error_when_outside=False,
        deburst=True,
        flip=False,
        apd_correction=True,
        bistatic_correction=True,
        verbose=False)

    # check similarity of x coordinate referenced to first col in raster
    # atol is set to a big value when testing against s1m master branch
    # because the orbit interpolation and projection is done differently
    np.testing.assert_allclose(s1_cols_pred + s1model.x_min,
                               cols_pred + burst_meta['burst_roi'][0], atol=1e-2)

    # check similarity of azimuth time
    azt_pred, _ = bmod.to_azt_rng(rows_pred, cols_pred)
    np.testing.assert_allclose(
        s1_row_pred / s1model.azimuth_frequency + s1model.burst_times[0][1],
        azt_pred)
