import numpy as np
import pyproj
from eos.products import sentinel1
from eos.sar import range_doppler


def test_projection():
    xml_path =\
        './tests/data/s1b-iw3-slc-vv-20190803t164007-20190803t164032-017424-020c57-006.xml'
    with open(xml_path) as f:
        xml_content = f.read()
    burst_meta = sentinel1.metadata.extract_burst_metadata(xml_content, burst_id=1)
    # create a Sentinel1BurstModel
    bmod = sentinel1.proj_model.burst_model_from_burst_meta(
        burst_meta,
        bistatic_correction=True,
        apd_correction=True,
        intra_pulse_correction=True,
        alt_fm_mismatch_correction=True)
    # create a grid of points
    x, y, w, h = bmod.burst_roi.to_roi()
    cols_grid, rows_grid = np.meshgrid(np.linspace(0, w - 1, 10), np.linspace(0, h - 1, 10))
    cols, rows = cols_grid.ravel(), rows_grid.ravel()
    alts = np.zeros_like(cols)

    # localize the points
    lon, lat, alt = bmod.localization(rows, cols, alts)

    # check if localized points are at alt = 0
    np.testing.assert_allclose(alts, alt, atol=1e-5)

    # now project these points back in the burst
    rows_pred, cols_pred, i_pred = bmod.projection(lon, lat, alt)

    # check if point fall back in the same location
    np.testing.assert_allclose(cols_pred, cols, atol=1e-3)
    np.testing.assert_allclose(rows_pred, rows, atol=1e-2)

    # check ability to query one point
    ptlon, ptlat, ptalt = bmod.localization(rows[0], cols[0], alts[0])
    assert isinstance(
        ptlon, float), "vectorized localization func failed on scalar input"

    # check ability to query one point
    ptrow, ptcol, pti = bmod.projection(lon[0], lat[0], alt[0])
    assert isinstance(
        ptrow, float), "vectorized projection func failed on scalar input"

    # check iterative_projection
    transform = pyproj.Transformer.from_crs(
        'epsg:4326', 'epsg:4978', always_xy=True)
    gx, gy, gz = transform.transform(lon, lat, alt)
    azt, rng, i = range_doppler.iterative_projection(bmod.orbit, gx, gy, gz)
    assert isinstance(
        azt, np.ndarray), "vectorized iterative projection func failed on array input"

    gx, gy, gz = range_doppler.iterative_localization(bmod.orbit, azt, rng,
                                                      np.zeros_like(alt),
                                                      (gx + 10, gy + 2, gz + 3))
    assert isinstance(gx, np.ndarray), \
        "vectorized iterative localization func failed on array input"

    azt, rng, i = range_doppler.iterative_projection(bmod.orbit,
                                                     gx[0], gy[0], gz[0])
    assert isinstance(azt, float),\
        "vectorized iterative projection func failed on scalar input"

    init_gxyz = (gx[0] + 10, gy[0] + 2, gz[0] + 3)

    gx, gy, gz = range_doppler.iterative_localization(bmod.orbit, azt, rng, 0,
                                                      init_gxyz)
    assert isinstance(
        gx, float), "vectorized iterative localization func failed on scalar input"
