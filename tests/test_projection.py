import numpy as np
import pyproj
import os

from eos.products import sentinel1
import eos.sar
from eos.sar import range_doppler
from eos.sar.orbit import Orbit


def test_projection():
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
        burst_meta, orbit, doppler, apd=True, bistatic=True, intra_pulse=True,
        alt_fm_mismatch=True)

    # create a Sentinel1BurstModel
    bmod = sentinel1.proj_model.burst_model_from_burst_meta(
        burst_meta, orbit, corrector)
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

    # check if point fall back in the same location
    np.testing.assert_allclose(cols_pred, cols, atol=1e-3)
    np.testing.assert_allclose(rows_pred, rows, atol=1e-3)

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


def test_projection_grd():
    xml_path = './tests/data/S1A_IW_GRDH_1SDV_20220609T022354_20220609T022419_043580_053410_DF62-vv-annotation.xml'
    with open(xml_path) as f:
        xml_content = f.read()
    meta = sentinel1.metadata.extract_grd_metadata(xml_content)

    # create an orbit
    orbit = Orbit(meta["state_vectors"])
    # create a corrector
    corrector = eos.sar.projection_correction.Corrector()

    # create a proj model
    proj_model = sentinel1.proj_model.grd_model_from_meta(meta, orbit, corrector)

    # create a grid of points
    cols_grid, rows_grid = np.meshgrid(np.linspace(-1000, proj_model.w + 1000, 100),
                                       np.linspace(-1000, proj_model.h + 1000, 100))
    cols, rows = cols_grid.ravel(), rows_grid.ravel()
    alts = np.zeros_like(cols)

    # localize the points
    lon, lat, alt = proj_model.localization(rows, cols, alts)

    # check if localized points are at alt = 0
    np.testing.assert_allclose(alts, alt, atol=1e-5)

    # now project these points back in the burst
    rows_pred, cols_pred, _ = proj_model.projection(lon, lat, alt)

    # check if point fall back in the same location
    np.testing.assert_allclose(cols_pred, cols, rtol=1e-3)
    np.testing.assert_allclose(rows_pred, rows, rtol=1e-3)

    # check ability to query one point
    ptlon, ptlat, ptalt = proj_model.localization(rows[0], cols[0], alts[0])
    assert isinstance(ptlon, float), "vectorized localization func failed on scalar input"

    # check ability to query one point
    ptrow, ptcol, pti = proj_model.projection(lon[0], lat[0], alt[0])
    assert isinstance(ptrow, float), "vectorized projection func failed on scalar input"


def test_projection_corner_reflectors():

    from math import floor, ceil
    from eos.sar import fourier_zoom, max_finding

    # subset of corner reflectors that give a mosaic of reasonable size
    # to understand why we picked those,
    # just do plt.scatter(lons, lats)
    cr_ids = [0, 1, 2, 7, 8,
              10, 11, 14, 15, 18, 19, 24,
              29, 30, 31, 32
              ]
    # coordinates of corner reflectors
    coords = np.loadtxt(
        "./tests/data/QLD_corner_reflector_positions_GDA2020.txt",
        skiprows=1, usecols=range(1, 7))[cr_ids]

    lats, lons, alts, gx, gy, gz = coords.T

    # products near 2020, 1, 1 in which we have the coordinates, otherwise the
    # velocity of deformation in the file needs to be used to get coordinates at date
    s3_bucket = 's3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/corner_reflectors_australia'
    safes = ['S1A_IW_SLC__1SSH_20200103T083235_20200103T083305_030633_03829E_71AD.SAFE',
             'S1A_IW_SLC__1SSH_20200103T083303_20200103T083331_030633_03829E_9342.SAFE']

    products = [sentinel1.product.SafeSentinel1ProductInfo(os.path.join(s3_bucket, safe)) for safe in safes]

    pol = "HH"
    swaths = ('iw1', 'iw2')
    calibration = "sigma"
    crop_size = 32
    zoom_factor = 32

    def fetch_orbits(pid, bursts):
        import phoenix.catalog
        phx_client = phoenix.catalog.Client()
        sentinel1.orbits.update_statevectors_using_phoenix(phx_client, pid, bursts,
                                                           force_type="orbpoe")
        return bursts

    asm = sentinel1.assembler.Sentinel1Assembler.from_products(
        products, pol, orbit_provider=fetch_orbits, swaths=swaths)

    # do a model for a mosaic that contains all CRS
    # just for the estimation of resampling matrices
    # then do a small crop (mosaic) per CR
    mosaic_model = asm.get_mosaic_model()

    rows, cols, _ = mosaic_model.projection(gx, gy, gz, crs='epsg:4978')

    roi_all = eos.sar.roi.Roi.from_bounds_tuple(eos.sar.roi.Roi.points_to_bbox(rows, cols))
    roi_all.add_margin(2 * crop_size, inplace=True).make_valid((mosaic_model.h, mosaic_model.w), inplace=True)

    # get affected bsids
    primary_cutter = asm.get_primary_cutter()
    all_bsids, _, _ = primary_cutter.get_debursting_rois(roi_all)

    dem = eos.dem.get_any_source()
    # get registration dem pts
    x, y, alt, crs = eos.sar.regist.get_registration_dem_pts(
        mosaic_model, roi=roi_all, dem=dem, sampling_ratio=1)

    # project in the mosaic
    azt_primary_flat, rng_primary_flat, _ = mosaic_model.projection(x, y, alt, crs=crs, as_azt_rng=True)

    pts_in_burst_mask = {}
    azt_primary = {}
    rng_primary = {}

    for bsid in all_bsids:
        burst_mask = primary_cutter.mask_pts_in_burst(bsid, azt_primary_flat, rng_primary_flat)
        pts_in_burst_mask[bsid] = burst_mask
        azt_primary[bsid] = azt_primary_flat[burst_mask]
        rng_primary[bsid] = rng_primary_flat[burst_mask]

    corrections = dict(
        bistatic=True,
        full_bistatic=True,
        apd=True,
        intra_pulse=True,
        alt_fm_mismatch=True
    )

    corrector_per_bsid = asm.get_corrector_per_bsid(all_bsids, **corrections)

    # here, the points are projected again with the corrections and matrices are fit
    # in theory, for the primary image, we could have avoided re-projection, but for
    # keeping the code simple, we use this function here
    burst_resampling_matrices = sentinel1.regist.secondary_registration_estimation(
        mosaic_model, primary_cutter, corrector_per_bsid, x, y, alt, crs,
        all_bsids, pts_in_burst_mask, primary_cutter, azt_primary, rng_primary)

    readers = asm.get_image_readers(products, all_bsids, pol, calibration)

    cols_pred = np.zeros_like(cols)
    rows_pred = np.zeros_like(cols)

    cols_meas = np.zeros_like(cols)
    rows_meas = np.zeros_like(cols)

    for idx, (r, c) in enumerate(zip(rows, cols)):

        col = round(c) - crop_size // 2
        row = round(r) - crop_size // 2

        # take roi around the prediction
        roi = eos.sar.roi.Roi(col, row, crop_size, crop_size)

        # prediction of coords with respect to new origin
        # notice how the prediction is the one without corrections
        # because the arrays have will be read and resampled according to corrections
        col_pred, row_pred = c - col, r - row
        # store result
        cols_pred[idx] = col_pred
        rows_pred[idx] = row_pred

        out_shape = roi.get_shape()
        out = np.full(out_shape, np.nan, dtype=np.csingle)

        # recompute relevant roi info
        bsids, within_burst_rois, write_rois = primary_cutter.get_debursting_rois(roi)

        # instantiate resamplers
        resamplers = {
            bsid: asm.get_burst_resampler(
                bsid,
                primary_cutter.get_burst_outer_roi_in_tiff(bsid).get_shape(),
                burst_resampling_matrices[bsid]) for bsid in bsids
        }

        # read, resample, but do not reramp
        sentinel1.deburst.warp_rois_read_resample_deburst(
            bsids, resamplers, within_burst_rois, primary_cutter,
            readers, write_rois, out_shape, out,
            get_complex=True, reramp=False)

        # zoom
        zoomed = fourier_zoom.fourier_zoom(out, z=zoom_factor)
        amp_zoomed = np.abs(zoomed)

        # Now find the max of the amplitude and compare with prediction
        search_roi = eos.sar.roi.Roi.from_bounds_tuple(
            (floor(col_pred - 1), floor(row_pred - 1), ceil(col_pred + 1), ceil(row_pred + 1))
        )

        subpix_max_measured, _ = max_finding.sub_pixel_maxima(amp_zoomed, search_roi,
                                                              zoom_factor=zoom_factor)

        assert len(subpix_max_measured), "No local max found in search region"

        # then just take the most significant maximum
        (row_measured, col_measured), _ = subpix_max_measured[0]

        assert row_measured is not None, "Quadratic polynomial fitting failed around prediction"

        rows_meas[idx] = row_measured
        cols_meas[idx] = col_measured

    np.testing.assert_allclose(cols_pred, cols_meas, rtol=0, atol=.5)  # atol high bcz there is a bias in range
    # apd being imperfect, and other corrections (ionosphere, tides...)

    np.testing.assert_allclose(rows_pred, rows_meas, rtol=0, atol=1e-1)
    # also for atol here, the azimuth bias (mainly tides)

    # note: the bias is actually not as high as the tol that was set

    # however, the standard deviation of the error is low
    assert (cols_pred - cols_meas).std() < 0.03, "Col standard deviation higher than expected"
    assert (rows_pred - rows_meas).std() < 0.03, "Row standard deviation higher than expected"
