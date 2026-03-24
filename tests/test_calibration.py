from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from rasterio.io import DatasetReader

from eos.products import sentinel1
from eos.products.sentinel1.catalog import CDSESentinel1SLCCatalogBackend
from eos.products.sentinel1.product import CDSEUnzippedSafeSentinel1SLCProductInfo
from eos.sar import io
from eos.sar.roi import Roi

# TODO: from a old and a new product, pre and post IPF 2.9.0

windows = {
    "top_left": Roi(1000, 100, 100, 50),  # top left
    "middle_center": Roi(18100, 12270, 50, 100),  # middle center
    "bottom_right": Roi(10000, 6000, 50, 100),  # bottom right
}

SWATHS = ["iw1", "iw2", "iw3"]
POLARIZATIONS = ["vv", "vh"]


@pytest.fixture(scope="module")
def img_infos(cdse_s3_session):
    """
    Read all calibration and noise xmls. Also read all numpy arrays (for all windows).
    """
    pid = "S1B_IW_SLC__1SDV_20190702T032447_20190702T032514_016949_01FE47_69C5"

    catalog_backend = CDSESentinel1SLCCatalogBackend()

    product = CDSEUnzippedSafeSentinel1SLCProductInfo.from_product_id(
        catalog_backend, cdse_s3_session, pid
    )
    calibration_per_swath_per_pol: dict[str, dict[str, str]] = dict()
    noise_per_swath_per_pol: dict[str, dict[str, str]] = dict()
    arrays_per_swath_per_pol_per_window: dict[
        str, dict[str, dict[str, NDArray[np.complex64]]]
    ] = dict()

    for swath in SWATHS:
        calibration_per_swath_per_pol[swath] = dict()
        noise_per_swath_per_pol[swath] = dict()
        arrays_per_swath_per_pol_per_window[swath] = dict()
        for pol in POLARIZATIONS:
            calibration_per_swath_per_pol[swath][pol] = product.get_xml_calibration(
                swath, pol
            )
            noise_per_swath_per_pol[swath][pol] = product.get_xml_noise(swath, pol)
            arrays_per_swath_per_pol_per_window[swath][pol] = dict()
            reader = product.get_image_reader(swath, pol)
            for win_txt, win_roi in windows.items():
                arr = io.read_window(reader, win_roi, get_complex=True)
                assert arr.dtype == np.complex64
                arrays_per_swath_per_pol_per_window[swath][pol][win_txt] = cast(
                    NDArray[np.complex64], arr
                )
            assert isinstance(reader, DatasetReader)
            reader.close()
    return (
        calibration_per_swath_per_pol,
        noise_per_swath_per_pol,
        arrays_per_swath_per_pol_per_window,
    )


def get_infos(swath, pol, method, with_noise):
    pid = "S1B_IW_SLC__1SDV_20190702T032447_20190702T032514_016949_01FE47_69C5"

    # # TODO replace DUMMY_URL with an actual path where the SNAP reference data will reside
    basepath = "DUMMY_URL/eos_test_data/test_calibration"
    snapmethod = method.capitalize() + "0"
    if with_noise:
        image_path = f"{basepath}/{pid}_T_Cal.data/{snapmethod}_{swath.upper()}_{pol.upper()}.img"
    else:
        image_path = (
            f"{basepath}/{pid}_Cal.data/{snapmethod}_{swath.upper()}_{pol.upper()}.img"
        )
    snap_reader = io.open_image(image_path)

    return snap_reader


def compare_arrays(calibrated_abs, calibrated_complex, uncalibrated_complex, snap=None):
    # make sure that compute the magnitude of the calibrated complex gives the same as the calibration of the magnitude
    assert np.allclose(np.abs(calibrated_complex), calibrated_abs)

    # and that the phase didn't change
    # since we can have zeros due to the thermal noise correction, make sure we compare angles where it makes sense
    m = np.abs(calibrated_complex) != 0
    assert np.allclose(
        np.angle(calibrated_complex)[m], np.angle(uncalibrated_complex)[m]
    )

    if snap is not None:
        # quite large absolute tolerance, it seems that some pixels can be quite different (why?)
        assert np.allclose(calibrated_abs, snap, atol=1e-2, rtol=1e-2)
        # but at least impose that the median is very similar
        assert np.allclose(
            np.median(calibrated_abs), np.median(snap), atol=1e-5, rtol=1e-4
        )


@pytest.mark.parametrize("with_noise", (False, True))
@pytest.mark.parametrize("method", ("gamma", "beta", "sigma"))
@pytest.mark.parametrize("swath", SWATHS)
@pytest.mark.parametrize("pol", POLARIZATIONS)
@pytest.mark.parametrize("window_txt", list(windows.keys()))
def test_calibration_without_noise(
    window_txt, pol, swath, method, with_noise, img_infos
):
    (
        calibration_per_swath_per_pol,
        noise_per_swath_per_pol,
        arrays_per_swath_per_pol_per_window,
    ) = img_infos
    calibration = calibration_per_swath_per_pol[swath][pol]
    noise = noise_per_swath_per_pol[swath][pol] if with_noise else None
    calibrator = sentinel1.calibration.Sentinel1Calibrator(calibration, noise)
    roi = windows[window_txt]
    h = roi.h
    w = roi.w

    # snap_reader = get_infos( swath, pol, method, with_noise    )

    imagec = arrays_per_swath_per_pol_per_window[swath][pol][window_txt]
    assert imagec.shape == (h, w)
    assert imagec.dtype == np.complex64

    image = np.abs(imagec)
    assert image.shape == (h, w)
    assert image.dtype == np.float32

    for clip in (True, False) if with_noise else (False,):
        arr = calibrator.calibrate_inplace(
            image.copy(), roi, method=method, dont_clip_noise=not clip
        )
        assert arr.dtype == image.dtype
        assert arr.shape == image.shape

        arrc = calibrator.calibrate_inplace(
            imagec.copy(), roi, method=method, dont_clip_noise=not clip
        )
        assert arrc.dtype == imagec.dtype
        assert arrc.shape == imagec.shape

        # snap = None
        # if not with_noise or not clip:
        # only check against snap if we ask for noise and clip or if we don't ask for noise correction
        # snap = io.read_window(snap_reader, roi, get_complex=False)

        # compare_arrays(arr, arrc, imagec, snap)


def test_inplaceness(img_infos):
    (
        calibration_per_swath_per_pol,
        _,
        _,
    ) = img_infos

    swath = "iw1"
    pol = "vv"

    calibration = calibration_per_swath_per_pol[swath][pol]
    noise = None
    calibrator = sentinel1.calibration.Sentinel1Calibrator(calibration, noise)

    arr = np.ones((20, 20), dtype=np.float32)
    roi = Roi(1000, 100, 20, 20)
    arr2 = calibrator.calibrate_inplace(arr, roi, "sigma")
    assert arr2 is arr
