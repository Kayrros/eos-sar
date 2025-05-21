import numpy as np
import pytest

from eos.products import sentinel1
from eos.sar import io
from eos.sar.roi import Roi

# TODO: from a old and a new product, pre and post IPF 2.9.0

windows = (
    (1000, 100, 100, 50),  # top left
    (18100, 12270, 50, 100),  # middle center
    (10000, 6000, 50, 100),  # bottom right
)


def get_infos(swath, pol, method, with_noise, s3_client):
    pid = "S1B_IW_SLC__1SDV_20190702T032447_20190702T032514_016949_01FE47_69C5"
    basepath = (
        "s3://kayrros-dev-satellite-test-data/sentinel-1/eos_test_data/test_calibration"
    )
    calibration = io.read_xml_file(
        f"{basepath}/{pid}.SAFE/{swath}-{pol}-calibration.xml", s3_client
    )
    noise = None
    if with_noise:
        noise = io.read_xml_file(
            f"{basepath}/{pid}.SAFE/{swath}-{pol}-noise.xml", s3_client
        )
    calibrator = sentinel1.calibration.Sentinel1Calibrator(calibration, noise)

    image_path = f"{basepath}/{pid}.SAFE/{swath}-{pol}.tiff"
    reader = io.open_image(image_path)

    snapmethod = method.capitalize() + "0"
    if with_noise:
        image_path = f"{basepath}/{pid}_T_Cal.data/{snapmethod}_{swath.upper()}_{pol.upper()}.img"
    else:
        image_path = (
            f"{basepath}/{pid}_Cal.data/{snapmethod}_{swath.upper()}_{pol.upper()}.img"
        )
    snap_reader = io.open_image(image_path)

    return calibrator, reader, snap_reader


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
@pytest.mark.parametrize("swath", ("iw1", "iw2", "iw3"))
@pytest.mark.parametrize("pol", ("vv", "vh"))
@pytest.mark.parametrize("window", windows)
def test_calibration_without_noise(window, pol, swath, method, with_noise, s3_client):
    _, _, w, h = window
    roi = Roi.from_roi_tuple(window)

    calibrator, reader, snap_reader = get_infos(
        swath, pol, method, with_noise, s3_client
    )

    image = io.read_window(reader, roi, get_complex=False)
    assert image.shape == (h, w)
    assert image.dtype == np.float32

    imagec = io.read_window(reader, roi)
    assert imagec.shape == (h, w)
    assert imagec.dtype == np.complex64

    for clip in (True, False) if with_noise else (False,):
        print(clip)

        arr = calibrator.calibrate_inplace(
            image.copy(), roi, method=method, dont_clip_noise=not clip
        )
        arrc = calibrator.calibrate_inplace(
            imagec.copy(), roi, method=method, dont_clip_noise=not clip
        )

        snap = None
        if not with_noise or not clip:
            # only check against snap if we ask for noise and clip or if we don't ask for noise correction
            snap = io.read_window(snap_reader, roi, get_complex=False)

        compare_arrays(arr, arrc, imagec, snap)


def test_inplaceness(s3_client):
    calibrator, _, _ = get_infos("iw1", "vv", "sigma", False, s3_client)
    arr = np.ones((20, 20), dtype=np.float32)
    roi = Roi(1000, 100, 20, 20)
    arr2 = calibrator.calibrate_inplace(arr, roi, "sigma")
    assert arr2 is arr
