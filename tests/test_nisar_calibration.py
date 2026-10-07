from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union, cast

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray

from eos.products.nisar.calibration import CalibrationReader, NisarRSLCCalibrator
from eos.products.nisar.metadata import Frequency, NisarRSLCMetadata, Polarization
from eos.sar import io
from eos.sar.io import RemoteH5Loader, Window
from eos.sar.roi import Roi

RSLC_SAMPLE_PATH = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/RSLC/NISAR_L1_PR_RSLC_002_030_A_019_2800_SHNA_A_20081127T061000_20081127T061014_D00404_N_F_J_001/NISAR_L1_PR_RSLC_002_030_A_019_2800_SHNA_A_20081127T061000_20081127T061014_D00404_N_F_J_001.h5"
FREQUENCY: Frequency = "A"
POLARIZATION: Polarization = "HH"
DATASET = f"science/LSAR/RSLC/swaths/frequency{FREQUENCY}/{POLARIZATION}"

# the sample product is 21559 lines x 6174 columns, with non-trivial
# sigma0/gamma0 LUTs and noise-equivalent backscatter (beta0 LUT is 1)
windows = {
    "top_left": Roi(100, 100, 100, 50),
    "middle_center": Roi(3000, 9800, 50, 100),
    "bottom_right": Roi(6000, 21400, 100, 50),
}

METHODS = ("gamma", "beta", "sigma")


def compare_arrays(calibrated_abs, calibrated_complex, uncalibrated_complex):
    # make sure that compute the magnitude of the calibrated complex gives the same as the calibration of the magnitude
    assert np.allclose(np.abs(calibrated_complex), calibrated_abs, atol=1e-6)
    # and that the phase didn't change
    # we make sure comparing angles makes sense
    m = np.abs(calibrated_complex) != 0
    assert np.allclose(
        np.angle(calibrated_complex)[m], np.angle(uncalibrated_complex)[m]
    )


@dataclass(frozen=True)
class _H5DatasetReader:
    """Minimal ImageReader over a 2D h5py dataset."""

    dataset: h5py.Dataset

    def read(
        self,
        indexes: Optional[Union[int, Sequence[int]]],
        window: Window,
        **kwargs: Any,
    ) -> NDArray[Any]:
        (row0, row1), (col0, col1) = window
        return self.dataset[row0:row1, col0:col1].astype(np.complex64)


@pytest.fixture(scope="module")
def h5_file():
    with RemoteH5Loader(RSLC_SAMPLE_PATH) as ds:
        yield ds


@pytest.fixture(scope="module")
def arrays_per_window(h5_file) -> dict[str, NDArray[np.complex64]]:
    return {
        win_txt: cast(
            NDArray[np.complex64],
            io.read_hdf5_window(
                h5_file[DATASET], win_roi, get_complex=True, boundless=False
            ),
        )
        for win_txt, win_roi in windows.items()
    }


@pytest.fixture(scope="module")
def calibrator(h5_file):
    meta = NisarRSLCMetadata.parse_metadata(h5_file)
    return NisarRSLCCalibrator(meta, frequency=FREQUENCY, polarization=POLARIZATION)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("window_txt", list(windows.keys()))
def test_calibration(window_txt, method, arrays_per_window, calibrator):
    roi = windows[window_txt]
    h = roi.h
    w = roi.w

    imagec = arrays_per_window[window_txt]
    assert imagec.shape == (h, w)
    assert imagec.dtype == np.complex64

    image = np.abs(imagec)
    assert image.shape == (h, w)
    assert image.dtype == np.float32

    for as_amplitude in (True, False):
        arr = calibrator.calibrate_inplace(
            image.copy(),
            roi,
            method=method,
            as_amplitude=as_amplitude,
        )
        assert arr.dtype == image.dtype
        assert arr.shape == image.shape
        assert np.isfinite(arr).all()
        assert (arr >= 0).all()

        arrc = calibrator.calibrate_inplace(
            imagec.copy(),
            roi,
            method=method,
            as_amplitude=as_amplitude,
        )
        assert arrc.dtype == imagec.dtype
        assert arrc.shape == imagec.shape

        compare_arrays(arr, arrc, imagec)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("window_txt", list(windows.keys()))
def test_calibration_matches_formula(window_txt, method, arrays_per_window, calibrator):
    roi = windows[window_txt]
    image = np.abs(arrays_per_window[window_txt])

    calib = calibrator._get_calibration_array(roi.to_roi(), method)
    assert calib.shape == image.shape
    assert (calib > 0).all()

    noise = calibrator._get_noise_array(roi.to_roi())
    assert noise.shape == image.shape
    # x0 = max(0, |RSLC|^2 - noiseEquivalentBackscatter) / x0_LUT^2
    expected = np.maximum(0, image.astype(np.float64) ** 2 - noise) / calib**2

    arr = calibrator.calibrate_inplace(
        image.copy(), roi, method=method, as_amplitude=False
    )
    np.testing.assert_allclose(arr, expected, rtol=1e-4, atol=1e-8)

    arr_amp = calibrator.calibrate_inplace(
        image.copy(), roi, method=method, as_amplitude=True
    )
    np.testing.assert_allclose(arr_amp, np.sqrt(expected), rtol=1e-4, atol=1e-6)


def test_noise_is_not_negative(calibrator):
    # negative noise values would add signal during the noise correction
    assert (calibrator._noise_values >= 0).all()
    for roi in windows.values():
        assert (calibrator._get_noise_array(roi.to_roi()) >= 0).all()


def test_noise_reduces_backscatter(arrays_per_window, calibrator):
    roi = windows["middle_center"]
    image = np.abs(arrays_per_window["middle_center"])

    calibrated = calibrator.calibrate_inplace(image.copy(), roi, "gamma")
    calib = calibrator._get_calibration_array(roi.to_roi(), "gamma")
    # x0 without noise correction = |RSLC|^2 / x0_LUT^2
    without_noise = image**2 / calib**2

    assert (calibrated <= without_noise * (1 + 1e-5)).all()
    assert (calibrated < without_noise).any()


def test_inplaceness(calibrator):
    arr = np.ones((20, 20), dtype=np.float32)
    roi = Roi(1000, 100, 20, 20)
    arr2 = calibrator.calibrate_inplace(arr, roi, "sigma")
    assert arr2 is arr


@pytest.mark.parametrize("as_amplitude", (True, False))
def test_calibration_reader(as_amplitude, h5_file, arrays_per_window, calibrator):
    roi = windows["middle_center"]
    col, row, w, h = roi.to_roi()
    window: Window = ((row, row + h), (col, col + w))
    reader = _H5DatasetReader(h5_file[DATASET])

    reader_out = CalibrationReader(
        reader, calibrator, method="gamma", as_amplitude=as_amplitude
    ).read(1, window)
    direct = calibrator.calibrate_inplace(
        arrays_per_window["middle_center"].copy(),
        roi,
        "gamma",
        as_amplitude=as_amplitude,
    )
    np.testing.assert_allclose(reader_out, direct)

    tiled = CalibrationReader(
        reader, calibrator, method="gamma", as_amplitude=as_amplitude, tile_size=32
    ).read(1, window)
    np.testing.assert_allclose(tiled, reader_out, rtol=1e-5)
