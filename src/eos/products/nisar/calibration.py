from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from eos.products.nisar.metadata import Frequency, NisarRSLCMetadata, Polarization
from eos.sar.calibration import apply_radiometric_calibration, bilinear_interpolation
from eos.sar.io import ImageReader, Window
from eos.sar.roi import Roi


def _time_range_to_line_col(
    azimuth_time,
    slant_range,
    azimuth_time_first: float,
    azimuth_time_interval: float,
    slant_range_first: float,
    slant_range_spacing: float,
):
    """
    Convert LUT sample coordinates (azimuth time, slant range) to raster (line, col)
    coordinates. They are fractional in general, as LUT grids are not aligned with
    the raster grid.
    """
    lines = (np.asarray(azimuth_time) - azimuth_time_first) / azimuth_time_interval
    pixels = (np.asarray(slant_range) - slant_range_first) / slant_range_spacing

    assert np.all(np.diff(lines) > 0), "LUT azimuth times should be increasing"
    assert np.all(np.diff(pixels) > 0), "LUT slant ranges should be increasing"

    return lines.astype(np.float64), pixels.astype(np.float64)


class NisarRSLCCalibrator:
    """
    Radiometric calibration for NISAR RSLC products, with optional
    noise-equivalent backscatter correction.

    Example
        >>> calibrator = NisarRSLCCalibrator(metadata, frequency="A", polarization="HH")
        >>> calibrator.calibrate_inplace(myarray, roi, "gamma")

    Note
        For more details, see the NISAR product description
        (https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/DOCS/NISAR_D-102268_RevE_NASA_SDS_Product_Specification_L1_RSLC_clean_w-sigs.pdf)
        on noise-equivlaent backscatter: it is provided in the same units as the image
        power, so `x0_ns = max(0, |RSLC|^2 - noiseEquivalentBackscatter) / x0_LUT^2`
        (x0 = beta0/gamma0/sigma0).
    """

    def __init__(
        self,
        metadata: NisarRSLCMetadata,
        frequency: Frequency,
        polarization: Polarization,
    ):
        freq_metadata = (
            metadata.frequency_a if frequency == "A" else metadata.frequency_b
        )
        assert freq_metadata is not None, f"Frequency {frequency} not available"
        assert polarization in freq_metadata.ne_backscatter, (
            f"Polarization {polarization} not available for frequency {frequency}"
        )

        self._load_calibration(metadata, freq_metadata)
        self._load_noise(metadata, freq_metadata, polarization)

    def calibrate_inplace(
        self,
        image,
        roi,
        method,
        as_amplitude: bool = False,
    ):
        assert method in ("sigma", "gamma", "beta")
        assert image.shape == roi.get_shape()

        window = roi.to_roi()
        calib_array = self._get_calibration_array(window, method)
        noise_array = self._get_noise_array(window)

        return apply_radiometric_calibration(
            image,
            calib_array,
            noise_array,
            dont_clip_noise=False,
            as_amplitude=as_amplitude,
        )

    def _load_calibration(self, metadata: NisarRSLCMetadata, freq_metadata):
        lines, pixels = _time_range_to_line_col(
            metadata.lut_azimuth_time,
            metadata.lut_slant_range,
            metadata.azimuth_time_first,
            metadata.azimuth_time_interval,
            freq_metadata.slant_range_first,
            freq_metadata.slant_range_spacing,
        )
        values = {
            method: np.array(getattr(metadata, f"lut_{method}0"))
            for method in ["beta", "sigma", "gamma"]
        }

        self._lines, self._pixels = lines, pixels
        self._values = values

        assert len(self._lines) == len(self._values["gamma"])
        assert (
            len(self._lines) * len(self._pixels)
            == np.asarray(self._values["gamma"]).size
        )

    def _load_noise(self, metadata: NisarRSLCMetadata, freq_metadata, polarization):
        lines, pixels = _time_range_to_line_col(
            freq_metadata.ne_backscatter_azimuth_time,
            freq_metadata.ne_backscatter_slant_range,
            metadata.azimuth_time_first,
            metadata.azimuth_time_interval,
            freq_metadata.slant_range_first,
            freq_metadata.slant_range_spacing,
        )

        self._noise_lines = lines
        self._noise_pixels = pixels
        self._noise_values = np.array(freq_metadata.ne_backscatter[polarization])

        assert len(self._noise_lines) == len(self._noise_values)
        assert (
            len(self._noise_lines) * len(self._noise_pixels)
            == np.asarray(self._noise_values).size
        )

    def _get_calibration_array(self, window, method):
        return bilinear_interpolation(
            window, self._lines, self._pixels, np.asarray(self._values[method])
        )

    def _get_noise_array(self, window):
        return bilinear_interpolation(
            window, self._noise_lines, self._noise_pixels, self._noise_values
        )


@dataclass(frozen=True)
class CalibrationReader(ImageReader):
    """Class to calibrate after reading the data"""

    reader: ImageReader
    """Any ImageReader object (has .read(index, window)). Reader to the raster of the product."""
    calibrator: NisarRSLCCalibrator
    """Calibrator on the same product (same frequency/polarization)."""
    method: str
    """Calibration method (either "sigma", "gamma", "beta")."""
    tile_size: Optional[int] = None
    """If not None, the calibration is done by tile, reducing the memory cost for large arrays."""
    as_amplitude: bool = True
    """By default, returns the raster in amplitude unit (same as the underlying raster).
    If False, the raster is returned in intensity unit."""

    def read(
        self,
        indexes: Optional[Union[int, Sequence[int]]],
        window: Window,
        **kwargs: Any,
    ) -> NDArray[Any]:
        """
        Read and calibrate the data.

        Parameters
        ----------
        indexes : int or list of int
            Band index.
        window : tuple
            ((row, row+h), (col, col+w)).

        Returns
        -------
        ndarray
            Array read and calibrated.

        """
        array = self.reader.read(indexes, window=window, **kwargs)

        (y, yh), (x, xw) = window
        h = yh - y
        w = xw - x
        roi = Roi(x, y, w, h)

        if self.tile_size is not None:
            ox, oy = roi.get_origin()
            for tile_roi in roi.split_into_tiles(self.tile_size, self.tile_size):
                roi_in_array = tile_roi.translate_roi(-ox, -oy)
                tile = roi_in_array.crop_array(array)
                # because the calibration operates on contiguous arrays, we copy the views
                tile[:] = self.calibrator.calibrate_inplace(
                    tile.copy(),
                    tile_roi,
                    self.method,
                    as_amplitude=self.as_amplitude,
                )
        else:
            self.calibrator.calibrate_inplace(
                array,
                roi,
                self.method,
                as_amplitude=self.as_amplitude,
            )

        return array
