from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import xmltodict
from numpy.typing import NDArray

from eos.sar.io import ImageReader, Window
from eos.sar.roi import Roi

from . import _calibration as _cal  # type: ignore


class CalibrationError(Exception):
    pass


class _AzimuthNoise:
    """
    /!\\ This comes from the s1c package.

    Noise Azimuth Vector
    """

    def __init__(
        self,
        swath,
        first_azimuth_line,
        first_range_sample,
        last_azimuth_line,
        last_range_sample,
        lines,
        noise_azimuth_LUT,
    ):
        self.swath = swath
        self.first_azimuth_line = first_azimuth_line
        self.first_range_sample = first_range_sample
        self.last_azimuth_line = last_azimuth_line
        self.last_range_sample = last_range_sample
        self.lines = lines
        self.noise_azimuth_LUT = noise_azimuth_LUT


def _get_noise_azimuth_blocks(noiseAzimuthVectorList):
    """
    /!\\ This comes from the s1c package.

    Instanciate the noise azimuth blocks.

    Args:
        noiseAzimuthVectorList (OrderedDict): OrderedDict of noiseAzimuthVectorList.

    Returns:
        list: list of _AzimuthNoise objects.
    """
    blocks = []
    for v in noiseAzimuthVectorList:
        blocks.append(
            _AzimuthNoise(
                v["swath"],
                int(v["firstAzimuthLine"]),
                int(v["firstRangeSample"]),
                int(v["lastAzimuthLine"]),
                int(v["lastRangeSample"]),
                list(map(float, v["line"]["#text"].split())),
                list(map(float, v["noiseAzimuthLut"]["#text"].split())),
            )
        )
    return blocks


def _read_lut_from_noise_xml(xml):
    """
    /!\\ This comes from the s1c package.

    Read the noise Look Up Table from a S1 xml noise calibration file.

    Args:
        xml (str): content of a Sentinel-1 noise xml file (e.g.
            output of open("/path/to/noise.xml").read())

    Return:
        list: list of line numbers
        list: list of lists of column numbers, where each list of column
            numbers corresponds to one element of the list of line numbers
        list: list of lists of values matching the list of lists of column
            numbers
        list: list of _AzimuthNoise objects
    """
    d = xmltodict.parse(xml)["noise"]

    # extract lists of points and values
    lines = []
    pixels = []
    values = []

    # handle pre and post IPF 2.9.0 naming convention
    if "noiseRangeVectorList" in d:
        # /!\ this was modified from s1c
        noise_azimuth_vector_list = d["noiseAzimuthVectorList"]
        # for some reason, when @count is 1, the list is not considered as such
        if int(noise_azimuth_vector_list["@count"]) == 1:
            noise_azimuth_vector_list = [
                noise_azimuth_vector_list["noiseAzimuthVector"],
            ]
        else:
            noise_azimuth_vector_list = noise_azimuth_vector_list["noiseAzimuthVector"]
        azimuth_blocks = _get_noise_azimuth_blocks(noise_azimuth_vector_list)

        noise_vector_list = d["noiseRangeVectorList"]["noiseRangeVector"]
        lut_key = "noiseRangeLut"
    else:
        azimuth_blocks = None
        noise_vector_list = d["noiseVectorList"]["noiseVector"]
        lut_key = "noiseLut"

    for v in noise_vector_list:
        lines.append(int(v["line"]))
        pixels.append(list(map(int, v["pixel"]["#text"].split())))
        values.append(list(map(float, v[lut_key]["#text"].split())))

    # check lists lengths
    if (
        len(lines) != len(pixels)
        or len(pixels) != len(values)
        or any(len(p) != len(v) for p, v in zip(pixels, values))
    ):
        raise RuntimeError("Unexpected data format in noise xml")

    return lines, pixels, values, azimuth_blocks


def _read_lut_from_calibration_xml(xml):
    """
    /!\\ This comes from the s1c package.

    Read the sigma, beta, gamma and DN samples from a S1 xml calibration file.

    Args:
        xml (str): content of a Sentinel-1 calibration xml file (e.g.
            output of open("/path/to/calibration.xml").read())

    Return:
        list: list of line numbers
        list: list of lists of column numbers, where each list of column
            numbers corresponds to one element of the list of line numbers
        dict: dictionary with four keys, and a list of lists of values matching
            the list of lists of column numbers
    """
    d = xmltodict.parse(xml)["calibration"]

    # extract lists of points and values
    lines = []
    pixels = []
    values: dict[str, list[list[float]]] = {
        "sigmaNought": [],
        "betaNought": [],
        "gamma": [],
        "dn": [],
    }
    for v in d["calibrationVectorList"]["calibrationVector"]:
        lines.append(float(v["line"]))
        pixels.append(list(map(float, v["pixel"]["#text"].split())))
        for k in values:
            values[k].append(list(map(float, v[k]["#text"].split())))

    # check arrays shapes
    if len(lines) != len(pixels):
        raise CalibrationError("Unexpected data format in calibration xml")
    for k in values:
        if len(pixels) != len(values[k]) or any(
            len(p) != len(v) for p, v in zip(pixels, values[k])
        ):
            raise CalibrationError("Unexpected data format in calibration xml")

    return lines, pixels, values


def _bilinear_interpolation(window, lines, pixels, values):
    x, y, w, h = window
    if y + h > lines[-1]:
        values = np.pad(values, ((0, 1), (0, 0)), mode="edge")
        lines = np.append(lines, y + h)

    if x + w > pixels[-1]:
        values = np.pad(values, ((0, 0), (0, 1)), mode="edge")
        pixels = np.append(pixels, x + w)

    res = _cal.bilinear_interpolation(
        window,
        lines.astype(np.int32),
        pixels.astype(np.int32),
        values.astype(np.float32),
    )
    return res


def _apply_radiometric_calibration(
    img, calib_coeffs, noise_coeffs, dont_clip_noise, as_amplitude: bool
):
    if np.iscomplexobj(img):
        assert img.dtype == np.complex64
        assert calib_coeffs.dtype == np.float32
        assert noise_coeffs is None or noise_coeffs.dtype == np.float32
        _cal.apply_radiometric_calibration_complex64(
            img, calib_coeffs, noise_coeffs, dont_clip_noise, as_amplitude
        )
        return img
    else:
        assert img.dtype == np.float32
        assert calib_coeffs.dtype == np.float32
        assert noise_coeffs is None or noise_coeffs.dtype == np.float32
        _cal.apply_radiometric_calibration_float32(
            img, calib_coeffs, noise_coeffs, dont_clip_noise, as_amplitude
        )
        return img


class Sentinel1Calibrator:
    """
    Radiometric calibration for Sentinel1 SLC products.

    Example
        >>> calibrator_noise = calibration.Sentinel1Calibrator(cal_xml, noise_xml, ipf)
        >>> calibrator_noise.calibrate_inplace(myarray, window, "beta")

    Note
        For more details, see https://sentinels.copernicus.eu/documents/247904/0/Thermal-Denoising-of-Products-Generated-by-Sentinel-1-IPF/11d3bd86-5d6a-4e07-b8bb-912c1093bf91

        s1tbx (8.0.5) does not clip to 0 and instead keep the original value of the pixel.
        Expect some small differences between eos-sar and SNAP because of this. Any other difference should be reported!
    """

    _noise_azimuth_blocks: Optional[list[_AzimuthNoise]]

    def __init__(
        self,
        calibration_xml_content: str,
        noise_xml_content: Optional[str] = None,
        ipf: Optional[str] = None,
    ):
        self._load_calibration(calibration_xml_content)
        self._ipf = ipf
        if noise_xml_content:
            self._load_noise(noise_xml_content)
            self.has_noise = True
        else:
            self.has_noise = False

    def calibrate_inplace(
        self, image, roi, method, dont_clip_noise=False, as_amplitude: bool = False
    ):
        """
        Apply the radiometric calibration on the given raster, at position `window` of the SLC tif image.

        Args
            image (np.array): array of shape (h, w), of type float32 or complex64
            roi (Roi): position in the source SLC tif image
            method (str): 'sigma' | 'gamma' | 'beta'
            dont_clip_noise (bool, default False):
                if true, during noise calibration, values are not clipped to 0 but stay positive
                this is what happens in the implementation of SNAP
            as_amplitude (bool, default False):
                if true, convert back to "amplitude unit" by dividing by sqrt(1e-9 + abs(array))

        Returns
            the calibration is applied in-place, the returned array is the same instance as the input image
            if you need an out-of-place calibration, copy the array first
        """
        assert method in ("sigma", "gamma", "beta")
        assert image.shape == roi.get_shape()

        # IPF defines the calibration methods with 'Nought' postfix except for gamma
        if method in ("sigma", "beta"):
            method += "Nought"

        window = roi.to_roi()
        calib_array = self._get_calibration_array(window, method)
        noise_array = self._get_noise_array(window) if self.has_noise else None

        return _apply_radiometric_calibration(
            image, calib_array, noise_array, dont_clip_noise, as_amplitude
        )

    def _load_calibration(self, calibration_xml_content):
        lines, pixels, values = _read_lut_from_calibration_xml(calibration_xml_content)

        self._lines = np.array(lines)
        self._pixels = np.array(pixels[0])
        self._values = values

        assert self._lines[0] <= 0
        assert self._pixels[0] <= 0
        # we kept only the first row of pixels, because they are all the same (it's a grid)
        assert all((p == self._pixels).all() for p in pixels)
        # we should have one value per grid node
        assert len(self._lines) == len(self._values["gamma"])
        assert (
            len(self._lines) * len(self._pixels)
            == np.asarray(self._values["gamma"]).size
        )

    def _load_noise(self, noise_xml_content):
        lines, pixels, values, azimuth_blocks = _read_lut_from_noise_xml(
            noise_xml_content
        )

        self._noise_lines = np.array(lines)
        self._noise_azimuth_blocks = azimuth_blocks

        # Densify the noise blocks to make the grid regular.
        # This is necessary for the GRD products when the pixels arrays sizes vary in azimuth.
        all_pixels = sorted(list(set(p for px in pixels for p in px)))
        self._noise_pixels = np.array(all_pixels)
        self._noise_values = np.zeros((len(lines), self._noise_pixels.size))
        for i in range(len(values)):
            # The following can happen at the begining of a GRD datatake (probably at the end too)
            # IW1__|IW2__|IW3__
            # x x x|0 0 0|0 0 0 (x represents an arbitrary range noise value (in `values`)
            # x x x|x x x|0 0 0 (sometimes the value is 0, which makes further interpolation quite bad)
            # x x x|x x x|x x x (because these 0 can even happen inside valid SAR regions)
            # so we replace the 0 by an extrapolation of the line.
            # This is not an accurate denoising factor, but it is much better than "0" which creates gradients after interpolation.
            nonzeros = np.asarray(values[i]) != 0
            pixels[i] = np.asarray(pixels[i])[nonzeros]
            values[i] = np.asarray(values[i])[nonzeros]

            # densify and extrapolate the range noise vectors
            self._noise_values[i, :] = np.interp(
                self._noise_pixels, pixels[i], values[i]
            )

            # some noise maps have negative values, which creates signal during the calibration
            # so we clip to 0 the noise map
            # ex: S1B_IW_GRDH_1SDV_20210605T230132_20210605T230150_027227_0340A2_9DF3 vv (lon lat -68.43241, -8.13822)
            self._noise_values[i, :] = np.maximum(self._noise_values[i, :], 0)

        # assertions
        assert self._noise_lines[0] <= 0
        assert self._noise_pixels[0] <= 0

        # we should have one value per grid node
        assert len(self._noise_lines) == len(self._noise_values)
        assert (
            len(self._noise_lines) * len(self._noise_pixels)
            == np.asarray(self._noise_values).size
        )

    def _get_calibration_array(self, window, method):
        values = np.array(self._values[method])
        return _bilinear_interpolation(window, self._lines, self._pixels, values)

    def _get_noise_array(self, window):
        range_noise = _bilinear_interpolation(
            window, self._noise_lines, self._noise_pixels, self._noise_values
        )

        if self._noise_azimuth_blocks is not None:
            x, y, w, h = window
            assert range_noise.shape == (h, w)

            for block in self._noise_azimuth_blocks:
                startx = block.first_range_sample
                endx = block.last_range_sample + 1  # excluded
                starty = block.first_azimuth_line
                endy = block.last_azimuth_line + 1  # excluded

                # check if we are out of the swath
                if x + w < startx or x >= endx:
                    continue
                if y + h < starty or y >= endy:
                    continue

                # interpolate the values along azimuth
                lines = block.lines
                noise_azimuth_LUT = block.noise_azimuth_LUT

                # interpolate the noise azimuth LUT
                ys = np.arange(max(y, starty), min(y + h, endy), dtype=np.int16)
                azimuth_noise = np.interp(ys, lines, noise_azimuth_LUT)

                # consider the intersection between the block and the ROI
                sx = max(x, startx) - x
                ex = min(x + w, endx) - x

                # adjust the range_noise
                range_noise[ys - y, sx:ex] *= azimuth_noise[:, None]

        # Scaling factor to apply while calibrating, following IPF version. Based on doc:
        # Masking "No-value" Pixels on GRD Products generated by the Sentinel - 1 ESA IPF
        if self._ipf is not None and self._ipf < "002.50":
            knoise = 75088.7
            range_noise *= knoise * self._values["dn"][0][0] ** 2

        return range_noise


@dataclass(frozen=True)
class CalibrationReader(ImageReader):
    """Class to calibrate after reading the data"""

    reader: ImageReader
    """Any ImageReader object (has .read(index, window)). Reader to the tiff of the product."""
    calibrator: Sentinel1Calibrator
    """Calibrator on the same product (same swath/polarization)."""
    method: str
    """Calibration method (either "sigma", "gamma", "beta")."""
    dont_clip_noise: bool = False
    """
    If true, during noise calibration, values are not clipped to 0 but stay positive.
    This is what happens in the implementation of SNAP. The default is False.
    """
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
                    self.dont_clip_noise,
                    as_amplitude=self.as_amplitude,
                )
        else:
            self.calibrator.calibrate_inplace(
                array,
                roi,
                self.method,
                self.dont_clip_noise,
                as_amplitude=self.as_amplitude,
            )

        return array
