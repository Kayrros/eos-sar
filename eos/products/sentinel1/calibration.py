import numpy as np
import xmltodict
import datetime

from eos.sar.roi import Roi
from . import _calibration as _cal


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
                float(v["firstAzimuthLine"]),
                float(v["firstRangeSample"]),
                float(v["lastAzimuthLine"]),
                float(v["lastRangeSample"]),
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
        if int(noise_azimuth_vector_list['@count']) == 1:
            noise_azimuth_vector_list = [noise_azimuth_vector_list["noiseAzimuthVector"], ]
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
        lines.append(float(v["line"]))
        pixels.append(list(map(float, v["pixel"]["#text"].split())))
        values.append(list(map(float, v[lut_key]["#text"].split())))

    # check lists lengths
    if (len(lines) != len(pixels) or
        len(pixels) != len(values) or
            any(len(p) != len(v) for p, v in zip(pixels, values))):
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
    values = {
        "sigmaNought": [],
        "betaNought": [],
        "gamma": [],
        "dn": []
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
        if (len(pixels) != len(values[k]) or
                any(len(p) != len(v) for p, v in zip(pixels, values[k]))):
            raise CalibrationError("Unexpected data format in calibration xml")

    return lines, pixels, values


def _bilinear_interpolation(window, lines, pixels, values):
    x, y, w, h = window
    if y + h > lines[-1]:
        values = np.pad(values, ((0, 1), (0, 0)), mode='edge')
        lines = np.append(lines, y + h)

    if x + w > pixels[-1]:
        values = np.pad(values, ((0, 0), (0, 1)), mode='edge')
        pixels = np.append(pixels, x + w)

    res = _cal.bilinear_interpolation(window, lines.astype(np.int32), pixels.astype(np.int32), values.astype(np.float32))
    return res


def _apply_radiometric_calibration(img, calib_coeffs, noise_coeffs, dont_clip_noise):
    if np.iscomplexobj(img):
        assert img.dtype == np.complex64
        assert calib_coeffs.dtype == np.float32
        assert noise_coeffs is None or noise_coeffs.dtype == np.float32
        _cal.apply_radiometric_calibration_complex64(img, calib_coeffs, noise_coeffs, dont_clip_noise)
        return img
    else:
        assert img.dtype == np.float32
        assert calib_coeffs.dtype == np.float32
        assert noise_coeffs is None or noise_coeffs.dtype == np.float32
        _cal.apply_radiometric_calibration_float32(img, calib_coeffs, noise_coeffs, dont_clip_noise)
        return img


def _get_product_date(calibration_xml_content):
    year, month, day = xmltodict.parse(calibration_xml_content)["calibration"][
        "adsHeader"]["startTime"].split('T')[0].split('-')
    return datetime.datetime(int(year), int(month), int(day))


class Sentinel1Calibrator:
    """
    Radiometric calibration for Sentinel1 SLC products.

    Example
        >>> calibrator_noise = calibration.Sentinel1Calibrator(cal_xml, noise_xml)
        >>> calibrator_noise.calibrate_inplace(myarray, window, "beta")

    Note
        s1tbx (8.0.5) does not clip to 0 and instead keep the original value of the pixel.
        Expect some small differences between eos and SNAP because of this. Any other difference should be reported!
    """

    def __init__(self, calibration_xml_content, noise_xml_content=None):
        self._date = _get_product_date(calibration_xml_content)
        self._load_calibration(calibration_xml_content)
        if noise_xml_content:
            self._load_noise(noise_xml_content)
            self.has_noise = True
        else:
            self.has_noise = False

    def calibrate_inplace(self, image, roi, method, dont_clip_noise=False):
        """
            Apply the radiometric calibration on the given raster, at position `window` of the SLC tif image.

            Args
                image (np.array): array of shape (h, w), of type float32 or complex64
                roi (Roi): position in the source SLC tif image
                method (str): 'sigma' | 'gamma' | 'beta'
                dont_clip_noise (bool, default False):
                    if true, during noise calibration, values are not clipped to 0 but stay positive
                    this is what happens in the implementation of SNAP

            Returns
                the calibration is applied in-place, the returned array is the same instance as the input image
                if you need an out-of-place calibration, copy the array first
        """
        assert method in ('sigma', 'gamma', 'beta')
        assert image.shape == roi.get_shape()

        # IPF defines the calibration methods with 'Nought' postfix except for gamma
        if method in ('sigma', 'beta'):
            method += 'Nought'

        window = roi.to_roi()
        calib_array = self._get_calibration_array(window, method)
        noise_array = self._get_noise_array(window) if self.has_noise else None

        return _apply_radiometric_calibration(image, calib_array, noise_array, dont_clip_noise)

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
        assert len(self._lines) == len(self._values['gamma'])
        assert len(self._lines) * len(self._pixels) == np.asarray(self._values['gamma']).size

    def _load_noise(self, noise_xml_content):
        lines, pixels, values, azimuth_blocks = _read_lut_from_noise_xml(noise_xml_content)

        self._noise_lines = np.array(lines)

        # re-define the pixel array as the pixels positions of the first line:
        # min and max positions for all the noise LUT are covered.
        # this could be necessary for the GRD products when the pixels arrays sizes vary with the line
        self._noise_pixels = np.array(pixels[0])

        # get the noise values according to the new pixels
        self._noise_values = np.zeros((len(lines), self._noise_pixels.size))
        for i in range(len(values)):
            self._noise_values[i, :] = np.interp(self._noise_pixels, pixels[i], values[i])

        # Scaling factor to apply while calibrating, following IPF version. Based on doc:
        # Masking "No-value" Pixels on GRD Products generated by the Sentinel - 1 ESA IPF
        if self._date < datetime.datetime(2015, 6, 30):
            knoise = 75088.7
            scaling_factor = knoise * calib_array[0, 0] ** 2
        else:
            scaling_factor = 1.0

        self._noise_values = self._noise_values * scaling_factor

        # assertions
        assert self._noise_lines[0] <= 0
        assert self._noise_pixels[0] <= 0

        # check again that all pixels used for the new grid correspond to the first row of pixels
        assert all(self._noise_pixels == np.array(pixels[0]))

        # we should have one value per grid node
        assert len(self._noise_lines) == len(self._noise_values)
        assert len(self._noise_lines) * len(self._noise_pixels) == np.asarray(self._noise_values).size

        if azimuth_blocks is not None:
            self._noise_azimuth_block = azimuth_blocks[0]
        else:
            self._noise_azimuth_block = None

    def _get_calibration_array(self, window, method):
        values = np.array(self._values[method])
        return _bilinear_interpolation(window, self._lines, self._pixels, values)

    def _get_noise_array(self, window):
        range_noise = _bilinear_interpolation(window, self._noise_lines, self._noise_pixels, self._noise_values)

        if self._noise_azimuth_block is not None:
            _, y, _, h = window
            ys = np.arange(y, y + h, dtype=np.int16)
            azimuth_noise = np.interp(ys, self._noise_azimuth_block.lines, self._noise_azimuth_block.noise_azimuth_LUT)
            range_noise *= azimuth_noise[:, None]

        return range_noise


class CalibrationReader:
    """Class to calibrate after reading the data"""

    def __init__(self, reader, calibrator: Sentinel1Calibrator,
                 method: str, dont_clip_noise=False):
        """
        Constructor.

        Parameters
        ----------
        reader : any reader object (has .read(index, window))
            Reader to the tiff of the product.
        calibrator : Sentinel1Calibrator
            Calibrator on the same product (same swath/polarization).
        method : str
            Calibration method (either "sigma", "gamma", "beta").
        dont_clip_noise : boolean, optional
            if true, during noise calibration, values are not clipped to 0 but stay positive
            this is what happens in the implementation of SNAP. The default is False.

        Returns
        -------
        None.

        """
        self.reader = reader
        self.calibrator = calibrator
        self.method = method
        self.dont_clip_noise = dont_clip_noise

    def read(self, index, window, **kwargs):
        """
        Read and calibrate the data.

        Parameters
        ----------
        index : int
            Band index.
        window : tuple
            ((row, row+h), (col, col+w)).

        Returns
        -------
        ndarray
            Array read and calibrated.

        """
        array = self.reader.read(index, window=window, **kwargs)

        (y, yh), (x, xw) = window
        h = yh - y
        w = xw - x
        roi = Roi(x, y, w, h)
        self.calibrator.calibrate_inplace(array, roi, self.method, self.dont_clip_noise)

        # undo the pow2 from the calibration
        return array / np.sqrt(1e-9 + np.abs(array))
