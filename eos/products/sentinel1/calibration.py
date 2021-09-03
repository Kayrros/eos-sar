import numpy as np
from s1c import read_lut_from_calibration_xml, read_lut_from_noise_xml

from . import _calibration as _cal


def bilinear_interpolation(window, lines, pixels, values):
    x, y, w, h = window
    if y + h > lines[-1]:
        values = np.pad(values, ((0, 1), (0, 0)), mode='edge')
        lines = np.append(lines, y + h)

    if x + w > pixels[-1]:
        values = np.pad(values, ((0, 0), (0, 1)), mode='edge')
        pixels = np.append(pixels, x + w)

    res = _cal.bilinear_interpolation(window, lines.astype(np.int32), pixels.astype(np.int32), values.astype(np.float32))
    return res


def apply_radiometric_calibration(img, calib_coeffs, noise_coeffs):
    if np.iscomplexobj(img):
        assert img.dtype == np.complex64
        assert calib_coeffs.dtype == np.float32
        assert noise_coeffs is None or noise_coeffs.dtype == np.float32
        _cal.apply_radiometric_calibration_complex64(img, calib_coeffs, noise_coeffs)
        return img
    else:
        assert img.dtype == np.float32
        assert calib_coeffs.dtype == np.float32
        assert noise_coeffs is None or noise_coeffs.dtype == np.float32
        _cal.apply_radiometric_calibration_float32(img, calib_coeffs, noise_coeffs)
        return img


class Calibrator:
    '''
    Example
    >>> calibrator_noise = calibration.Calibrator(cal_xml, noise_xml)
    >>> calibrator_noise.calibrate_inplace(myarray, window, "beta")

    Note
        s1tbx (8.0.5) does not clip to 0 and instead keep the original value of the pixel.
        Expect some small differences between eos and SNAP because of this. Any other difference should be reported!
    '''

    def __init__(self, calibration_xml_content, noise_xml_content=None):
        self._load_calibration(calibration_xml_content)
        if noise_xml_content:
            self._load_noise(noise_xml_content)
            self.has_noise = True
        else:
            self.has_noise = False

    def calibrate_inplace(self, image, window, method):
        assert method in ('sigma', 'gamma', 'beta')

        # IPF defines the calibration methods with 'Nought' postfix except for gamma
        if method in ('sigma', 'beta'):
            method += 'Nought'

        calib_array = self._get_calibration_array(window, method)
        noise_array = self._get_noise_array(window) if self.has_noise else None
        return apply_radiometric_calibration(image, calib_array, noise_array)

    def _load_calibration(self, calibration_xml_content):
        lines, pixels, values = read_lut_from_calibration_xml(calibration_xml_content)

        self._lines = np.array(lines)
        self._pixels = np.array(pixels[0])
        self._values = values

    def _load_noise(self, noise_xml_content):
        lines, pixels, values, azimuth_blocks = read_lut_from_noise_xml(noise_xml_content)

        self._noise_lines = np.array(lines)
        self._noise_pixels = np.array(pixels[0])
        self._noise_values = np.array(values)

        if azimuth_blocks is not None:
            self._noise_azimuth_block = azimuth_blocks[0]
        else:
            self._noise_azimuth_block = None

    def _get_calibration_array(self, window, method):
        values = np.array(self._values[method])
        return bilinear_interpolation(window, self._lines, self._pixels, values)

    def _get_noise_array(self, window):
        range_noise = bilinear_interpolation(window, self._noise_lines, self._noise_pixels, self._noise_values)

        if self._noise_azimuth_block is not None:
            _, y, _, h = window
            ys = np.arange(y, y + h, dtype=np.int16)
            azimuth_noise = np.interp(ys, self._noise_azimuth_block.lines, self._noise_azimuth_block.noise_azimuth_LUT)
            range_noise *= azimuth_noise[:,None]

        return range_noise
