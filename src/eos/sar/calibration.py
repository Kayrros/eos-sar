import numpy as np

from . import _calibration as _cal  # type: ignore


def bilinear_interpolation(window, lines, pixels, values):
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


def apply_radiometric_calibration(
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
