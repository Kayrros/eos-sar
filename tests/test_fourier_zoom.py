import numpy as np
import pytest

from eos.sar.fourier_zoom import fourier_zoom

val_range = range(1, 8)


@pytest.mark.parametrize("w", val_range)
@pytest.mark.parametrize("h", val_range)
@pytest.mark.parametrize("z", val_range)
def test_fourier_zoom(w, h, z):
    """
    Generate a random image of specified shape and zoom with specified zoom factor.
    Assert that values at integer positions modulo zoom_factor are equal to original
    values before.

    Parameters
    ----------
    w : int
        width of image.
    h : int
        height of image.
    z : int, optional
        zoom_factor. The default is 2.

    Returns
    -------
    None.

    """
    # image with random float values between -1000 and +1000
    image = (2 * np.random.random((h, w)) - 1) * 1e3

    zoomed_image = fourier_zoom(image, z)

    # the pixel values of the zoomed image at positions (0, 0), (0, z), (0,
    # 2*z), ..., (z, 0), (z, z), ... should be equal to the original image
    # values
    np.testing.assert_allclose(zoomed_image[::z, ::z], image)
