# Copyright (C) 2019, Carlo de Franchis <carlo.de-franchis@ens-paris-saclay.fr>

import os
import argparse
import warnings

import rasterio
import numpy as np


warnings.filterwarnings("ignore",
                        category=rasterio.errors.NotGeoreferencedWarning)


def fourier_zoom(image, z=2):
    """
    Zoom an image by zero-padding its Discrete Fourier transform.

    Args:
        image (np.ndarray): 2D grid of pixel values.
        z (float): Factor by which to multiply the dimensions of the image.
            Must be >= 1.

    Returns:
        np.ndarray: zoomed image.
    """
    h, w = image.shape

    # zero padding sizes
    left = np.ceil((z - 1) * w / 2).astype(int)
    right = np.floor((z - 1) * w / 2).astype(int)
    top = np.ceil((z - 1) * h / 2).astype(int)
    bottom = np.floor((z - 1) * h / 2).astype(int)

    # Fourier transform with the zero-frequency component at the center
    ft = np.fft.fftshift(np.fft.fft2(image))

    # the zoom-in is performed by zero padding the Fourier transform
    ft = np.pad(ft, [(top, bottom), (left, right)], 'constant', constant_values=0+0j)

    # apply ifftshift before taking the inverse Fourier transform
    out = np.fft.ifft2(np.fft.ifftshift(ft))

    # if the input is a real-valued image, then keep only the real part
    if np.isrealobj(image):
        out = np.real(out)

    # to preserve the values of the original samples, the L2 norm has to be
    # multiplied by z*z.
    out *= z * z

    # clip values to avoid overflow when casting to input dtype
    if np.issubdtype(image.dtype, np.integer):
        i = np.iinfo(image.dtype)
        out = np.round(out)
    else:
        i = np.finfo(image.dtype)
    out = np.clip(out, i.min, i.max)

    # return the image casted to the input data type
    return out.astype(image.dtype)


def _test_fourier_zoom(w, h, z=2):
    """
    """
    # image with random float values between -1000 and +1000
    image = (2 * np.random.random((h, w)) - 1) * 1e3

    zoomed_image = fourier_zoom(image, z)

    # the pixel values of the zoomed image at positions (0, 0), (0, z), (0,
    # 2*z), ..., (z, 0), (z, z), ... should be equal to the original image
    # values
    np.testing.assert_allclose(zoomed_image[::z, ::z], image)


def test_fourier_zoom():
    """
    """
    for w in range(1, 8):
        for h in range(1, 8):
            for z in range(1, 8):
                _test_fourier_zoom(w, h, z)


def rasterio_write(path, array, profile={}, tags={}):
    """
    Write a numpy array in a tiff or png file with rasterio.

    Args:
        path (str): path to the output tiff/png file
        array (numpy array): 2D or 3D array containing the image to write.
        profile (dict): rasterio profile (ie dictionary of metadata)
        tags (dict): dictionary with additional geotiff tags
    """
    # determine the driver based on the file extension
    extension = os.path.splitext(path)[1].lower()
    if extension in ['.tif', '.tiff']:
        driver = 'GTiff'
    elif extension in ['.png']:
        driver = 'png'
    else:
        raise NotImplementedError('format {} not supported'.format(extension))

    # read image size and number of bands
    array = np.atleast_3d(array)
    height, width, nbands = array.shape

    # define image metadata dict
    profile.update(driver=driver, count=nbands, width=width, height=height,
                   dtype=array.dtype)

    # write to file
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(np.transpose(array, (2, 0, 1)))
        dst.update_tags(**tags)


def main(input_image_path, output_image_path, zoom_factor):
    """
    """
    with rasterio.open(input_image_path, 'r') as f:
        img = f.read().squeeze()

    rasterio_write(output_image_path,
                   fourier_zoom(img, zoom_factor),
                   profile={"tiled": True,
                            "compress": "deflate"})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Fourier zoom'))
    parser.add_argument('input_image', help=('path to the input image file'))
    parser.add_argument('output_image', help=('path to the output image file'))
    parser.add_argument('zoom', type=float, help=('zoom factor'))
    args = parser.parse_args()
    main(args.input_image, args.output_image, args.zoom)
