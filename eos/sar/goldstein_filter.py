from functools import partial
import itertools
import multiprocessing.dummy

import numpy as np
from scipy.ndimage import uniform_filter

from eos.sar.roi import Roi

# TODO check weighting scheme
"""
normalization might be required around the borders of the image?
"""


def triangular_filter(fft_size):
    """
    Triangular weighting used for patch recombination.

    Parameters
    ----------
    fft_size : int
        Size of the patch to be denoised. For efficient fft processing, this is
        usually a power of 2.

    Returns
    -------
    filt : np.2darray (fft_size, fft_size)
        Triangular filter used for patch recombination.

    """
    half_fft = fft_size / 2
    curr_idx = np.arange(fft_size)
    filt_tri = 1 - np.abs(curr_idx - half_fft + .5) / half_fft
    # divide by 2 to normalize, assuming overlap of fft_size/4
    tri_row = filt_tri / 2
    return np.outer(tri_row, tri_row)


def extract_patch_rois(img_shape, step):
    """
    Extract overlapped patches of size (4*step, 4*step) from an image at each step pixels.

    Parameters
    ----------
    img_shape : tuple (h, w)
        Shape of the image from which windows are to be extracted.
    step : int
        Step to take in pixels between patches. The patch shape will be (4 * step, 4 * step).

    Yields
    -------
    patch_roi : eos.sar.roi.Roi
        Roi of the patch to extract.

    """
    # TODO verif pour les bords
    height, width = img_shape

    win_size = 4 * step

    x_index = [x for x in range(0, width - win_size, step)]
    x_index.append(width - win_size)

    y_index = [x for x in range(0, height - win_size, step)]
    y_index.append(height - win_size)

    indices = list(itertools.product(x_index, y_index))

    for col, row in indices:
        patch_roi = Roi(col, row, win_size, win_size)
        yield patch_roi


def transform_one_window(patch_roi, full_ifg, alpha, window_size, filt_triangle):
    """
    Filter the patch using Goldstein's method [1].

    Parameters
    ----------
    patch_roi : eos.sar.roi.Roi
        Roi of the InSAR patch to extract from an interferogram obtained
        by the hermitian product of a primary image and secondary image.
    full_ifg : np.2darray
        Full (not cropped at the patch) interferogram array.
    alpha : float
        Float in [0, 1] that controls the filtering power. 0 means no filtering
        and 1 means maximal filtering.
    window_size : int
        Size of the uniform smoothing filter applied to the power spectrum.
    filt_triangle : np.2darray
        Triangular filter used for patch recombination.

    Returns
    -------
    patch_roi : eos.sar.roi.Roi
        Roi of the InSAR patch extracted, to be used for recombination.
    filtered : np.2darray
        Filtered patch.

    Notes
    -----
    The Goldstein filter is designed in the Fourier domain. First, the amplitude
    of the Fourier transform of the complex interferogram is taken. Then it is smoothed.
    Then it is taken to the power alpha. This is the filter that multiplies the fourier
    transform of the patch. An inverse fft is taken and a spatial triangular weighting
    is applied for patch recombination.

    [1] R. M. Goldstein and C. L. Werner, “Radar interferogram filtering for geophysical applications,” 1998.

    """
    assert patch_roi.get_shape() == filt_triangle.shape, "Patches and triangular filter must have the same shape"

    fft_win = np.fft.fft2(patch_roi.crop_array(full_ifg))

    # TODO: Change mode? mode="constant"
    filtered = uniform_filter(np.fft.fftshift(np.abs(fft_win)),
                              size=window_size)

    filtered = np.fft.ifftshift(filtered) ** alpha

    assert filtered[0, 0], "Something is wrong, the filter is supposed to be\
    low-pass, the 0 frequency should not be suppressed"
    # normalize the filter so that its coefficients spatially sum up to 1
    filtered /= filtered[0, 0]

    filtered = np.multiply(fft_win, filtered)
    del fft_win  # not needed anymore

    filtered = np.fft.ifft2(filtered)

    filtered = np.multiply(filtered, filt_triangle)

    return patch_roi, filtered


def apply(img, fft_size: int = 32, window_size: int = 5, alpha: float = .5, nworkers: int = 1):
    """
    Apply the Goldstein filtering for an Interferogram.

    Parameters
    ----------
    img : np.2darray
        Interferogram to be filtered.
    fft_size : int
        Size of the patches that are extracted and denoised. For efficient fft processing, this is
        usually a power of 2. The default is 32.
    window_size : int
        Size of the uniform smoothing filter applied to the power spectrum. The default is 5.
    alpha : float
        Float in [0, 1] that controls the filtering power. 0 means no filtering
        and 1 means maximal filtering. The default is 0.5.
    nworkers : int, optional
        The number of cores to use in parallel processing. The default is 1.

    Returns
    -------
    out_image : np.2darray
        Filtered interferogram.

    """
    assert img.dtype in (np.csingle, np.cdouble)
    assert alpha >= 0 and alpha <= 1, "alpha out of bounds"
    assert window_size < fft_size, "smoothing window size should be less then patch size"

    filt_triangle = triangular_filter(fft_size)
    out_image = np.zeros(img.shape, dtype=img.dtype)
    patch_roi_generator = extract_patch_rois(img.shape, int(fft_size / 4))

    proc_window = partial(transform_one_window, full_ifg=img, alpha=alpha,
                          window_size=window_size, filt_triangle=filt_triangle)

    if nworkers > 1:
        pool = multiprocessing.dummy.Pool(nworkers)
        result = pool.map(proc_window, patch_roi_generator)
    else:
        result = map(proc_window, patch_roi_generator)

    for patch_roi, transformed_win in result:
        col, row, w, h = patch_roi.to_roi()
        out_image[row:row + h, col:col + w] += transformed_win

    if nworkers > 1:
        pool.close()

    return out_image
