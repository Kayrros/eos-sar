import itertools
import multiprocessing.dummy
from functools import partial

import numpy as np
from scipy.ndimage import uniform_filter

from eos.sar.roi import Roi


def triangular_filter(step: int):
    """
    Triangular weighting used for patch recombination.

    Parameters
    ----------
    step : int
        Stride of patch selection. Patches of size 4 * step will be taken each step.

    Returns
    -------
    filt : np.2darray (fft_size, fft_size)
        Triangular filter used for patch recombination.

    """
    half_size = 2 * step
    curr_idx = np.arange(4 * step)
    filt_tri = 1 - np.abs(curr_idx - half_size + 0.5) / half_size
    # divide by 2 to normalize, because of stride of step
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
        Stride of patch selection. Patches of size 4 * step will be taken each step.

    Yields
    -------
    patch_roi : eos.sar.roi.Roi
        Roi of the patch to extract.

    """
    height, width = img_shape

    win_size = 4 * step

    x_index = np.arange(0, width - win_size + 1, step)
    y_index = np.arange(0, height - win_size + 1, step)

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
    assert patch_roi.get_shape() == filt_triangle.shape, (
        "Patches and triangular filter must have the same shape"
    )

    fft_win = patch_roi.crop_array(full_ifg)
    nan_mask = np.isnan(fft_win)

    any_nan = np.any(nan_mask)

    if np.all(nan_mask):
        # if all the patch is NaN, return NaN as a filtered result
        return patch_roi, fft_win
    elif any_nan:
        # otherwise, convert NaNs to zeros
        fft_win = fft_win.copy()
        fft_win[nan_mask] = 0

    fft_win = np.fft.fft2(fft_win)

    filtered = uniform_filter(
        np.fft.fftshift(np.abs(fft_win)), size=window_size, mode="wrap"
    )

    filtered = np.fft.ifftshift(filtered) ** alpha

    # there is a risk to raise this error if the patch is fully 0
    # In practice, we should not have 0 data in the interferogram
    assert filtered[0, 0], (
        "Something is wrong, the filter is supposed to be\
    low-pass, the 0 frequency should not be suppressed"
    )
    # normalize the filter so that its coefficients spatially sum up to 1
    filtered /= filtered[0, 0]

    filtered = np.multiply(fft_win, filtered)
    del fft_win  # not needed anymore

    filtered = np.fft.ifft2(filtered)

    if any_nan:
        # put back NaN values
        filtered[nan_mask] = np.nan

    filtered = np.multiply(filtered, filt_triangle)

    return patch_roi, filtered


def dim_padding(length: int, step: int) -> tuple[int, int]:
    """
    Determine the padding needed to apply to a dimension to reduce border effects.

    Parameters
    ----------
    length : int
        Length of samples in the dimension.
    step : int
        Stride of patch selection. Patches of size 4 * step will be taken each step.

    Returns
    -------
    tuple[int, int]
        Tuple of lower and upper padding.

    """
    # on the start of the interval, the padding is obtained easily
    pad = 3 * step

    # on the end of the interval, the padding needs to account for
    # the ceil to the nearest multiple of step + 3 * step
    upper_pad = np.ceil(length / step) * step - length  # integer amount of step
    upper_pad += pad

    return pad, int(upper_pad)


def pad_img(img, step: int):
    """
    Pad an image to minimize border effects.

    Parameters
    ----------
    img : np.ndarray
        Image to pad.
    step : int
        Stride of patch selection. Patches of size 4 * step will be taken each step.

    Returns
    -------
    padded_img : np.ndarray
        Padded image.
    orig_roi : eos.sar.roi.Roi
        Roi of original image in the padded image.

    """
    h, w = img.shape

    up_pad, down_pad = dim_padding(h, step)
    left_pad, right_pad = dim_padding(w, step)

    padded_img = np.pad(
        img, ((up_pad, down_pad), (left_pad, right_pad)), constant_values=np.nan
    )

    orig_roi = Roi(left_pad, up_pad, w, h)

    return padded_img, orig_roi


def apply(
    img, step: int = 8, window_size: int = 5, alpha: float = 0.5, nworkers: int = 1
):
    """
    Apply the Goldstein filtering for an Interferogram. If the img contains NaNs, the output
    image will contain NaNs at the same location. NaNs are converted to zeros in the computation
    and nearby values might be affected (attenuation or ripples). In the worst case, affected values are pixels spanning from
    the NaN pixel until 4 * step in both dimensions.

    Parameters
    ----------
    img : np.2darray
        Interferogram to be filtered.
    step : int
        Stride of patch selection. Patches of size 4 * step will be taken each step.
        For efficient fft processing, this is usually a power of 2. The default is 8.
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
    assert window_size < 4 * step, (
        "smoothing window size should be less than patch size"
    )

    filt_triangle = triangular_filter(step)

    img, orig_roi = pad_img(img, step)

    out_image = np.zeros(img.shape, dtype=img.dtype)

    patch_roi_generator = extract_patch_rois(img.shape, step)

    proc_window = partial(
        transform_one_window,
        full_ifg=img,
        alpha=alpha,
        window_size=window_size,
        filt_triangle=filt_triangle,
    )

    if nworkers > 1:
        pool = multiprocessing.dummy.Pool(nworkers)
        result = pool.map(proc_window, patch_roi_generator)
    else:
        result = map(proc_window, patch_roi_generator)

    for patch_roi, transformed_win in result:
        col, row, w, h = patch_roi.to_roi()
        out_image[row : row + h, col : col + w] += transformed_win

    if nworkers > 1:
        pool.close()

    return orig_roi.crop_array(out_image)
