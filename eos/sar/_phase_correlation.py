""" 
Note:
    Most of the code is was taken and adapted from centreborelli/itsr (Carlo de Franchis and Charles Hessel)
"""
from typing import Optional

import numpy as np
from scipy import ndimage, optimize


def sinc(x, a: float, b: float, c: float) -> float:
    """
    Parametric sinus cardinal: a, b, c are real numbers, x is a vector.
    """
    return c * np.sinc(b * (x - a))


def sinc_fit(x, y) -> tuple[float, float, float]:
    """
    Return the triplet (a, b, c) minimizing ||sinc(x, a, b, c) - y||^2.
    """
    try:
        return optimize.curve_fit(sinc, x, y, p0=[x[1], 1, 1])[0]
    except RuntimeError:  # Optimal parameters not found
        return np.nan, np.nan, np.nan


def phase_correlation(a, b):
    """
    Compute phase correlation of two images given by their Fourier transforms.

    Args:
        a, b: 2D complex ndarrays, Fourier transforms of images a and b

    Returns:
        2D ndarray with phase correlation image
    """
    c = np.nan_to_num(a * np.conj(b))
    s = np.abs(c)
    c = np.divide(c, s, out=np.zeros_like(c), where=s != 0)
    ifft = np.real(np.fft.ifft2(c))
    ifft = np.fft.fftshift(ifft)
    return ifft


def registering_shift_from_phase_correlation(
    u_fft,
    v_fft,
    pc_threshold: float = 0.0,
    second_max_threshold: Optional[float] = None,
    max_shift: Optional[float] = None,
):
    """
    Estimate the best translation registering an image on top of another.

    Args:
        u_fft, v_fft: 2D numpy arrays containing the Fourier Transforms of the
            input images
        second_max_threshold: discard shift (return nans) if the ratio between
            the second largest local maximum and the largest one is inferior to
            this threshold
        pc_threshold: discard shift (return nans) if the maximal correlation
            value found is inferior to this threshold

    Returns:
        numpy array containing the output translation vector
        maximal phase correlation score
    """
    c = phase_correlation(u_fft, v_fft)

    # find the local maxima in the max_shift range
    local_max = np.transpose((ndimage.maximum_filter(c, size=3) == c).nonzero())
    if max_shift is not None:
        ind = np.where(
            np.linalg.norm(
                local_max - np.floor(np.array(u_fft.shape) / 2).astype(int), axis=1
            )
            < max_shift
        )
        local_max = local_max[ind]
    local_max_values = c[local_max[:, 0], local_max[:, 1]]

    if len(local_max_values) == 0:
        return np.array([np.nan, np.nan]), np.nan

    if np.max(local_max_values) < pc_threshold:
        return np.array([np.nan, np.nan]), np.nan

    # check if the first maximum is strong enough wrt to the second maximum
    if second_max_threshold is not None and len(local_max_values) > 1:
        a, b = sorted(np.partition(local_max_values, -2)[-2:], reverse=True)
        if b / a > second_max_threshold:
            return np.array([np.nan, np.nan]), np.nan

    y, x = local_max[np.argmax(local_max_values)]
    if 0 < y < c.shape[0] - 1:
        t = [y - 1, y, y + 1]
        v = [c[s, x] for s in t]
        ay, by, cy = sinc_fit(t, v)
    else:
        return np.array([np.nan, np.nan]), np.nan
    if 0 < x < c.shape[1] - 1:
        t = [x - 1, x, x + 1]
        v = [c[y, s] for s in t]
        ax, bx, cx = sinc_fit(t, v)
    else:
        return np.array([np.nan, np.nan]), np.nan
    return np.array([ay, ax]) - np.floor(np.array(u_fft.shape) / 2), max(cx, cy)


def compute_fft(b):
    return np.fft.fft2(np.nan_to_num(b)).astype(np.complex64)
