import numpy as np
import scipy.ndimage as ndimage


def _uniform_spatial_filter(u, filter_size):
    return ndimage.uniform_filter(u, size=filter_size, mode="nearest")


def _gaussian_spatial_filter(u, filter_size):
    if isinstance(filter_size, tuple):
        sigma = map(lambda f: f / 3, filter_size)
    else:
        sigma = filter_size / 3
    return ndimage.gaussian_filter(u, sigma=sigma, mode="nearest")


SPATIAL_FILTERS = {
    'uniform': _uniform_spatial_filter,
    'gaussian': _gaussian_spatial_filter,
}


def _compute_filtered_magnitude(im, filter_size, spatial_filter):
    assert im.dtype in (np.csingle, np.cdouble)
    return spatial_filter(np.abs(im) ** 2, filter_size)


def _modify_borders(coherence, filter_size):
    if isinstance(filter_size, tuple):
        hs = list(map(lambda f: f // 2, filter_size))
    else:
        hs = (filter_size // 2, filter_size // 2)
    if hs[0]:
        coherence[:hs[0],:] = np.nan
        coherence[-hs[0]:,:] = np.nan
    if hs[1]:
        coherence[:,:hs[1]] = np.nan
        coherence[:,-hs[1]:] = np.nan


def on_pair(im1, im2, filter_size, eps=1e-10,
            set_borders_to_nan=False,
            might_contain_nans=False,
            spatial_filter='uniform'):
    """
        Compute the coherence on a pair of complex images.

        im1, im2: complex array of type np.csingle (faster) or np.cdouble (slower).
        filter_size (int or (int,int)): size of the spatial filter

        If one of the input images contains nans, make sure to set might_contain_nans to True (or 'overwrite' to allow to overwrite your arrays). The result will have nans set at the same positions, but values near the nans might be wrong.
        The borders in the resulting coherence map will be wrong, use set_borders_to_nan=True to set them to nan.
    """
    spatial_filter = SPATIAL_FILTERS[spatial_filter]

    mask = None
    if might_contain_nans:
        mask = np.isnan(im1) | np.isnan(im2)
        if might_contain_nans != 'overwrite':
            im1 = im1.copy()
            im2 = im2.copy()
        im1[mask] = 0
        im2[mask] = 0

    inf1 = _compute_filtered_magnitude(im1, filter_size, spatial_filter)
    inf2 = _compute_filtered_magnitude(im2, filter_size, spatial_filter)
    sup = spatial_filter(im2 * np.conjugate(im1), filter_size)
    coherence = np.abs(sup) / (np.sqrt(inf1 * inf2) + eps)

    if mask is not None:
        coherence[mask] = np.nan

    if set_borders_to_nan:
        _modify_borders(coherence, filter_size)

    return coherence
