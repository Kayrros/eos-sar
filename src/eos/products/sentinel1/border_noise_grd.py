import numpy as np


def compute_mask_top_bottom(mask):
    """
    Compute a top or bottom border mask
    Parameters
    ----------
    mask : ndarray of bool
        first roughly computed mask that needs to be refined

    Returns
    -------
    out_mask : ndarray of bool
        output mask for top or down border
    """
    assert mask.dtype == np.bool_

    out_mask = np.full(mask.shape, False)
    for i in range(mask.shape[1]):
        col = mask[:, i]
        if np.any(col):
            index = np.nonzero(col)[0][0]
            out_mask[index:, i] = True

    return out_mask


def compute_mask_left_right(mask):
    """
    Compute a left or right border mask
    Parameters
    ----------
    mask : ndarray of bool
        first roughly computed mask that needs to be refined

    Returns
    -------
    out_mask : ndarray of bool
        output mask for top or down border
    """
    assert mask.dtype == np.bool_

    out_mask = np.full(mask.shape, False)
    for i in range(mask.shape[0]):
        line = mask[i, :]
        if np.any(line):
            index = np.nonzero(line)[0][0]
            out_mask[i, index:] = True

    return out_mask


def get_border_flags(mask):
    """
    Find type of border on the mask
    Parameters
    ----------
    mask: ndarray of bool
        first roughly computed mask that needs to be refined

    Returns
    -------
    top : bool
        flag for top border
    bottom : bool
        flag for bottom border
    left : bool
        flag for left border
    right : bool
        flag for right border
    """
    assert mask.dtype == np.bool_

    top = mask[0, :].mean() < 0.5
    bottom = mask[-1, :].mean() < 0.5
    left = mask[:, 0].mean() < 0.5
    right = mask[:, -1].mean() < 0.5

    return top, bottom, left, right


def compute_border_masks(mask, top_border, bottom_border, left_border, right_border):
    """
    Compute a mask for each border and merge them
    Parameters
    ----------
    mask : ndarray of bool
        first roughly computed mask that needs to be refined
    top_border : bool
        flag for top border
    bottom_border  : bool
        flag for bottom border
    left_border : bool
        flag for left border
    right_border : bool
        flag for right border

    Returns
    -------
    out_mask : ndarray of bool
        final mask
    """
    assert mask.dtype == np.bool_

    out_mask = np.full(mask.shape, True)

    if top_border:
        out_mask &= compute_mask_top_bottom(mask)

    if bottom_border:
        out_mask &= np.flipud(compute_mask_top_bottom(np.flipud(mask)))

    if left_border:
        out_mask &= compute_mask_left_right(mask)

    if right_border:
        out_mask &= np.fliplr(compute_mask_left_right(np.fliplr(mask)))

    return out_mask


def compute_border_mask(img):
    """
    Compute a border mask (merging all borders together)
    Parameters
    ----------
    img : ndarray of float
        GRD raster from which the mask is computed. Has to be already calibrated and denoised

    Returns
    -------
    mask : ndarray of bool
        final mask
    """
    assert img.dtype == np.float32

    # first roughly computed mask that needs to be refined
    init_mask = img > 0.0

    top, bottom, left, right = get_border_flags(init_mask)
    if any([top, bottom, left, right]):
        mask = compute_border_masks(init_mask, top, bottom, left, right)
    else:
        mask = np.full(img.shape, True)

    return mask


def apply_border_mask(img, mask, no_data=np.nan):
    """
    Compute and apply a border mask
    Parameters
    ----------
    img : ndarray of float
        raster to which the mask is applied
    mask : ndarray of bool
        computed mask
    no_data : float
        no data value, default is np.nan

    Returns
    -------
    img : raster with masked no data
    """
    assert mask.dtype == np.bool_

    img[~mask] = no_data

    return img
