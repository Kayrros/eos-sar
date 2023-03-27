import numpy as np
import eos.sar


def hrepeat(arr, w):
    """Flip a 1D array vertically and repeat horizontally w times."""
    return np.repeat(arr.reshape(-1, 1), repeats=w, axis=1)


def vrepeat(arr, h):
    """Flip a 1D array horizontally, and repeat vertically h times."""
    return np.repeat(arr.reshape(1, -1), repeats=h, axis=0)


def first_nonzero(arr, axis, invalid_val=-1):
    """Compute the index of the first non zero entry along an axis. If all
    entries are zeroes, invalid_val is returned"""
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def geom_to_raster(approx_geom, transform, px_is_area=True):
    """
    Get the (col, row) coordinates of an approximate geometry in a raster
    defined by a transform.

    Parameters
    ----------
    approx_geom : list of tuples
        Each element is a (lon, lat) tuple.
    transform : affine.Affine
        Raster transform.
    px_is_area : bool, optional
        Convention of a px being an area as opposed to being a point.
        The default is True.

    Returns
    -------
    list of tuples
        Each element is a (col, row) tuple.

    """
    lons = np.array([g[0] for g in approx_geom])
    lats = np.array([g[1] for g in approx_geom])

    col, row = ~transform * (lons, lats)

    if px_is_area:
        # PIXEL_IS_AREA
        col -= 0.5
        row -= 0.5
    return [(c, r) for c, r in zip(col, row)]


def geom_to_raster_roi(approx_geom, transform, raster_shape, px_is_area=True):
    """
    Get the bounding box of an approximate geometry inside a raster defined
    by a transform.

    Parameters
    ----------
    approx_geom : list of tuples
        Each element is a (lon, lat) tuple.
    transform : affine.Affine
        Raster transform.
    raster_shape : tuple
        Shape of the parent raster.
    px_is_area : bool, optional
        Convention of a px being an area as opposed to being a point.
        The default is True.

    Returns
    -------
    raster_roi : eos.sar.roi.Roi
        Roi of the bounding box of the approximate geometry in the raster.

    """
    raster_coords = geom_to_raster(approx_geom, transform, px_is_area)
    col = [coord[0] for coord in raster_coords]
    row = [coord[1] for coord in raster_coords]
    # deduce the bouding box
    raster_bounds = eos.sar.roi.Roi.points_to_bbox(row, col)
    raster_roi = eos.sar.roi.Roi.from_bounds_tuple(raster_bounds)

    raster_roi.make_valid(parent_shape=raster_shape, inplace=True)

    return raster_roi


def raster_xy_grid(raster_shape, transform, px_is_area=True):
    """
    Get the x, y (often longitude and latitude) grid of the raster.

    Parameters
    ----------
    raster_shape : tuple
        (h, w).
    transform : affine.Affine
        Raster transform.
    px_is_area : bool, optional
        Convention of a px being an area as opposed to being a point.
        The default is True.

    Returns
    -------
    x : ndarray (h, w)
        x coordinate for each raster point.
    y : ndarray (h, w)
        y coordinate for each raster pointIPTION.

    """
    # get dem points in crs
    col, row = np.meshgrid(
        np.arange(raster_shape[1]), np.arange(raster_shape[0]))
    col = col.ravel()
    row = row.ravel()

    if px_is_area:
        # Add 0.5 for pixel is area
        col = col + 0.5
        row = row + 0.5

    # to earth coordinates
    x, y = transform * (col, row)
    # reshape
    x = x.reshape(raster_shape)
    y = y.reshape(raster_shape)
    return x, y


def compare(phi1, phi2):
    ''' Compare two phases by their coherence'''
    return np.abs(np.sum(np.exp(1j * (phi1 - phi2)))) / phi1.size


def wrap(phi):
    ''' Wrap phi to [-pi, pi]'''
    return (phi + np.pi) % (2 * np.pi) - np.pi


def check_input_len(input_var, out_len, error_msg='shape mismatch'):
    """Check that the length of the input array is compatible with a certain\
        desired length. If a scalar is given, create a constant array with\
        desired length."""
    in_var = np.atleast_1d(input_var)
    if len(in_var) == 1:
        return in_var * np.ones(out_len)
    else:
        assert len(in_var) == out_len, error_msg
        return in_var


def filter_list(full_list, ids):
    """
    Filter list and keep only the ids given in the parameters.

    Parameters
    ----------
    full_list : list
        List to be filtered.
    ids : list
        Ids to select elements in the lists.

    Returns
    -------
    list
        Filtered list.

    """
    return list(map(full_list.__getitem__, ids))


def arr_in_interval(arr, arr_min, arr_max):
    """
    Compute mask of values in array that are within interval(inclusive).

    Parameters
    ----------
    arr : np.ndarray
        Array to be checked.
    arr_min : float
        lower value of interval.
    arr_max : float
        upper value of interval.

    Returns
    -------
    ndarray (boolean array)
        Mask of values in interval.

    """
    return np.logical_and(arr >= arr_min, arr <= arr_max)


def stitch_arrays(iter, out_shape, out=None):
    """
    Stitch individual rectangular arrays into an image of known shape, by
    writing the arrays into the given locations.

    Parameters
    ----------
    iter : generator of tuples (array ndarray, write_roi eos.sar.roi.Roi)
        Each `array` is a rectangular array to be used in the mosaic.
        Each `write_roi` indicates where we should write the
        rectangular array in the output image.
    out_shape : tuple
        (h, w) Shape of output image.
    out : ndarray, optional
        Alternative output array in which to place the result.
        It must have the same shape as the expected output,
        but the type of the output values will be cast if necessary.

    Returns
    -------
    out : ndarray
        Output mosaic.
        Equals to the `out` parameter if specified.
        The dtype of the array will be the same as the dtype of the
        first array in the list `iter`.
        Equals to None if the list `iter` is empty.

    """
    for arr, write_roi in iter:
        if out is None:
            out = np.full(out_shape, np.nan, dtype=arr.dtype)

        assert arr.shape == write_roi.get_shape(), "array shape must match write roi shape"
        write_roi.assert_valid(out_shape)
        col_min, row_min, w, h = write_roi.to_roi()
        out[row_min:row_min + h, col_min:col_min + w] = arr

    return out


def write_array(arr, write_roi, out_shape):
    """
    Write an array inside another of shape out_shape using the roi location.

    Parameters
    ----------
    arr : ndarray
        Array to be written.
    write_roi : eos.sar.roi.Roi
        Roi where array needs to be written.
    out_shape : tuple
        Output array shape.

    Returns
    -------
    out_arr : ndarray
        Shifted output array.

    """
    return stitch_arrays([(arr, write_roi)], out_shape)
