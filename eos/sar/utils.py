import numpy as np 


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

