import rasterio

def read_window(tiff_path, roi):
    """Read window inside the tiff of a complex image.

    Parameters
    ----------
    tiff_path : str
        Path to the tiff.
    roi : tuple
        (x,y,w,h) location to read from in the tiff file.

    Returns
    -------
    array : ndarray
        np.complex64 image.

    """
    x, y, w, h = roi
    with rasterio.open(tiff_path) as db:
        array = db.read(1, window=(
            (y, y+h), (x, x+w))).astype('complex64')
    return array


def read_windows(tiff_path, rois):
    """Read windows inside the tiff of a complex image.

    Parameters
    ----------
    tiff_path : str
        Path to the tiff.
    rois : list of tuples
        (x,y,w,h) location to read from in the tiff file.

    Returns
    -------
    array : ndarray
        np.complex64 image.

    """
    arrays = []
    with rasterio.open(tiff_path) as db:
        for roi in rois: 
            x, y, w, h = roi
            arrays.append(db.read(1, window=(
                (y, y+h), (x, x+w))).astype('complex64'))
    return arrays