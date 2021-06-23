"""IO for the burst array."""
import rasterio

# read the burst arrays


def read_window(tiff_path, roi):
    """Read window inside the tiff.

    Parameters
    ----------
    tiff_path : str
        Path to the tiff of the subswath.
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

        