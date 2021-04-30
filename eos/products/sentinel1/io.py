import rasterio

# read the burst arrays


def read_burst(tiff_path, burst_roi):
    '''
    

    Parameters
    ----------
    tiff_path : str
        Path to the tiff of the burst subswath.
    burst_roi : tuple
        (x,y,w,h) location of the burst in the tiff file.

    Returns
    -------
    burst_array : ndarray
        burst np.complex64 image.

    '''
    x, y, w, h = burst_roi
    with rasterio.open(tiff_path) as db:
        burst_array = db.read(1, window=(
            (y, y+h), (x, x+w))).astype('complex64')
    return burst_array
