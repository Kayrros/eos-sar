import numpy as np
from scipy.optimize import least_squares
from eos.sar import regist


def max_2d(array):
    """
    Return the coordinates of the maximum of a 2D numpy array.

    Parameters
    ----------
    array : np.2darray
        Input array where we wish to find the maximum.

    Returns
    -------
    tuple of ints
        (row, col) indices of the maximum.

    """
    return np.unravel_index(np.argmax(array), array.shape)


def get_local_maxima(array, *, sort=True):
    """
    Find all local maxima within an array of pixels.
    Local maxima are pixels whose intensity is greater than all of their neighbors.
    This excludes points on the edge of the vector.

    Parameters
    ----------
    array : np.2darray
        Section of the image to search.
    sort : bool
        If True, the results are sorted in descending order of intensities.
    Returns
    -------
    list[tuple[tuple, float]]
        The local maximas that were found. Each element of the list is a tuple containing:
            tuple: the (row, col)-coords of the maximum.
            float: the intensity of the pixel.
        the list is sorted by intensity (decreasing) if sort is True.

    """
    out = []
    for i in range(1, array.shape[0] - 1):
        for j in range(1, array.shape[1] - 1):
            window = array[i - 1:i + 2, j - 1:j + 2]
            if max_2d(window) == (1, 1):
                out.append(((i, j), window[1, 1]))
    if sort:
        return sorted(out, key=lambda x: x[1], reverse=True)
    else:
        return out


def interpolate_window(image):
    """
    Performs quadratic interpolation to find the sub-pixel maximum of an
    image.

    Parameters
    ----------
    image : np.2darray
        Section of the image to interpolate.

    Returns
    -------
    tuple(float, float)
        row, col coordinate of the interpolated maximum.
    float
        Interpolated intensity of the maximum.

    """
    h, w = image.shape

    assert h > 2 and w > 2, "Image too small"

    # Fit a bivariate second-order polynomial to the data

    rows = np.arange(image.shape[0])
    cols = np.arange(image.shape[1])

    def parse_coefs(c):
        '''Parse bivariate second-order polynomial coefficients into a
        matrix which can be passed to np.polynomial.polynomial functions.
        Args:
            c (list): Coefficients to parse.
                c must be of length 6 of the form: [A, B, C, D, E, F] where
                P = Ax**2 + By**2 + Cx + Dy + Exy + F
        '''
        A, B, C, D, E, F = c
        return np.array([F, D, B, C, E, 0, A, 0, 0]).reshape(3, 3)

    def objective_function(coefs):
        return (np.polynomial.polynomial.polygrid2d(rows, cols, parse_coefs(coefs))
                - image).ravel()

    c = least_squares(objective_function, [1, 1, 1, 1, 1, 1], method='lm')

    assert c.success, "Polynomial fitting failed"

    c = c.x
    A, B, C, D, E, F = c

    # Now that the polynomial has been fit, we compute its maximum.
    # closed form equations from setting the gradient to 0
    denum = 4 * A * B - E**2

    assert denum, "Not possible to find maxium with polynomial"

    row_max = (D * E - 2 * B * C) / denum
    col_max = (C * E - 2 * A * D) / denum
    intensity = np.polynomial.polynomial.polyval2d(row_max, col_max, parse_coefs(c))

    return (row_max, col_max), intensity


def sub_pixel_maxima(zoomed_image, search_roi_in_original_image,
                     zoom_factor=1):
    """
    Finds all local maxima in a rectangular section of an image and calculate
    their sub-pixel coordinates. The returned values are in the orginal (not zoomed)
    coordinate system.

    Parameters
    ----------
    zoomed_image : np.ndarray
        Zoomed image from which to take a crop for the search.
    search_roi_in_original_image : eos.sar.roi.Roi
        Search region in the original image (before zoom), all local maximas in this
        region will be found.
    zoom_factor : float, optional
        Zoom factor that was applied on the image. The default is 1.

    Returns
    -------
    row_maxima : np.1darray
        Subpixel row locations of the local maximas, in the original (not zoomed) image
        coordinate system.
    col_maxima : np.1darray
        Subpixel col locations of the local maximas, in the original (not zoomed) image
        coordinate system.
    intensities : np.1darray
        Intensity at the maximum locations, sorted by descending order (i.e. most to least significant max).

    """
    search_roi = regist.zoom_roi(search_roi_in_original_image, zoom_factor)
    search_roi_orig = search_roi.get_origin()
    search_array = search_roi.crop_array(zoomed_image)
    # maxima are in search_array coordinates (i.e. the crop of the zoomed image)
    maxima = get_local_maxima(search_array, sort=False)

    result = []
    for maximum, _ in maxima:
        # crop again around each local maximum and do quadratic max finding
        window = search_array[
            maximum[0] - 1:maximum[0] + 2,
            maximum[1] - 1:maximum[1] + 2
        ]

        # row_max and col_max are subpixel maximas in the (3, 3) window coord system
        (row_max, col_max), intensity = interpolate_window(window)

        # Now that we have the coords of the max relative to the subwindow "window", we have to get them back in image coordinates.
        # First we go from "window" coordinates to "search_array" coordinates, then from "search_array" to "zoomed_image".

        # First step
        row_max = row_max + maximum[0] - 1
        col_max = col_max + maximum[1] - 1

        # Second step
        col_max = (col_max + search_roi_orig[0]) / zoom_factor
        row_max = (row_max + search_roi_orig[1]) / zoom_factor

        result.append(((row_max, col_max), intensity))

    # at this stage, we sort
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return result
