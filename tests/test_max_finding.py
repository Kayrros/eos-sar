import numpy as np
import pytest

from eos.sar import max_finding, roi


def test_max2d():
    """Test the discrete array max finding (simplest max finding, uses numpy argmax)."""
    im_size = 50
    array = np.zeros((im_size, im_size))
    r_max = np.random.randint(0, im_size)
    c_max = np.random.randint(0, im_size)
    array[r_max, c_max] = 1
    row_max, col_max = max_finding.max_2d(array)
    assert row_max == r_max
    assert col_max == c_max


def get_paraboloid(h, w):
    """
    Get a random simulated paraboloid of specified shape
    that attains the maximum within the returned image.
    The max coordinates are also returned.

    Parameters
    ----------
    h : int
        height of image to simulate.
    w : int
        width of image to simulate.

    Returns
    -------
    simulated : np.ndarray
        Simulated paraboloid image
    coeffs : np.ndarray (3, 3)
        Coefficients matrix of the 2d polynomial, compatible with np.polynomial.polynomial.
    mr : float
        Subpixel row position of the max.
    mc : float
        Subpixel col position of the max.

    """
    # ground truth maxima locations
    mr = np.random.uniform(h // 2 - 0.5, h // 2 + 0.5)
    mc = np.random.uniform(w // 2 - 0.5, w // 2 + 0.5)
    # ground truth poly coeffs
    coeffs = - np.array([[mc**2 + mr**2, -2 * mc, 1],
                         [-2 * mr, 0, 0], [1, 0, 0]])
    # generate polynomial image
    simulated = np.polynomial.polynomial.polygrid2d(np.arange(h), np.arange(w),
                                                    coeffs)
    min_simulated = np.amin(simulated)
    # translate so the min is 0
    coeffs[0, 0] = coeffs[0, 0] - min_simulated
    simulated -= min_simulated

    return simulated, coeffs, mr, mc


image_shapes = ((3, 3),
                (30, 50),
                (300, 500)
                )


@pytest.mark.parametrize("image_shape", image_shapes)
def test_max_finding(image_shape):
    h, w = image_shape

    gt_image, gt_coeffs, mr, mc = get_paraboloid(h, w)

    max_row, max_col, intensity = max_finding.interpolate_window(gt_image)

    np.testing.assert_allclose(mr, max_row, atol=1e-3)
    np.testing.assert_allclose(mc, max_col, atol=1e-3)


def get_gaussian(h, w):
    """
    Simulate a spatial guassian

    Parameters
    ----------
    h : int
        height of image to simulate.
    w : int
        width of image to simulate.

    Returns
    -------
    simulated : np.ndarray
        Simuated image.
    intensity : float
        Max value reached by the gaussian.
    mr : float
        Subpix row position of max.
    mc : float
        Subpix col position of max.

    """
    # ground truth maxima locations
    mr = np.random.uniform(h // 2 - 0.5, h // 2 + 0.5)
    mc = np.random.uniform(w // 2 - 0.5, w // 2 + 0.5)

    Cols, Rows = np.meshgrid(np.arange(w), np.arange(h))

    sig = min(h // 5, w // 5)

    dist_squared = (Cols - mc) ** 2 + (Rows - mr) ** 2
    intensity = np.random.uniform(1, 2)

    simulated = intensity * np.exp(- dist_squared / sig ** 2)

    return simulated, intensity, mr, mc


def simulate_set_of_maximas(h, w, step=3, p=0.1):
    """
    Simulate set of maximas. The image is gridded with step and the probability
    that a simulation is done within a grid cell is controlled by p.

    Parameters
    ----------
    h : int
        height of image to simulate.
    w : int
        width of image to simulate.
    step : int, optional
        Grid step size. The default is 3.
    p : float, optional
        Probability of simulation within a cell.
        Controls the number of simulated cells. The default is 0.1.

    Returns
    -------
    simulated_image : np.ndarray
        Simulated image.
    mrs : np.ndarray
        Supbix row positions of the maximas.
    mcs : np.ndarray
        Subpix col positions of the maximas.
    intensities : np.ndarray
        Max intensities reached by the peaks.

    """
    # this way ensure non overlapping paraboloids
    candidate_rows = np.arange(0, h - step - 1, step)
    candidate_cols = np.arange(0, w - step - 1, step)
    Candidate_cols, Candidate_rows = np.meshgrid(
        candidate_cols, candidate_rows)
    mask = 0
    while not np.any(mask):
        # at least one location validated for simulation
        mask = np.random.binomial(
            1, p=p, size=Candidate_cols.shape).astype(bool)

    cols_maximas = Candidate_cols[mask]
    rows_maximas = Candidate_rows[mask]

    n_maximas = len(cols_maximas)

    mrs = np.zeros(n_maximas)
    mcs = np.zeros(n_maximas)
    intensities = np.zeros(n_maximas)
    simulated_image = np.zeros((h, w))
    for ids, (col_max, row_max) in enumerate(zip(cols_maximas, rows_maximas)):
        para, intensity, mr, mc = get_gaussian(step // 2, step // 2)
        mrs[ids] = mr + row_max
        mcs[ids] = mc + col_max
        intensities[ids] = intensity
        simulated_image[row_max:row_max + step //
                        2, col_max:col_max + step // 2] += para

    # sort by descending intensities
    sorting_indices = np.argsort(intensities)[::-1]

    return simulated_image, mrs[sorting_indices], mcs[sorting_indices], intensities[sorting_indices]


def test_sub_pixel_max():
    """Test the finding of a set of subpixel maximas within an image."""
    # simulate zoomed image where some are contains parabolas (actually we use gaussians instead)
    image_size = 256
    zoom_factor = 2
    image_zoomed = np.zeros(
        (image_size * zoom_factor, image_size * zoom_factor))
    grid_size = image_size // 10
    area_with_maximas = roi.Roi(grid_size,
                                grid_size,
                                image_size - 2 * grid_size,
                                image_size - 2 * grid_size)
    col, row, w, h = area_with_maximas.to_roi()
    # define local paraboloid maximas within this area
    simulated_image, mrs, mcs, intensities_simu = simulate_set_of_maximas(
        h * zoom_factor, w * zoom_factor, step=grid_size * zoom_factor, p=0.1 / zoom_factor)

    mrs += row * zoom_factor
    mcs += col * zoom_factor

    image_zoomed[row * zoom_factor:(row + h) * zoom_factor,
                 col * zoom_factor:(col + w) * zoom_factor] += simulated_image

    # now test subpixel maxima
    row_maxima, col_maxima, intensities = max_finding.sub_pixel_maxima(
        image_zoomed,
        area_with_maximas,
        zoom_factor)

    assert len(mrs) == len(row_maxima)
    np.testing.assert_allclose(row_maxima, mrs / zoom_factor,
                               atol=1e-2, verbose=True)
    np.testing.assert_allclose(col_maxima, mcs / zoom_factor,
                               atol=1e-2, verbose=True)
    np.testing.assert_allclose(intensities, intensities_simu,
                               atol=1e-2, verbose=True)
