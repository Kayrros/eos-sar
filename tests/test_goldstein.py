import pytest
import numpy as np
from eos.sar import goldstein_filter


@pytest.mark.parametrize('step', (4, 8))
@pytest.mark.parametrize('window_size', (3, 4, 5))
@pytest.mark.parametrize('alpha', (0, 0.5, 1))
@pytest.mark.parametrize('nworkers', (1, 4))
def test_goldstein(step, window_size, alpha, nworkers):
    h = w = 256
    ifg = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    filtered = goldstein_filter.apply(
        ifg, step, window_size, alpha, nworkers)
    assert filtered.dtype == np.complex128

    ifg = ifg.astype(np.complex64)
    filtered = goldstein_filter.apply(
        ifg, step, window_size, alpha, nworkers)

    assert filtered.dtype == np.complex64


def test_goldstein_wrong_img_type():
    h = w = 256
    ifg = np.random.randn(h, w)
    with pytest.raises(AssertionError):
        goldstein_filter.apply(
            ifg)


@pytest.mark.parametrize('alpha', (-1.1, 2))
def test_goldstein_wrong_alpha_bound(alpha):
    h = w = 256
    ifg = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    with pytest.raises(AssertionError):
        goldstein_filter.apply(
            ifg, alpha=alpha)


def test_goldstein_wrong_win_size():
    h = w = 256
    ifg = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    with pytest.raises(AssertionError):
        goldstein_filter.apply(
            ifg, step=8, window_size=33)


def apply_tri_by_patches_and_recombine(img, step):
    img, orig_roi = goldstein_filter.pad_img(img, step)

    out_image = np.zeros(img.shape, dtype=img.dtype)

    patch_roi_generator = goldstein_filter.extract_patch_rois(img.shape, step)
    filt_triangle = goldstein_filter.triangular_filter(step)

    for roi in patch_roi_generator:
        col, row, w, h = roi.to_roi()
        out_image[row: row + h, col:col + w] += filt_triangle * roi.crop_array(img)

    return orig_roi.crop_array(out_image)


def test_border_and_normalization():
    size = (256, 256)
    ones_array = np.ones(size, dtype=np.complex64)
    step = 8
    filtered = goldstein_filter.apply(ones_array, step)
    filtered_phase = np.angle(filtered)
    filtered_amp = np.abs(filtered)

    border = 3 * step

    np.testing.assert_equal(filtered_phase[border:-border, border:-border], 0)
    np.testing.assert_equal(filtered_amp[border:-border, border:-border], 1)

    # check that our border strategy is favorable for triangular filters
    triangular_filtered = apply_tri_by_patches_and_recombine(ones_array, step)
    np.testing.assert_equal(np.angle(triangular_filtered), 0)
    np.testing.assert_equal(np.abs(triangular_filtered), 1)


def test_nodata():
    size = (256, 256)

    ones_array = np.ones(size, dtype=np.complex64)

    nan_mask = np.zeros_like(ones_array, dtype=bool)
    # Full nan patch on the border
    nan_mask[:32, :32] = True
    # a nan line
    nan_mask[50:70, 60] = True

    ones_array[nan_mask] = np.nan

    filtered = goldstein_filter.apply(ones_array)

    assert np.all(np.isnan(filtered) == nan_mask), "NaNs did not propagate"
    assert np.all(filtered[120:130, 120:130] == 1), "Away from NaNs not affected"
