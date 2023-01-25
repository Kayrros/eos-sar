import pytest
import numpy as np
from eos.sar import goldstein_filter


@pytest.mark.parametrize('fft_size', (16, 32))
@pytest.mark.parametrize('window_size', (3, 4, 5))
@pytest.mark.parametrize('alpha', (0, 0.5, 1))
@pytest.mark.parametrize('nworkers', (1, 4))
def test_goldstein(fft_size, window_size, alpha, nworkers):
    h = w = 256
    ifg = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    filtered = goldstein_filter.apply(
        ifg, fft_size, window_size, alpha, nworkers)
    assert filtered.dtype == np.complex128

    ifg = ifg.astype(np.complex64)
    filtered = goldstein_filter.apply(
        ifg, fft_size, window_size, alpha, nworkers)

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
            ifg, fft_size=32, window_size=33)
