import numpy as np
import pytest

from eos.sar import coherence


def test_coherence_dtypes():
    h, w = 50, 100

    u1 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    u2 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    v1 = coherence.on_pair(u1, u2, filter_size=3)
    assert v1.dtype == np.float64

    u1 = u1.astype(np.complex64)
    u2 = u2.astype(np.complex64)
    v2 = coherence.on_pair(u1, u2, filter_size=3)
    assert v2.dtype == np.float32

    assert v1.shape == v2.shape == (h, w)
    assert np.allclose(v1, v2)


def test_coherence_wrong_dtype():
    h, w = 50, 100
    u1 = np.random.randn(h, w)
    u2 = np.random.randn(h, w)
    with pytest.raises(AssertionError):
        coherence.on_pair(u1, u2, filter_size=3)


def test_coherence_filter_size():
    h, w = 20, 30

    u1 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    u2 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()

    v1 = coherence.on_pair(u1, u2, filter_size=3)
    v2 = coherence.on_pair(u1, u2, filter_size=(3, 3))
    assert np.allclose(v1, v2)

    v1 = coherence.on_pair(u1, u2, filter_size=(1, 5))
    v2 = coherence.on_pair(u1, u2, filter_size=(5, 1))
    # the vertical TV should be higher than the horizontal TV if the filter was horizontal
    assert np.abs(np.diff(v1, axis=0)).sum() > np.abs(np.diff(v1, axis=1)).sum()
    # the vertical TV should be lower than the horizontal TV if the filter was horizontal
    assert np.abs(np.diff(v2, axis=0)).sum() < np.abs(np.diff(v2, axis=1)).sum()


def test_coherence_wrong_filter_size():
    h, w = 20, 30
    u1 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    u2 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    with pytest.raises(AssertionError):
        coherence.on_pair(u1, u2, filter_size=4)
    with pytest.raises(AssertionError):
        coherence.on_pair(u1, u2, filter_size=(5, 3, 5))


def test_coherence_might_contain_nans_without_nan():
    h, w = 20, 30

    u1 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    u2 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    v1 = coherence.on_pair(u1, u2, filter_size=3, might_contain_nans=False)
    v2 = coherence.on_pair(u1, u2, filter_size=3, might_contain_nans=True)
    assert np.allclose(v1, v2)

    assert np.isnan(v1).sum() == 0
    assert np.isnan(v2).sum() == 0


def test_coherence_might_contain_nans_with_nan():
    h, w = 20, 30

    u1 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    u2 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    u1[2, 3] = np.nan
    v1 = coherence.on_pair(u1, u2, filter_size=3, might_contain_nans=False)
    v2 = coherence.on_pair(u1, u2, filter_size=3, might_contain_nans=True)

    # if might_contain_nans == 'overwrite', then u1 will be modified
    assert np.isnan(u1).sum() == 1
    v3 = coherence.on_pair(u1, u2, filter_size=3, might_contain_nans="overwrite")
    assert np.isnan(u1).sum() == 0

    assert not np.allclose(v1, v2, equal_nan=True)
    assert np.allclose(v2, v3, equal_nan=True)

    assert np.isnan(v1).sum() >= 2
    assert np.isnan(v2).sum() == 1
    assert np.isnan(v3).sum() == 1


def test_coherence_set_borders_to_nan():
    h, w = 20, 30

    u1 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    u2 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()

    v1 = coherence.on_pair(u1, u2, filter_size=3, set_borders_to_nan=False)
    v2 = coherence.on_pair(u1, u2, filter_size=3, set_borders_to_nan=True)

    assert not np.allclose(v1, v2, equal_nan=True)
    assert np.isnan(v1).sum() == 0
    assert np.isnan(v2).sum() == v1.size - v1[1:-1, 1:-1].size
    assert np.allclose(v1[1:-1, 1:-1], v2[1:-1, 1:-1])


def test_coherence_spatial_filter():
    h, w = 20, 30

    u1 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()
    u2 = np.random.randn(h, w, 2).view(np.cdouble).squeeze()

    # just check that it doesn't crash, at least
    # spatial_filter='gaussian' is not really meant to be used anyway
    coherence.on_pair(u1, u2, filter_size=3, spatial_filter="uniform")
    coherence.on_pair(u1, u2, filter_size=3, spatial_filter="gaussian")
