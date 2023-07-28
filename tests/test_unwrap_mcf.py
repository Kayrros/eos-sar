import pytest
import numpy as np
from eos.sar import unwrapping
from eos.sar.utils import wrap


@pytest.mark.parametrize("dtype,atol", [(np.float32, 1e-4), (np.float64, 1e-10)])
def test_unwrap(dtype, atol):
    h, w = 50, 100

    # simulate a horizontal phase ramp, with a slope a bit less then pi
    slope_x = 2 * np.pi / 3
    slope_y = 2 * np.pi / 3

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    phase = (xx * slope_x + yy * slope_y).astype(dtype)

    # at this stage, no gradient is more then pi, so the unwrapping should be perfect
    assert np.all(np.abs(np.diff(phase, axis=0)) < np.pi)
    assert np.all(np.abs(np.diff(phase, axis=1)) < np.pi)

    wrapped = wrap(phase)

    unwrapped = unwrapping.mcf(wrapped)

    assert unwrapped.dtype == dtype

    error = unwrapped - phase
    # unwrapping up to a constant
    bias = np.median(error)
    error -= bias
    unwrapped -= bias

    np.testing.assert_allclose(error, 0, atol=atol)

    noise = np.random.normal(scale=np.pi / 12, size=phase.shape).astype(dtype)
    noisy_phase = phase + noise
    # should give between 0.1 and 0.4 % of high gradients (empirically)
    perctg = np.mean(np.abs(np.diff(noisy_phase, axis=0)) > np.pi) * 100
    assert perctg > 0
    perctg = np.mean(np.abs(np.diff(noisy_phase, axis=1)) > np.pi) * 100
    assert perctg > 0

    wrapped = wrap(noisy_phase)

    unwrapped = unwrapping.mcf(wrapped)

    assert unwrapped.dtype == dtype

    error = unwrapped - noisy_phase
    # unwrapping up to a constant
    bias = np.median(error)
    error -= bias
    unwrapped -= bias
    assert np.mean(np.abs(error) > atol) < 0.01, "more than 1% of px affected by error"
