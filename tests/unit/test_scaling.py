import numpy as np
import pytest

from cascada_control.scaling import MotionLimiter, OneEuroFilter, deadband, scale_delta


def test_scale_8_to_1_and_per_axis():
    np.testing.assert_allclose(scale_delta([1e-3, 0, -1e-3], 8), [0.125e-3, 0, -0.125e-3])
    np.testing.assert_allclose(scale_delta([1, 1, 1], [8, 4, 2]), [0.125, 0.25, 0.5])
    with pytest.raises(ValueError):
        scale_delta([1, 0, 0], 0)


def test_deadband_holds_then_jumps_without_bias():
    held = np.zeros(3)
    held = deadband([1e-6, 0, 0], held, 2e-6)
    np.testing.assert_array_equal(held, 0)
    held = deadband([3e-6, 0, 0], held, 2e-6)
    np.testing.assert_array_equal(held, [3e-6, 0, 0])


def test_one_euro_converges_to_step():
    f = OneEuroFilter(1.0, 20.0, 1.0)
    f.reset(np.zeros(3))
    for _ in range(500):
        y = f(np.array([1e-3, 0, 0]), 0.01)
    np.testing.assert_allclose(y, [1e-3, 0, 0], atol=1e-12)


def test_limiter_respects_speed_and_accel_and_lands():
    lim = MotionLimiter(0.02, 0.5)
    lim.reset(np.zeros(3))
    dt, prev_v, ps = 0.01, np.zeros(3), []
    for _ in range(300):
        ps.append(lim(np.array([0.01, 0, 0]), dt))
        assert np.linalg.norm(lim.v) <= 0.02 + 1e-12
        assert np.linalg.norm(lim.v - prev_v) <= 0.5 * dt + 1e-12
        prev_v = lim.v.copy()
    np.testing.assert_allclose(ps[-1], [0.01, 0, 0], atol=1e-9)
    assert max(p[0] for p in ps) <= 0.01 + 1e-9  # sin sobrepaso
