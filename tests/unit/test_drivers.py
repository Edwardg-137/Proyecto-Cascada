import numpy as np
import pytest

from cascada_control.drivers import BusError, Calibration, MockBus, counts_to_rad, rad_to_counts


def test_counts_round_trip_within_one_count():
    cal = Calibration((1, 2), (2048, 1000), (1, -1))
    q = np.array([0.5, -0.25])
    q2 = counts_to_rad(rad_to_counts(q, cal), cal)
    assert np.all(np.abs(q2 - q) <= np.pi / 4096)  # ≤ media cuenta
    assert rad_to_counts([0, 0], cal) == {1: 2048, 2: 1000}


def test_mock_bus_failure_raises():
    bus = MockBus({1: 2048})
    bus.fail = True
    with pytest.raises(BusError):
        bus.sync_read_positions([1])
