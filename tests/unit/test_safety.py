import numpy as np

from cascada_control import safety


def test_watchdog_rule():
    assert not safety.watchdog(0.0, None, 0.05).allowed
    assert safety.watchdog(1.0499, 1.0, 0.05).allowed
    assert not safety.watchdog(1.0501, 1.0, 0.05).allowed


def test_workspace_reject_and_clamp():
    lo, hi = [-1, -1, 0], [1, 1, 1]
    assert safety.workspace([0, 0, 0.5], lo, hi).allowed
    d = safety.workspace([0, 0, 1.5], lo, hi, "reject")
    assert not d.allowed and "fuera" in d.reason
    d = safety.workspace([0, 2, -1], lo, hi, "clamp")
    assert d.allowed and d.reason
    np.testing.assert_array_equal(d.value, [0, 1, 0])


def test_joint_limits():
    assert safety.joint_limits([0, 0], np.array([-1, -1]), np.array([1, 1])).allowed
    assert not safety.joint_limits([0, 1.1], np.array([-1, -1]), np.array([1, 1])).allowed
