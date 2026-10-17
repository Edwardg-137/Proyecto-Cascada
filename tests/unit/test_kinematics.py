import numpy as np

from cascada_control.kinematics import axis_angle_R, fk, ik, rot_log
from tests.conftest import HOME_Q


def test_chain_from_mjcf(left):
    chain = left[0]
    assert chain.names == ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
    np.testing.assert_allclose(chain.pos[0], [0, -0.0452, 0.0165])  # Rotation_Pitch pos del MJCF


def test_jacobian_matches_finite_difference(left):
    chain = left[0]
    p, R, Jp, Jo = fk(chain, HOME_Q, jacobian=True)
    h = 1e-7
    for i in range(chain.n):
        dq = np.zeros(chain.n)
        dq[i] = h
        p2, R2 = fk(chain, HOME_Q + dq)
        np.testing.assert_allclose((p2 - p) / h, Jp[:, i], atol=1e-6)
        np.testing.assert_allclose(rot_log(R2 @ R.T) / h, Jo[:, i], atol=1e-6)


def test_ik_round_trip(left):
    chain = left[0]
    rng = np.random.default_rng(0)
    mid, half = (chain.upper + chain.lower) / 2, (chain.upper - chain.lower) / 2
    for _ in range(50):
        q_true = mid + 0.7 * half * rng.uniform(-1, 1, chain.n)
        p, R = fk(chain, q_true)
        q0 = np.clip(q_true + rng.uniform(-0.1, 0.1, chain.n), chain.lower, chain.upper)
        q, ok = ik(chain, p, R, q0)
        p2, R2 = fk(chain, q)
        assert ok
        assert np.linalg.norm(p2 - p) < 1e-6           # 1 µm
        assert np.linalg.norm(rot_log(R2 @ R.T)) < 1e-6  # rad (pose alcanzable)


def test_rot_log_inverts_axis_angle():
    for v in ([0.3, -0.2, 0.1], [0, 0, np.pi - 1e-4], [1e-12, 0, 0]):
        v = np.array(v)
        R = axis_angle_R(v / np.linalg.norm(v), np.linalg.norm(v))
        np.testing.assert_allclose(rot_log(R), v, atol=1e-9)
