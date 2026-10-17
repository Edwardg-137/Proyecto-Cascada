import numpy as np
import pytest

from cascada_control.drivers import MockBus, rad_to_counts
from cascada_control.kinematics import fk
from cascada_control.teleop import ArmTeleop, PoseSample, ScriptedPoseSource, run_cycle
from tests.conftest import HOME_Q

I = np.eye(3)
P0 = np.zeros(3)


def test_both_arms_configured(cfg):
    from cascada_control.teleop import arm_setup
    assert set(cfg["arms"]) == {"left", "right"}
    for name in cfg["arms"]:
        arm_setup(cfg, name)


def test_scripted_loop_on_mock_bus_with_dropout(left):
    chain, tc, cal = left
    arm = ArmTeleop(chain, tc, HOME_Q)
    bus = MockBus(rad_to_counts(HOME_Q, cal))
    script = [PoseSample(P0 + [k * 1e-5, 0, 0], I) for k in range(100)] + [None] * 10
    src = ScriptedPoseSource(script)
    dt = tc.period_s
    for k in range(110):
        run_cycle(k * dt, src, True, arm, bus, cal)
    assert arm.status.startswith("watchdog")
    assert len(bus.writes) == 110
    assert bus.writes[-1] == bus.writes[-2]  # metas congeladas tras el disparo
    assert bus.positions == rad_to_counts(arm.q, cal)
    # El modelo recorrió 0.99 mm en x → consigna ≈ 0.124 mm (con el retardo del filtro).
    moved = fk(chain, arm.q)[0] - fk(chain, HOME_Q)[0]
    assert 0.08e-3 < moved[0] <= 0.99e-3 / 8 + 1e-6
    assert np.linalg.norm(moved[1:]) < 1e-6


def test_bus_error_is_safety_event(left):
    chain, tc, cal = left
    arm = ArmTeleop(chain, tc, HOME_Q)
    bus = MockBus(rad_to_counts(HOME_Q, cal))
    src = ScriptedPoseSource([PoseSample(P0, I)] * 5)
    run_cycle(0.0, src, True, arm, bus, cal)
    bus.fail = True
    assert run_cycle(0.01, src, True, arm, bus, cal) is None
    assert arm.status.startswith("bus") and not arm.engaged


def test_fk_matches_mujoco(left):
    mujoco = pytest.importorskip("mujoco")
    from cascada_control.sim import load_model

    chain = left[0]
    model = load_model()
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.uniform(chain.lower, chain.upper)
        data.qpos[:5] = q
        mujoco.mj_kinematics(model, data)
        R_mj = data.xmat[bid].reshape(3, 3)
        p_mj = data.xpos[bid] + R_mj @ chain.tcp
        p, R = fk(chain, q)
        np.testing.assert_allclose(p, p_mj, atol=1e-9)
        np.testing.assert_allclose(R, R_mj, atol=1e-9)


def test_teleop_on_mujoco_bus(left):
    pytest.importorskip("mujoco")
    from cascada_control.sim import MujocoBus

    chain, tc, cal = left
    arm = ArmTeleop(chain, tc, HOME_Q)
    bus = MujocoBus(cal, HOME_Q)
    script = [PoseSample(P0, I)] + [PoseSample(P0 + [8e-3, 0, 0], I)] * 300
    src = ScriptedPoseSource(script)
    for k in range(len(script)):
        q_meas = run_cycle(k * tc.period_s, src, True, arm, bus, cal)
    assert arm.status == "transmitiendo"
    # Consigna: +1 mm en x. Lo que se mide en sim incluye el modelo de actuador
    # (P con kp=50, gravedad) y la cuantización del encoder: solo se comprueba que sigue.
    p_cmd = fk(chain, arm.q)[0] - fk(chain, HOME_Q)[0]
    np.testing.assert_allclose(p_cmd, [1e-3, 0, 0], atol=1e-6)
    assert np.all(np.isfinite(q_meas))
