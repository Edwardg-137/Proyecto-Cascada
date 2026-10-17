import numpy as np

from cascada_control.kinematics import fk
from cascada_control.teleop import ArmTeleop, PoseSample
from tests.conftest import HOME_Q

I = np.eye(3)
P0 = np.array([0.30, 0.10, 0.05])  # pose arbitraria de la herramienta modelo


def run(arm, poses, t0=0.0, clutch=True):
    dt, q = arm.cfg.period_s, None
    for k, p in enumerate(poses):
        q = arm.step(t0 + k * dt, PoseSample(np.asarray(p), I), clutch)
    return q, t0 + len(poses) * dt


def test_1mm_model_is_0125mm_follower_full_chain(left):
    """FK → escala → filtro/límites → IK → FK: 1 mm modelo = 0.125 mm seguidor (±1 µm)."""
    chain, cfg, _ = left
    for axis in range(3):
        arm = ArmTeleop(chain, cfg, HOME_Q)
        p_start = fk(chain, HOME_Q)[0]
        step = np.zeros(3)
        step[axis] = 1e-3
        q, _ = run(arm, [P0] + [P0 + step] * 300)
        moved = fk(chain, q)[0] - p_start
        assert arm.status == "transmitiendo"
        assert np.linalg.norm(moved - step / 8) < 1e-6, (axis, moved)


def test_clutch_release_and_reengage_no_jump(left):
    chain, cfg, _ = left
    arm = ArmTeleop(chain, cfg, HOME_Q)
    q, t = run(arm, [P0] + [P0 + [2e-3, 0, 0]] * 200)
    p_before = fk(chain, q)[0]
    # Embrague suelto: el operador mueve el modelo 40 mm; el seguidor no se mueve.
    q_rel, t = run(arm, [P0 + [42e-3, 0, 0]] * 50, t, clutch=False)
    np.testing.assert_array_equal(q_rel, q)
    # Re-pisar: re-indexa en la nueva pose; primer ciclo sin salto.
    q_re, t = run(arm, [P0 + [42e-3, 0, 0]] * 5, t)
    assert np.linalg.norm(fk(chain, q_re)[0] - p_before) < 1e-9
    # Y después sigue en relativo: +1 mm modelo = +0.125 mm desde ahí.
    q_end, _ = run(arm, [P0 + [43e-3, 0, 0]] * 300, t)
    assert np.linalg.norm(fk(chain, q_end)[0] - p_before - [0.125e-3, 0, 0]) < 1e-6


def test_watchdog_trips_in_configured_time(left):
    chain, cfg, _ = left
    arm = ArmTeleop(chain, cfg, HOME_Q)
    q, t = run(arm, [P0] * 10)
    t_last = t - cfg.period_s  # instante de la última muestra
    timeout = cfg.watchdog_timeout_s
    assert timeout == 0.050
    arm.step(t_last + timeout - 1e-4, None, True)
    assert arm.status == "transmitiendo"
    q_trip = arm.step(t_last + timeout + 1e-4, None, True)
    assert arm.status.startswith("watchdog")
    np.testing.assert_allclose(q_trip, q, atol=1e-12)
    # Vuelven las muestras con el embrague pisado: sigue detenido hasta soltarlo.
    arm.step(t_last + 0.2, PoseSample(P0 + [5e-3, 0, 0], I), True)
    assert not arm.engaged
    arm.step(t_last + 0.21, PoseSample(P0 + [5e-3, 0, 0], I), False)
    q_re = arm.step(t_last + 0.22, PoseSample(P0 + [5e-3, 0, 0], I), True)
    assert arm.engaged
    np.testing.assert_allclose(q_re, q, atol=1e-12)  # re-indexa sin salto


def test_workspace_reject_holds_inside(left):
    chain, cfg, _ = left
    arm = ArmTeleop(chain, cfg, HOME_Q)
    y_min = cfg.workspace_lower[1]
    # Pedir -0.2 m en y al seguidor (1.6 m del modelo), más allá del límite de la caja.
    q, _ = run(arm, [P0] + [P0 + [0, -1.6, 0]] * 600)
    p = fk(chain, q)[0]
    assert "fuera" in arm.status
    assert p[1] >= y_min - 1e-9
    assert p[1] - y_min < 1e-3  # avanzó hasta cerca del límite y se detuvo


def test_workspace_clamp_mode(left):
    from dataclasses import replace

    chain, cfg, _ = left
    arm = ArmTeleop(chain, replace(cfg, workspace_mode="clamp"), HOME_Q)
    q, _ = run(arm, [P0] + [P0 + [0, -1.6, 0]] * 600)
    p = fk(chain, q)[0]
    assert abs(p[1] - cfg.workspace_lower[1]) < 1e-6
    assert "recortado" in arm.status


def test_orientation_copied_1_to_1_not_scaled(left):
    """Girar el modelo 2° en torno al eje de la muñeca gira la herramienta 2° (no 0.25°)."""
    from cascada_control.kinematics import axis_angle_R, rot_log

    chain, cfg, _ = left
    arm = ArmTeleop(chain, cfg, HOME_Q)
    p0, R0 = fk(chain, HOME_Q)
    _, _, _, Jo = fk(chain, HOME_Q, jacobian=True)
    axis = Jo[:, 4]  # eje de Wrist_Roll: rotación alcanzable con 5 ejes
    Rm = axis_angle_R(axis, np.deg2rad(2))
    arm.step(0.0, PoseSample(P0, I), True)
    for k in range(1, 50):
        q = arm.step(k * cfg.period_s, PoseSample(P0, Rm), True)
    p, R = fk(chain, q)
    assert np.linalg.norm(p - p0) < 1e-6
    np.testing.assert_allclose(np.rad2deg(np.linalg.norm(rot_log(R @ R0.T))), 2.0, atol=1e-6)
