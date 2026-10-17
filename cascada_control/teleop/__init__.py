"""Bucle de teleoperación: pose de la herramienta modelo → objetivos articulares."""
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import yaml

from cascada_control import kinematics as kin
from cascada_control import safety
from cascada_control.drivers import Bus, BusError, Calibration, counts_to_rad, rad_to_counts
from cascada_control.scaling import MotionLimiter, OneEuroFilter, deadband, scale_delta


class PoseSample(NamedTuple):
    p: np.ndarray  # posición (m) en el marco de la herramienta modelo
    R: np.ndarray  # orientación 3×3


class PoseSource(Protocol):
    """Fuente de pose de la herramienta modelo. Método de medición: POR DEFINIR."""

    def read(self) -> PoseSample | None: ...


class ScriptedPoseSource:
    """Devuelve las muestras dadas en orden; None = sin muestra en ese ciclo."""

    def __init__(self, samples):
        self._it = iter(samples)

    def read(self):
        return next(self._it, None)


@dataclass(frozen=True)
class TeleopConfig:
    period_s: float
    scale: object
    orientation: str
    deadband_m: float
    one_euro: dict
    max_speed_m_s: float
    max_accel_m_s2: float
    watchdog_timeout_s: float
    workspace_lower: np.ndarray
    workspace_upper: np.ndarray
    workspace_mode: str
    model_to_follower: np.ndarray


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def arm_setup(cfg, arm):
    """(Chain, TeleopConfig, Calibration) para un brazo de config/teleop.yaml."""
    t, a, m = cfg["teleop"], cfg["arms"][arm], cfg["model"]
    if t["orientation"] != "copy":
        raise ValueError("solo está implementada orientation: copy (1:1, sin escala)")
    chain = kin.load_chain(kin.SO_ARM100_MJCF, m["tip_body"], a["tcp_offset_m"])
    tc = TeleopConfig(
        period_s=1 / t["rate_hz"], scale=t["position_scale"], orientation=t["orientation"],
        deadband_m=t["deadband_m"], one_euro=t["one_euro"],
        max_speed_m_s=t["max_speed_m_s"], max_accel_m_s2=t["max_accel_m_s2"],
        watchdog_timeout_s=t["watchdog_timeout_ms"] / 1e3,
        workspace_lower=np.array(a["workspace"]["lower_m"], float),
        workspace_upper=np.array(a["workspace"]["upper_m"], float),
        workspace_mode=a["workspace"]["mode"],
        model_to_follower=np.array(a["model_to_follower_rotation"], float),
    )
    cal = Calibration(tuple(a["servo_ids"]), tuple(a["offsets_counts"]), tuple(a["signs"]))
    return chain, tc, cal


class ArmTeleop:
    """Controlador de un brazo seguidor.

    - Embrague suelto → mantiene la última consigna. Al pisarlo se re-indexa:
      la pose actual del modelo y la del seguidor pasan a ser la referencia (sin salto).
    - Watchdog: sin muestra en más de `watchdog_timeout_s` → mantiene y exige
      soltar el embrague antes de volver a transmitir.
    """

    def __init__(self, chain, cfg: TeleopConfig, q_init):
        self.chain, self.cfg = chain, cfg
        self.q = np.asarray(q_init, float).copy()
        self.engaged = False
        self.need_release = False
        self.last_sample = None
        self.last_sample_time = None
        self.last_now = None
        self.status = "init"
        self.filter = OneEuroFilter(**cfg.one_euro)
        self.limiter = MotionLimiter(cfg.max_speed_m_s, cfg.max_accel_m_s2)

    def _hold(self, reason, latch=False):
        self.engaged = False
        self.need_release |= latch
        self.status = reason
        return self.q.copy()

    def fault(self, reason):
        """Evento de seguridad externo (p. ej. error de bus)."""
        return self._hold(reason, latch=True)

    def step(self, now, sample: PoseSample | None, clutch: bool):
        dt = self.cfg.period_s if self.last_now is None else max(now - self.last_now, 1e-6)
        self.last_now = now
        if sample is not None:
            self.last_sample, self.last_sample_time = sample, now

        wd = safety.watchdog(now, self.last_sample_time, self.cfg.watchdog_timeout_s)
        if not wd.allowed:
            return self._hold(wd.reason, latch=True)
        if not clutch:
            self.need_release = False
            return self._hold("embrague suelto: manteniendo")
        if self.need_release:
            return self._hold(self.status)

        s, M = self.last_sample, self.cfg.model_to_follower
        if not self.engaged:  # re-indexado
            self.ref_model = (np.asarray(s.p, float), np.asarray(s.R, float))
            self.ref_follower = kin.fk(self.chain, self.q)
            self.held = np.zeros(3)
            self.filter.reset(self.ref_follower[0])
            self.limiter.reset(self.ref_follower[0])
            self.engaged = True
            self.status = "transmitiendo"
            return self.q.copy()

        delta = scale_delta(M @ (s.p - self.ref_model[0]), self.cfg.scale)
        self.held = deadband(delta, self.held, self.cfg.deadband_m)
        p_goal = self.ref_follower[0] + self.held
        # Orientación copiada 1:1 (rotación relativa, sin escala).
        R_goal = M @ (s.R @ self.ref_model[1].T) @ M.T @ self.ref_follower[1]

        p_cmd = self.limiter(self.filter(p_goal, dt), dt)
        ws = safety.workspace(p_cmd, self.cfg.workspace_lower, self.cfg.workspace_upper,
                              self.cfg.workspace_mode)
        if not ws.allowed:
            self.limiter.reset(kin.fk(self.chain, self.q)[0])
            self.status = ws.reason
            return self.q.copy()
        q_new, ok = kin.ik(self.chain, ws.value, R_goal, self.q)
        jl = safety.joint_limits(q_new, self.chain.lower, self.chain.upper)
        if not ok or not jl.allowed:
            self.limiter.reset(kin.fk(self.chain, self.q)[0])
            self.status = jl.reason or "IK: objetivo no alcanzable"
            return self.q.copy()
        self.q = q_new
        self.status = ws.reason or "transmitiendo"
        return self.q.copy()


def run_cycle(now, source: PoseSource, clutch, arm: ArmTeleop, bus: Bus, cal: Calibration):
    """Un ciclo: leer fuente → controlador → escribir metas al bus."""
    q = arm.step(now, source.read(), clutch)
    try:
        bus.sync_write_goal_positions(rad_to_counts(q, cal))
        return counts_to_rad(bus.sync_read_positions(list(cal.ids)), cal)
    except BusError as e:
        arm.fault(f"bus: {e}")
        return None
