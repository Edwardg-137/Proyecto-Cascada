"""Reglas de seguridad de software: una función pura por regla.

Complementan al e-stop por hardware; nunca lo reemplazan (ver docs/safety.md).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""
    value: object = None


def watchdog(now, last_sample_time, timeout_s):
    """Sin muestra de pose en más de `timeout_s` → no permitido."""
    if last_sample_time is None:
        return Decision(False, "watchdog: sin muestras de pose")
    age = now - last_sample_time
    if age > timeout_s:
        return Decision(False, f"watchdog: última muestra hace {age * 1e3:.1f} ms > {timeout_s * 1e3:.1f} ms")
    return Decision(True)


def workspace(p, lower, upper, mode="reject"):
    """Caja permitida del TCP. mode='reject' rechaza; 'clamp' recorta e informa."""
    p, lo, hi = (np.asarray(v, float) for v in (p, lower, upper))
    if np.all((p >= lo) & (p <= hi)):
        return Decision(True, value=p)
    if mode == "clamp":
        return Decision(True, "workspace: objetivo recortado al límite", np.clip(p, lo, hi))
    if mode == "reject":
        return Decision(False, f"workspace: objetivo {np.round(p, 4).tolist()} fuera de la caja")
    raise ValueError(f"modo de workspace desconocido: {mode!r}")


def joint_limits(q, lower, upper):
    q = np.asarray(q, float)
    bad = np.flatnonzero((q < lower) | (q > upper))
    if bad.size:
        return Decision(False, f"límites articulares excedidos en ejes {bad.tolist()}")
    return Decision(True, value=q)
