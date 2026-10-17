"""Escala cartesiana, zona muerta, filtro 1€ y limitador de velocidad/aceleración."""
import numpy as np


def scale_delta(delta, scale):
    """Delta de posición de la herramienta modelo → delta del seguidor.

    `scale` es un escalar (8 → 1 mm modelo = 0.125 mm) o un vector por eje.
    Solo posición: la fuerza y la orientación no pasan por aquí.
    """
    s = np.asarray(scale, float)
    if np.any(s <= 0):
        raise ValueError("la escala debe ser > 0")
    return np.asarray(delta, float) / s


def deadband(x, held, band):
    """Mantiene `held` hasta que x se aleja más de `band`; entonces salta a x (sin sesgo)."""
    x = np.asarray(x, float)
    return x if np.linalg.norm(x - held) > band else held


class OneEuroFilter:
    """Filtro 1€ (Casiez et al., CHI 2012) para vectores."""

    def __init__(self, min_cutoff_hz, beta, d_cutoff_hz):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff_hz, beta, d_cutoff_hz
        self.x = self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1 / (2 * np.pi * cutoff)
        return 1 / (1 + tau / dt)

    def reset(self, x):
        self.x, self.dx = np.asarray(x, float).copy(), np.zeros_like(x, dtype=float)

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        if self.x is None:
            self.reset(x)
            return self.x.copy()
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * np.linalg.norm(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


class MotionLimiter:
    """Sigue un objetivo con |v| ≤ v_max y |a| ≤ a_max, frenando a tiempo."""

    def __init__(self, v_max, a_max):
        self.v_max, self.a_max = v_max, a_max
        self.p = self.v = None

    def reset(self, p):
        self.p, self.v = np.asarray(p, float).copy(), np.zeros(3)

    def __call__(self, target, dt):
        e = np.asarray(target, float) - self.p
        d = np.linalg.norm(e)
        ad = self.a_max * dt  # velocidad máxima que permite frenar en d con pasos discretos
        speed = min(self.v_max, d / dt, -ad / 2 + np.sqrt(ad * ad / 4 + 2 * self.a_max * d))
        v_des = e / d * speed if d > 0 else np.zeros(3)
        dv = v_des - self.v
        n = np.linalg.norm(dv)
        if n > self.a_max * dt:
            dv *= self.a_max * dt / n
        self.v = self.v + dv
        self.p = self.p + self.v * dt
        return self.p.copy()
