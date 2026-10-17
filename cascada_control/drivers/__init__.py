"""Acceso al bus de servos. Solo hay implementación simulada.

El driver real (SDK Feetech SMS_STS, sync read/write a 1 Mbps) queda pendiente.
Registro citado en la investigación: carga actual en la dirección 60 (±1000);
no se usan otras direcciones hasta verificarlas contra la referencia publicada.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

COUNTS_PER_REV = 4096  # encoder STS3215 de 12 bit


class BusError(Exception):
    """Error de comunicación: el bucle lo trata como evento de seguridad."""


class Bus(Protocol):
    def sync_read_positions(self, ids: list[int]) -> dict[int, int]: ...
    def sync_write_goal_positions(self, goals: dict[int, int]) -> None: ...


@dataclass(frozen=True)
class Calibration:
    """Ids de servo, cuenta en q=0 y signo por eje. Único lugar de conversión."""
    ids: tuple
    offsets: tuple
    signs: tuple


def rad_to_counts(q, cal):
    k = COUNTS_PER_REV / (2 * np.pi)
    return {i: int(round(o + s * qi * k)) for i, qi, o, s in zip(cal.ids, q, cal.offsets, cal.signs)}


def counts_to_rad(counts, cal):
    k = 2 * np.pi / COUNTS_PER_REV
    return np.array([s * (counts[i] - o) * k for i, o, s in zip(cal.ids, cal.offsets, cal.signs)])


class MockBus:
    """Servo ideal: la posición presente es la última meta escrita."""

    def __init__(self, initial: dict[int, int]):
        self.positions = dict(initial)
        self.writes: list[dict[int, int]] = []
        self.fail = False

    def sync_read_positions(self, ids):
        if self.fail:
            raise BusError("mock: fallo de bus")
        return {i: self.positions[i] for i in ids}

    def sync_write_goal_positions(self, goals):
        if self.fail:
            raise BusError("mock: fallo de bus")
        self.writes.append(dict(goals))
        self.positions.update(goals)
