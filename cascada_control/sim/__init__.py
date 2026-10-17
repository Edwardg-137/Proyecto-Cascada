"""Backend MuJoCo opcional (`uv sync --extra sim`): implementa la interfaz Bus.

Las mallas del MJCF no se vendorizan; se quitan al cargar (las inercias son
explícitas en el MJCF, así que la dinámica no cambia; las colisiones con malla sí).
"""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from cascada_control.drivers import Calibration, counts_to_rad, rad_to_counts
from cascada_control.kinematics import SO_ARM100_MJCF


def load_model(mjcf=SO_ARM100_MJCF):
    root = ET.parse(mjcf).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "mesh" or (child.tag == "geom" and child.get("mesh")):
                parent.remove(child)
            elif child.tag == "geom" and child.get("class") in ("visual", "collision", "motor_visual"):
                parent.remove(child)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class MujocoBus:
    """Bus simulado: metas → ctrl de los actuadores de posición; avanza `substeps` por escritura."""

    def __init__(self, cal: Calibration, q_init, substeps=10):
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.cal, self.substeps = cal, substeps
        n = len(cal.ids)
        self.data.qpos[:n] = q_init
        self.data.ctrl[:n] = q_init
        mujoco.mj_forward(self.model, self.data)

    def sync_write_goal_positions(self, goals):
        n = len(self.cal.ids)
        self.data.ctrl[:n] = counts_to_rad(goals, self.cal)
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)

    def sync_read_positions(self, ids):
        counts = rad_to_counts(self.data.qpos[: len(self.cal.ids)], self.cal)
        return {i: counts[i] for i in ids}
