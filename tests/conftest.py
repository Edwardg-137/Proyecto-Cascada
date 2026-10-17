from pathlib import Path

import numpy as np
import pytest

from cascada_control.teleop import arm_setup, load_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "teleop.yaml"
HOME_Q = np.array([0.0, -1.57, 1.57, 1.57, -1.57])  # keyframe "home" del MJCF (sin pinza)


@pytest.fixture
def cfg():
    return load_config(CONFIG)


@pytest.fixture
def left(cfg):
    return arm_setup(cfg, "left")
