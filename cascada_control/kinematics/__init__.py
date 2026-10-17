"""FK/IK del SO-ARM100 leídas del MJCF oficial (ver models/so_arm100/SOURCE.md)."""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

SO_ARM100_MJCF = Path(__file__).parent / "models" / "so_arm100" / "so_arm100.xml"


def _vec(s, default):
    return np.array([float(v) for v in s.split()]) if s else np.array(default, float)


def quat_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_R(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def R_to_quat(R):
    """Cuaternión (w, x, y, z) con w >= 0 (método de Shepperd)."""
    t = np.trace(R)
    i = int(np.argmax([t, R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = 2 * np.sqrt(1 + t)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif i == 1:
        s = 2 * np.sqrt(1 + 2 * R[0, 0] - t)
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif i == 2:
        s = 2 * np.sqrt(1 + 2 * R[1, 1] - t)
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + 2 * R[2, 2] - t)
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    q = np.array(q)
    return q if q[0] >= 0 else -q


def rot_log(R):
    """Vector eje·ángulo de R (rad)."""
    q = R_to_quat(R)
    n = np.linalg.norm(q[1:])
    return np.zeros(3) if n < 1e-15 else 2 * np.arctan2(n, q[0]) * q[1:] / n


def _body_R(el):
    # MuJoCo por defecto: eulerseq="xyz" intrínseco, ángulos en radianes (compiler angle="radian").
    if el.get("quat"):
        return quat_to_R(_vec(el.get("quat"), None))
    if el.get("euler"):
        a, b, c = _vec(el.get("euler"), None)
        return axis_angle_R([1, 0, 0], a) @ axis_angle_R([0, 1, 0], b) @ axis_angle_R([0, 0, 1], c)
    return np.eye(3)


def _joint_defaults(root):
    out = {}

    def walk(node, inherited):
        j = node.find("joint")
        attrs = {**inherited, **(j.attrib if j is not None else {})}
        if node.get("class"):
            out[node.get("class")] = attrs
        for child in node.findall("default"):
            walk(child, attrs)

    for d in root.findall("default"):
        walk(d, {})
    return out


@dataclass(frozen=True)
class Chain:
    names: tuple
    pos: np.ndarray      # (n_links, 3) posición del cuerpo respecto al padre
    rot: np.ndarray      # (n_links, 3, 3)
    axis: np.ndarray     # (n_links, 3) eje articular en el marco del cuerpo
    lower: np.ndarray
    upper: np.ndarray
    tcp: np.ndarray      # offset del TCP en el marco del cuerpo final

    @property
    def n(self):
        return len(self.names)


def load_chain(mjcf=SO_ARM100_MJCF, tip_body="Fixed_Jaw", tcp_offset=(0, 0, 0)):
    root = ET.parse(mjcf).getroot()
    defaults = _joint_defaults(root)

    def find_path(body, path):
        path = path + [body]
        if body.get("name") == tip_body:
            return path
        for child in body.findall("body"):
            p = find_path(child, path)
            if p:
                return p
        return None

    path = None
    for b in root.find("worldbody").findall("body"):
        path = path or find_path(b, [])
    if path is None:
        raise ValueError(f"cuerpo {tip_body!r} no encontrado en {mjcf}")

    names, pos, rot, axis, lo, hi = [], [], [], [], [], []
    for body in path:
        joints = body.findall("joint")
        if len(joints) != 1:
            if body is path[0] and not joints and body.get("pos") is None and body.get("quat") is None:
                continue  # base fija en el origen
            raise ValueError(f"cuerpo {body.get('name')}: se espera exactamente una articulación")
        j = joints[0]
        attrs = {**defaults.get(j.get("class", ""), {}), **j.attrib}
        if attrs.get("type", "hinge") != "hinge" or attrs.get("pos") not in (None, "0 0 0"):
            raise ValueError(f"articulación {j.get('name')}: solo hinge en el origen del cuerpo")
        r = _vec(attrs.get("range"), [-np.pi, np.pi])
        names.append(j.get("name"))
        pos.append(_vec(body.get("pos"), [0, 0, 0]))
        rot.append(_body_R(body))
        axis.append(_vec(attrs.get("axis"), [0, 0, 1]))
        lo.append(r[0])
        hi.append(r[1])
    return Chain(tuple(names), np.array(pos), np.array(rot), np.array(axis),
                 np.array(lo), np.array(hi), np.asarray(tcp_offset, float))


def fk(chain, q, jacobian=False):
    """Pose del TCP (p, R) en el marco de la base; opcionalmente Jp, Jo (3×n)."""
    p, R = np.zeros(3), np.eye(3)
    origins, axes = [], []
    for i in range(chain.n):
        p = p + R @ chain.pos[i]
        R = R @ chain.rot[i]
        origins.append(p)
        axes.append(R @ chain.axis[i])
        R = R @ axis_angle_R(chain.axis[i], q[i])
    p = p + R @ chain.tcp
    if not jacobian:
        return p, R
    Jo = np.array(axes).T
    Jp = np.array([np.cross(a, p - o) for a, o in zip(axes, origins)]).T
    return p, R, Jp, Jo


def ik(chain, p_target, R_target, q0, tol=1e-9, max_iter=100, damping=1e-6, max_step=0.2):
    """IK con prioridad: posición exacta; orientación en el espacio nulo.

    El SO-ARM100 tiene 5 ejes: no siempre existe la orientación pedida; se
    minimiza su error sin sacrificar la posición. `R_target=None` → solo posición.
    Un eje en su límite que empuja hacia fuera se congela en esa iteración.
    Devuelve (q, convergió en posición).
    """
    q = np.clip(np.asarray(q0, float), chain.lower, chain.upper)
    for _ in range(max_iter):
        p, R, Jp, Jo = fk(chain, q, jacobian=True)
        ep = p_target - p
        eo = None if R_target is None else rot_log(R_target @ R.T)
        free = np.ones(chain.n, bool)
        for _ in range(chain.n):
            dq = _step(Jp * free, Jo * free, ep, eo, damping)
            out = ((q <= chain.lower) & (dq < 0)) | ((q >= chain.upper) & (dq > 0))
            if not np.any(out & free):
                break
            free &= ~out
        big = np.max(np.abs(dq))
        if big > max_step:
            dq *= max_step / big
        q_next = np.clip(q + dq, chain.lower, chain.upper)
        moved = np.max(np.abs(q_next - q))
        q = q_next
        if moved < 1e-10:  # convergió o quedó atascado en un límite articular
            break
    p, _ = fk(chain, q)
    return q, bool(np.linalg.norm(p_target - p) < max(tol * 10, 1e-9))


def _step(Jp, Jo, ep, eo, damping):
    n = Jp.shape[1]
    Jp_pinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + damping**2 * np.eye(3))
    dq = Jp_pinv @ ep
    if eo is not None:
        N = np.eye(n) - Jp_pinv @ Jp
        dq = dq + N @ np.linalg.pinv(Jo @ N, rcond=1e-6) @ (eo - Jo @ dq)
    return dq
