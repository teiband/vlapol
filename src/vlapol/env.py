"""MuJoCo pick-and-place environment: floating parallel-jaw gripper with a wrist camera."""

from __future__ import annotations

from importlib import resources

import mujoco
import numpy as np

CUBE_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.10, 0.10),
    "blue": (0.10, 0.25, 0.85),
    "yellow": (0.90, 0.80, 0.10),
}

STATE_NAMES = ["x", "y", "z", "yaw", "gripper"]
ACTION_NAMES = ["dx", "dy", "dz", "dyaw", "gripper"]

MAX_DPOS = 0.02  # m per control step at |action| = 1
MAX_DYAW = 0.15  # rad per control step at |action| = 1
FINGER_OPEN = 0.04  # finger joint travel (m) when fully open
CUBE_HALF = 0.02
TARGET_RADIUS = 0.04
SUCCESS_RADIUS = 0.03

_SPAWN_RANGE = 0.12  # cube / target xy are sampled in [-r, r]^2
_MIN_CUBE_TARGET_DIST = 0.10
_TCP_LOW = np.array([-0.25, -0.25, 0.012])
_TCP_HIGH = np.array([0.25, 0.25, 0.40])
_YAW_LIMIT = 1.5
_GRIPPER_HOME_Z = 0.25  # world z of the TCP when the gz joint is at 0


def task_string(color: str) -> str:
    return f"pick up the {color} cube and place it on the green target"


class PickPlaceEnv:
    """Pick the cube, place it on the green disc.

    Observation: ``{"state": float32[5], "images": {camera: uint8[H, W, 3]}}``
        state = TCP x, y, z (m), yaw (rad), gripper opening in [0, 1].
    Action: float32[5] in [-1, 1] = dx, dy, dz, dyaw (scaled deltas on the pose setpoint)
        and the absolute gripper command (-1 = closed, +1 = open).
    """

    def __init__(
        self,
        cameras: tuple[str, ...] = ("wrist",),
        image_size: tuple[int, int] = (256, 256),
        control_hz: int = 20,
        max_steps: int = 300,
        seed: int | None = None,
    ):
        xml = resources.files("vlapol").joinpath("assets/pick_place.xml").read_text()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.cameras = tuple(cameras)
        self.image_size = tuple(image_size)  # (height, width)
        self.control_hz = control_hz
        self.max_steps = max_steps
        self._substeps = round(1.0 / (control_hz * self.model.opt.timestep))
        self._rng = np.random.default_rng(seed)
        self._renderer: mujoco.Renderer | None = None

        m = self.model
        self._tcp_site = m.site("tcp").id
        self._cube_geom = m.geom("cube").id
        self._cube_qpos = m.joint("cube").qposadr[0]
        self._target_mocap = m.body("target").mocapid[0]
        self._finger_qpos = m.joint("finger_left").qposadr[0]
        self._yaw_qpos = m.joint("gyaw").qposadr[0]
        self._pose_qpos = m.joint("gx").qposadr[0]  # gx, gy, gz, gyaw are contiguous

        self.cube_color = "red"
        self.task = task_string(self.cube_color)
        self._setpoint = np.zeros(4)  # x, y, z (world), yaw
        self._t = 0

    # ------------------------------------------------------------------ helpers
    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp_site].copy()

    @property
    def yaw(self) -> float:
        return float(self.data.qpos[self._yaw_qpos])

    @property
    def gripper_opening(self) -> float:
        return float(np.clip(self.data.qpos[self._finger_qpos] / FINGER_OPEN, 0.0, 1.0))

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.qpos[self._cube_qpos : self._cube_qpos + 3].copy()

    @property
    def cube_yaw(self) -> float:
        w, x, y, z = self.data.qpos[self._cube_qpos + 3 : self._cube_qpos + 7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    @property
    def target_pos(self) -> np.ndarray:
        return self.data.mocap_pos[self._target_mocap].copy()

    def _write_ctrl(self, gripper: float) -> None:
        self.data.ctrl[0:2] = self._setpoint[0:2]
        self.data.ctrl[2] = self._setpoint[2] - _GRIPPER_HOME_Z
        self.data.ctrl[3] = self._setpoint[3]
        self.data.ctrl[4] = FINGER_OPEN * (np.clip(gripper, -1.0, 1.0) + 1.0) / 2.0

    # ---------------------------------------------------------------------- API
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        mujoco.mj_resetData(self.model, self.data)

        cube_xy = rng.uniform(-_SPAWN_RANGE, _SPAWN_RANGE, 2)
        while True:
            target_xy = rng.uniform(-_SPAWN_RANGE, _SPAWN_RANGE, 2)
            if np.linalg.norm(target_xy - cube_xy) >= _MIN_CUBE_TARGET_DIST:
                break
        cube_yaw = rng.uniform(-0.6, 0.6)
        q = self._cube_qpos
        self.data.qpos[q : q + 3] = [*cube_xy, CUBE_HALF]
        self.data.qpos[q + 3 : q + 7] = [np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)]
        self.data.mocap_pos[self._target_mocap] = [*target_xy, 0.001]

        self.cube_color = str(rng.choice(list(CUBE_COLORS)))
        self.model.geom_rgba[self._cube_geom] = [*CUBE_COLORS[self.cube_color], 1.0]
        self.task = task_string(self.cube_color)

        start = np.array([*rng.uniform(-0.04, 0.04, 2), rng.uniform(0.20, 0.26), 0.0])
        p = self._pose_qpos
        self.data.qpos[p : p + 2] = start[0:2]
        self.data.qpos[p + 2] = start[2] - _GRIPPER_HOME_Z
        self.data.qpos[self._finger_qpos : self._finger_qpos + 2] = FINGER_OPEN
        self._setpoint = start
        self._write_ctrl(gripper=1.0)
        self._t = 0
        mujoco.mj_forward(self.model, self.data)
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._setpoint[0:3] = np.clip(self._setpoint[0:3] + action[0:3] * MAX_DPOS, _TCP_LOW, _TCP_HIGH)
        self._setpoint[3] = np.clip(self._setpoint[3] + action[3] * MAX_DYAW, -_YAW_LIMIT, _YAW_LIMIT)
        self._write_ctrl(gripper=action[4])
        mujoco.mj_step(self.model, self.data, self._substeps)
        self._t += 1

        success = self.is_success()
        truncated = self._t >= self.max_steps and not success
        info = {"is_success": success, "cube_to_target": self._cube_to_target()}
        return self._observe(), float(success), success, truncated, info

    def _cube_to_target(self) -> float:
        return float(np.linalg.norm(self.cube_pos[:2] - self.target_pos[:2]))

    def is_success(self) -> bool:
        """Cube resting on the target, released, gripper retreated above it."""
        cube = self.cube_pos
        return bool(
            self._cube_to_target() < SUCCESS_RADIUS
            and cube[2] < CUBE_HALF + 0.005
            and self.gripper_opening > 0.7
            and self.tcp_pos[2] > cube[2] + 0.05
        )

    def _observe(self) -> dict:
        state = np.array([*self.tcp_pos, self.yaw, self.gripper_opening], dtype=np.float32)
        return {"state": state, "images": {cam: self.render(cam) for cam in self.cameras}}

    def render(self, camera: str = "scene", size: tuple[int, int] | None = None) -> np.ndarray:
        """RGB uint8 image (H, W, 3). ``size`` other than the env's creates a one-off renderer."""
        if size is not None and tuple(size) != self.image_size:
            with mujoco.Renderer(self.model, height=size[0], width=size[1]) as renderer:
                renderer.update_scene(self.data, camera=camera)
                return renderer.render()
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.image_size[0], width=self.image_size[1])
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
