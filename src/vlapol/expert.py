"""Scripted pick-and-place expert. Uses privileged sim state (cube / target pose), not images."""

from __future__ import annotations

import numpy as np

from .env import CUBE_HALF, MAX_DPOS, MAX_DYAW, PickPlaceEnv

HOVER_Z = 0.13  # pre-grasp height
CARRY_Z = 0.20  # high enough that the wrist camera sees the target anywhere on the table
GRASP_Z = CUBE_HALF
RELEASE_Z = CUBE_HALF + 0.008
RETREAT_Z = 0.14
OPEN, CLOSE = 1.0, -1.0


class ScriptedExpert:
    """Waypoint state machine: hover -> descend -> grasp -> lift -> carry -> lower -> release -> retreat."""

    def __init__(self, env: PickPlaceEnv, noise: float = 0.0, seed: int | None = None):
        self.env = env
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.phase = 0
        self._wait = 0
        # The grasp yaw is fixed at the start; a cube is symmetric under 90 degree turns.
        self._grasp_yaw = (self.env.cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4

    def _move(self, goal_xyz, yaw, gripper, tol=0.006, speed=1.0) -> tuple[np.ndarray, bool]:
        env = self.env
        err = np.asarray(goal_xyz) - env.tcp_pos
        yaw_err = yaw - env.yaw
        action = np.zeros(5, dtype=np.float32)
        action[0:3] = np.clip(0.6 * err / MAX_DPOS, -speed, speed)
        action[3] = np.clip(0.6 * yaw_err / MAX_DYAW, -1.0, 1.0)
        action[4] = gripper
        return action, bool(np.linalg.norm(err) < tol and abs(yaw_err) < 0.05)

    def _hold(self, gripper: float, steps: int) -> tuple[np.ndarray, bool]:
        self._wait += 1
        action = np.zeros(5, dtype=np.float32)
        action[4] = gripper
        return action, self._wait >= steps

    def act(self) -> np.ndarray:
        env = self.env
        cube, target, yaw = env.cube_pos, env.target_pos, self._grasp_yaw
        transit = self.phase in (0, 4)
        if self.phase == 0:
            action, done = self._move([cube[0], cube[1], HOVER_Z], yaw, OPEN, tol=0.01)
        elif self.phase == 1:
            action, done = self._move([cube[0], cube[1], GRASP_Z], yaw, OPEN, tol=0.004, speed=0.6)
        elif self.phase == 2:
            action, done = self._hold(CLOSE, steps=8)
        elif self.phase == 3:
            action, done = self._move([env.tcp_pos[0], env.tcp_pos[1], CARRY_Z], yaw, CLOSE, tol=0.01)
        elif self.phase == 4:
            action, done = self._move([target[0], target[1], CARRY_Z], yaw, CLOSE, tol=0.006)
        elif self.phase == 5:
            action, done = self._move([target[0], target[1], RELEASE_Z], yaw, CLOSE, tol=0.004, speed=0.6)
        elif self.phase == 6:
            action, done = self._hold(OPEN, steps=8)
        else:
            action, done = self._move([env.tcp_pos[0], env.tcp_pos[1], RETREAT_Z], yaw, OPEN, tol=0.01)
            done = False
        if done:
            self.phase += 1
            self._wait = 0
        if self.noise > 0 and transit:
            # Perturb free-space motion only; the feedback law corrects it, which yields recovery data.
            action[0:3] += self._rng.normal(0.0, self.noise, 3).astype(np.float32)
        return np.clip(action, -1.0, 1.0)
