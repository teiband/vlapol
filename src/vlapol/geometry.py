"""Rigid-transform helpers. All poses are 4x4 homogeneous matrices T_a_b (maps b-coords into a)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# ARCore camera poses use the OpenGL convention (x right, y up, looking along -z).
# Right-multiplying by this converts to the OpenCV convention (x right, y down, z forward).
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def pose7_to_T(pose7) -> np.ndarray:
    """[tx, ty, tz, qx, qy, qz, qw] -> 4x4."""
    p = np.asarray(pose7, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(p[3:7]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def T_to_pose7(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_quat()])


def arcore_pose_to_T_world_cam(pose7) -> np.ndarray:
    """ARCore camera pose -> T_world_cam with an OpenCV-convention camera frame."""
    return pose7_to_T(pose7) @ GL_TO_CV


def make_T(translation, rotvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    T[:3, 3] = translation
    return T


def inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    R = T[:3, :3]
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def relative(T_w_a: np.ndarray, T_w_b: np.ndarray) -> np.ndarray:
    """Pose of b expressed in frame a."""
    return inv(T_w_a) @ T_w_b


def T_to_vec6(T: np.ndarray) -> np.ndarray:
    """4x4 -> [x, y, z, rx, ry, rz] (rotation vector)."""
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])


def vec6_to_T(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return make_T(v[:3], v[3:6])
