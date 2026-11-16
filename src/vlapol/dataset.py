"""LeRobotDataset schema shared by demo collection and evaluation."""

from __future__ import annotations

from .env import ACTION_NAMES, STATE_NAMES

DEFAULT_REPO_ID = "local/vlapol_pickplace"
ROBOT_TYPE = "floating_gripper"


def image_key(camera: str) -> str:
    return f"observation.images.{camera}"


def make_features(cameras: tuple[str, ...], image_size: tuple[int, int]) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
    }
    for camera in cameras:
        features[image_key(camera)] = {
            "dtype": "video",
            "shape": (image_size[0], image_size[1], 3),
            "names": ["height", "width", "channels"],
        }
    return features
