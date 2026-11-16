import numpy as np
import pytest

from vlapol.env import PickPlaceEnv, task_string
from vlapol.expert import ScriptedExpert


def rollout(env: PickPlaceEnv, seed: int, noise: float = 0.0) -> bool:
    env.reset(seed=seed)
    expert = ScriptedExpert(env, noise=noise, seed=seed)
    for _ in range(env.max_steps):
        _, _, success, truncated, _ = env.step(expert.act())
        if success or truncated:
            break
    return success


@pytest.mark.parametrize("noise", [0.0, 0.1])
def test_expert_solves_task(noise):
    env = PickPlaceEnv(cameras=())
    assert all(rollout(env, seed, noise) for seed in range(20))


def test_idle_policy_does_not_succeed():
    env = PickPlaceEnv(cameras=(), max_steps=50)
    env.reset(seed=0)
    for _ in range(50):
        obs, reward, success, truncated, _ = env.step(np.array([0, 0, 0, 0, 1.0]))
    assert not success and truncated and reward == 0.0
    assert obs["state"].shape == (5,) and obs["state"].dtype == np.float32


def test_reset_is_deterministic_and_sets_task():
    env = PickPlaceEnv(cameras=())
    a = env.reset(seed=7)["state"]
    cube, target, task = env.cube_pos, env.target_pos, env.task
    b = env.reset(seed=7)["state"]
    assert np.array_equal(a, b) and np.array_equal(cube, env.cube_pos) and np.array_equal(target, env.target_pos)
    assert task == env.task == task_string(env.cube_color)


def test_wrist_camera_image():
    env = PickPlaceEnv(cameras=("wrist",), image_size=(96, 128))
    try:
        image = env.reset(seed=0)["images"]["wrist"]
    except Exception as err:  # noqa: BLE001 - no OpenGL backend on this machine
        pytest.skip(f"offscreen rendering unavailable: {err}")
    assert image.shape == (96, 128, 3) and image.dtype == np.uint8
    assert image.std() > 5  # not a blank frame
    env.close()
