"""Collect scripted-expert demonstrations straight into a LeRobotDataset.

    python -m vlapol.collect --root data/vlapol_pickplace --episodes 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import DEFAULT_REPO_ID, ROBOT_TYPE, image_key, make_features
from .env import PickPlaceEnv
from .expert import ScriptedExpert


def collect(
    root: Path,
    episodes: int = 100,
    repo_id: str = DEFAULT_REPO_ID,
    cameras: tuple[str, ...] = ("wrist",),
    image_size: tuple[int, int] = (256, 256),
    fps: int = 20,
    noise: float = 0.1,
    seed: int = 0,
    resume: bool = False,
) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if resume and (root / "meta" / "info.json").exists():
        dataset = LeRobotDataset.resume(repo_id, root=root)
        if set(make_features(cameras, image_size)) - set(dataset.features):
            raise ValueError(f"--cameras {cameras} do not match the existing dataset at {root}")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"{root} is not empty; pass --resume to append or choose another --root")
        dataset = LeRobotDataset.create(
            repo_id,
            fps=fps,
            features=make_features(cameras, image_size),
            root=root,
            robot_type=ROBOT_TYPE,
            use_videos=True,
        )

    done = dataset.num_episodes
    env = PickPlaceEnv(cameras=cameras, image_size=image_size, control_hz=fps)
    # Seeds continue after the ones already used, so appended episodes are new layouts.
    episode_seed = seed + done
    try:
        while done < episodes:
            obs = env.reset(seed=episode_seed)
            expert = ScriptedExpert(env, noise=noise, seed=episode_seed)
            episode_seed += 1
            success = False
            for _ in range(env.max_steps):
                action = expert.act()
                frame = {"observation.state": obs["state"], "action": action, "task": env.task}
                for camera in cameras:
                    frame[image_key(camera)] = obs["images"][camera]
                dataset.add_frame(frame)
                obs, _, success, truncated, _ = env.step(action)
                if success or truncated:
                    break
            if not success:  # keep only successful demonstrations
                dataset.clear_episode_buffer()
                print(f"seed {episode_seed - 1}: expert failed, episode discarded")
                continue
            dataset.save_episode()
            done += 1
            print(f"episode {done}/{episodes} saved ({env.task!r})")
    finally:
        dataset.finalize()
        env.close()
    print(f"dataset: {root}  episodes={dataset.num_episodes}  frames={dataset.num_frames}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="dataset directory")
    parser.add_argument("--episodes", type=int, default=100, help="total episodes the dataset should hold")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--cameras", nargs="+", default=["wrist"], choices=["wrist", "scene"])
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.1, help="std of expert action noise in free-space motion")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="append to an existing dataset at --root")
    args = parser.parse_args()
    collect(
        root=args.root,
        episodes=args.episodes,
        repo_id=args.repo_id,
        cameras=tuple(args.cameras),
        image_size=tuple(args.image_size),
        fps=args.fps,
        noise=args.noise,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
