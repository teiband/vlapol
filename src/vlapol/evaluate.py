"""Closed-loop evaluation of a trained policy in the MuJoCo pick-and-place env.

    python -m vlapol.evaluate --policy-path <run>/checkpoints/020000/pretrained_model --episodes 50

Use ``--expert`` instead of ``--policy-path`` to sanity-check the env with the scripted expert.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import ROBOT_TYPE, image_key
from .env import PickPlaceEnv
from .expert import ScriptedExpert

EVAL_SEED_OFFSET = 1_000_000  # layouts disjoint from the demo seeds used by vlapol.collect


class PolicyRunner:
    """Wraps a LeRobot policy and its pre/post-processors for single-env rollouts."""

    def __init__(self, policy_path: str | Path, device: str | None = None):
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy_path = str(policy_path)

        config = PreTrainedConfig.from_pretrained(policy_path)
        config.device = self.device
        self.policy = get_policy_class(config.type).from_pretrained(policy_path, config=config)
        self.policy.to(self.device).eval()

        # The saved preprocessor already holds the dataset normalization stats and the camera
        # rename map used in training; only the device has to follow this machine.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

    def reset(self) -> None:
        self.policy.reset()  # drop the queued action chunk

    def act(self, obs: dict, task: str) -> np.ndarray:
        torch = self.torch
        batch = {"observation.state": torch.from_numpy(obs["state"]), "task": task, "robot_type": ROBOT_TYPE}
        for camera, image in obs["images"].items():
            # uint8 HWC -> float CHW in [0, 1], as LeRobotDataset returns frames in training
            batch[image_key(camera)] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        with torch.inference_mode():
            action = self.policy.select_action(self.preprocessor(batch))
            action = self.postprocessor(action)
        return action.squeeze(0).float().cpu().numpy()


def checkpoint_cameras(policy_path: str | Path) -> tuple[tuple[str, ...], tuple[int, int]] | None:
    """Cameras and image size of the training dataset, read from the checkpoint's train config."""
    cfg_file = Path(policy_path) / "train_config.json"
    if not cfg_file.is_file():
        return None
    root = json.loads(cfg_file.read_text()).get("dataset", {}).get("root")
    info_file = Path(root) / "meta" / "info.json" if root else None
    if info_file is None or not info_file.is_file():
        return None
    features = json.loads(info_file.read_text())["features"]
    cams = [k for k, ft in features.items() if ft["dtype"] in ("video", "image")]
    height, width = features[cams[0]]["shape"][:2]
    return tuple(k.removeprefix("observation.images.") for k in cams), (height, width)


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")))
        container.mux(stream.encode())


def _video_frame(env: PickPlaceEnv, obs: dict) -> np.ndarray:
    """Scene overview next to the policy's own camera views."""
    panels = [env.render("scene"), *(obs["images"][c] for c in env.cameras if c != "scene")]
    return np.concatenate(panels, axis=1)


def evaluate(
    runner: PolicyRunner | None,
    episodes: int = 50,
    cameras: tuple[str, ...] = ("wrist",),
    image_size: tuple[int, int] = (256, 256),
    fps: int = 20,
    max_steps: int = 300,
    seed: int = 0,
    output_dir: Path | None = None,
    videos: int = 5,
) -> dict:
    """Roll out ``runner`` (or the scripted expert when ``None``) and return success statistics."""
    env = PickPlaceEnv(cameras=cameras, image_size=image_size, control_hz=fps, max_steps=max_steps)
    results = []
    try:
        for ep in range(episodes):
            ep_seed = EVAL_SEED_OFFSET + seed + ep
            obs = env.reset(seed=ep_seed)
            expert = ScriptedExpert(env) if runner is None else None
            if runner is not None:
                runner.reset()
            record = output_dir is not None and ep < videos
            frames = []
            success, info = False, {}
            for step in range(max_steps):
                if record:
                    frames.append(_video_frame(env, obs))
                action = expert.act() if runner is None else runner.act(obs, env.task)
                obs, _, success, truncated, info = env.step(action)
                if success or truncated:
                    break
            results.append(
                {
                    "episode": ep,
                    "seed": ep_seed,
                    "task": env.task,
                    "success": bool(success),
                    "steps": step + 1,
                    "cube_to_target_m": round(info["cube_to_target"], 4),
                }
            )
            print(
                f"episode {ep + 1}/{episodes}: {'success' if success else 'FAIL'} in {step + 1} steps  {env.task!r}",
                flush=True,
            )
            if record:
                frames.append(_video_frame(env, obs))
                tag = "success" if success else "fail"
                write_video(output_dir / f"episode_{ep:03d}_{tag}.mp4", frames, fps)
    finally:
        env.close()

    n_success = sum(r["success"] for r in results)
    summary = {
        "episodes": episodes,
        "successes": n_success,
        "success_rate": n_success / max(episodes, 1),
        "mean_steps_when_successful": float(np.mean([r["steps"] for r in results if r["success"]] or [0])),
        "per_episode": results,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "eval.json").write_text(json.dumps(summary, indent=2))
    print(f"success rate: {n_success}/{episodes} = {summary['success_rate']:.1%}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--policy-path", help="checkpoint 'pretrained_model' directory (or a Hub repo id)")
    source.add_argument("--expert", action="store_true", help="evaluate the scripted expert instead of a policy")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--cameras", nargs="+", choices=["wrist", "scene"], help="default: as in training")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"), help="default: as in training")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    parser.add_argument("--n-action-steps", type=int, help="actions executed per predicted chunk before re-planning")
    parser.add_argument("--output-dir", type=Path, help="write eval.json and rollout videos here")
    parser.add_argument("--videos", type=int, default=5, help="number of episodes to record")
    args = parser.parse_args()

    trained = None if args.expert else checkpoint_cameras(args.policy_path)
    cameras = tuple(args.cameras) if args.cameras else (trained[0] if trained else ("wrist",))
    image_size = tuple(args.image_size) if args.image_size else (trained[1] if trained else (256, 256))

    runner = None
    if not args.expert:
        runner = PolicyRunner(args.policy_path, device=args.device)
        if args.n_action_steps:
            runner.policy.config.n_action_steps = args.n_action_steps
    evaluate(
        runner,
        episodes=args.episodes,
        cameras=cameras,
        image_size=image_size,
        fps=args.fps,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        videos=args.videos,
    )


if __name__ == "__main__":
    main()
