"""Fine-tune with ``lerobot-train``, mirroring checkpoints to a persistent directory (Google Drive).

Re-running the same command resumes from the newest checkpoint, so a Colab disconnect only
loses the steps since the last save:

    python -m vlapol.train --dataset-root data/vlapol_pickplace \\
        --run-dir /content/drive/MyDrive/vlapol/runs/smolvla --steps 20000

Training itself writes to ``--local-dir``: Drive is slow for many small writes and does not
support the ``checkpoints/last`` symlink that lerobot creates. Finished checkpoints are copied
to ``--run-dir`` in the background. Unknown arguments are forwarded to ``lerobot-train``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from .dataset import DEFAULT_REPO_ID

CHECKPOINTS = "checkpoints"
TRAIN_CONFIG = Path("pretrained_model") / "train_config.json"


def list_checkpoints(run_dir: Path) -> list[Path]:
    """Complete step checkpoints (``checkpoints/<digits>``), oldest first."""
    ckpt_dir = Path(run_dir) / CHECKPOINTS
    if not ckpt_dir.is_dir():
        return []
    found = [p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit() and (p / TRAIN_CONFIG).is_file()]
    return sorted(found, key=lambda p: int(p.name))


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = list_checkpoints(run_dir)
    return checkpoints[-1] if checkpoints else None


def _prune(run_dir: Path, keep: int, protect: Path | None = None) -> None:
    for old in list_checkpoints(run_dir)[:-keep]:
        if protect is None or old.resolve() != protect.resolve():
            shutil.rmtree(old, ignore_errors=True)


def sync_checkpoints(local_dir: Path, run_dir: Path, keep: int) -> Path | None:
    """Copy the newest finished local checkpoint to ``run_dir`` and prune old ones on both sides.

    lerobot points ``checkpoints/last`` at a checkpoint only after it is fully written, so the
    link target is always safe to copy.
    """
    last = Path(local_dir) / CHECKPOINTS / "last"
    if not last.is_symlink():
        return None
    src = last.resolve()
    if not (src / TRAIN_CONFIG).is_file():
        return None
    dst = Path(run_dir) / CHECKPOINTS / src.name
    if not dst.exists():
        tmp = dst.with_name(src.name + ".partial")  # not all digits, so never mistaken for a checkpoint
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, tmp)
        tmp.rename(dst)
        print(f"[vlapol] checkpoint {src.name} -> {dst}", flush=True)
    _prune(run_dir, keep)
    _prune(local_dir, keep, protect=src)
    return dst


def dataset_camera_keys(dataset_root: Path) -> list[str]:
    info = json.loads((Path(dataset_root) / "meta" / "info.json").read_text())
    return [k for k, ft in info["features"].items() if ft["dtype"] in ("video", "image")]


def camera_rename_map(policy_path: str, dataset_root: Path) -> dict[str, str]:
    """Map dataset camera keys onto the camera keys the pretrained policy was trained with.

    A pretrained checkpoint keeps its own image feature names; lerobot requires the dataset's
    cameras to be a subset of them (or renamed to be). Returns {} when no renaming is needed.
    """
    import lerobot.policies  # noqa: F401  (registers the policy config types, e.g. 'smolvla')
    from lerobot.configs import FeatureType, PreTrainedConfig

    config = PreTrainedConfig.from_pretrained(policy_path)
    policy_cams = [k for k, ft in (config.input_features or {}).items() if ft.type is FeatureType.VISUAL]
    dataset_cams = dataset_camera_keys(dataset_root)
    if not policy_cams or set(dataset_cams) <= set(policy_cams):
        return {}
    if len(dataset_cams) > len(policy_cams):
        raise ValueError(f"dataset has cameras {dataset_cams} but {policy_path} only takes {policy_cams}")
    return dict(zip(dataset_cams, policy_cams, strict=False))


def build_command(args: argparse.Namespace, extra: list[str]) -> list[str] | None:
    local_dir, run_dir = Path(args.local_dir).resolve(), Path(args.run_dir).resolve()
    newest = max(
        (c for c in (latest_checkpoint(run_dir), latest_checkpoint(local_dir)) if c is not None),
        key=lambda p: int(p.name),
        default=None,
    )
    if newest is None:
        if local_dir.exists():
            shutil.rmtree(local_dir)  # leftover from a run that never reached a checkpoint
        cmd = [
            "lerobot-train",
            f"--dataset.repo_id={args.repo_id}",
            f"--dataset.root={Path(args.dataset_root).resolve()}",
            f"--output_dir={local_dir}",
            f"--batch_size={args.batch_size}",
            f"--steps={args.steps}",
            f"--save_freq={args.save_freq}",
            f"--num_workers={args.num_workers}",
        ]
        if args.policy_path:
            cmd += [f"--policy.path={args.policy_path}", "--policy.push_to_hub=false"]
            rename_map = camera_rename_map(args.policy_path, args.dataset_root)
            if rename_map:
                cmd.append(f"--rename_map={json.dumps(rename_map)}")
        print(f"[vlapol] starting a new run from {args.policy_path or 'scratch'}", flush=True)
        return cmd + extra

    step = int(newest.name)
    if step >= args.steps:
        print(f"[vlapol] run already finished: checkpoint {newest.name} >= --steps {args.steps}", flush=True)
        return None
    local_ckpt = local_dir / CHECKPOINTS / newest.name
    if not (local_ckpt / TRAIN_CONFIG).is_file():
        shutil.rmtree(local_dir, ignore_errors=True)
        local_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(newest, local_ckpt)
    print(f"[vlapol] resuming from step {step} ({newest})", flush=True)
    return [
        "lerobot-train",
        f"--config_path={local_ckpt / TRAIN_CONFIG}",
        "--resume=true",
        f"--output_dir={local_dir}",
        f"--dataset.root={Path(args.dataset_root).resolve()}",
        f"--steps={args.steps}",
    ] + extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="persistent directory (e.g. on Google Drive)")
    parser.add_argument("--local-dir", type=Path, default=Path("outputs/train/vlapol_smolvla"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--policy-path",
        default="lerobot/smolvla_base",
        help="pretrained policy to fine-tune; pass '' and e.g. --policy.type=act to train from scratch",
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-freq", type=int, default=2_000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--keep", type=int, default=2, help="checkpoints to keep (locally and in --run-dir)")
    parser.add_argument("--sync-interval", type=float, default=30.0, help="seconds between checkpoint syncs")
    args, extra = parser.parse_known_args()

    cmd = build_command(args, extra)
    if cmd is None:
        return
    print("[vlapol] " + " ".join(cmd), flush=True)

    stop = threading.Event()

    def sync_loop() -> None:
        while not stop.wait(args.sync_interval):
            try:
                sync_checkpoints(args.local_dir, args.run_dir, args.keep)
            except OSError as err:  # e.g. Drive hiccup: retry on the next tick
                print(f"[vlapol] checkpoint sync failed, will retry: {err}", flush=True)

    syncer = threading.Thread(target=sync_loop, daemon=True)
    syncer.start()
    try:
        returncode = subprocess.call(cmd)
    finally:
        stop.set()
        syncer.join()
        sync_checkpoints(args.local_dir, args.run_dir, args.keep)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
