from pathlib import Path

from vlapol.train import latest_checkpoint, list_checkpoints, sync_checkpoints


def make_checkpoint(run: Path, step: int) -> Path:
    ckpt = run / "checkpoints" / f"{step:06d}"
    (ckpt / "pretrained_model").mkdir(parents=True)
    (ckpt / "pretrained_model" / "train_config.json").write_text("{}")
    last = run / "checkpoints" / "last"
    if last.is_symlink():
        last.unlink()
    last.symlink_to(ckpt.name)
    return ckpt


def test_sync_copies_latest_and_prunes(tmp_path):
    local, drive = tmp_path / "local", tmp_path / "drive"
    assert sync_checkpoints(local, drive, keep=2) is None
    for step in (1000, 2000, 3000):
        make_checkpoint(local, step)
        sync_checkpoints(local, drive, keep=2)
    assert [p.name for p in list_checkpoints(drive)] == ["002000", "003000"]
    assert [p.name for p in list_checkpoints(local)] == ["002000", "003000"]
    assert latest_checkpoint(drive).name == "003000"
    assert not (drive / "checkpoints" / "last").exists()  # no symlink on the Drive side


def test_incomplete_directories_are_ignored(tmp_path):
    (tmp_path / "checkpoints" / "001000.partial").mkdir(parents=True)
    (tmp_path / "checkpoints" / "002000").mkdir()  # no train_config.json yet
    assert latest_checkpoint(tmp_path) is None


def test_camera_rename_map_in_fresh_process(tmp_path):
    """Runs in a subprocess: the policy registry must be populated by vlapol.train itself."""
    import json
    import subprocess
    import sys

    import pytest

    pytest.importorskip("lerobot")
    (tmp_path / "ds" / "meta").mkdir(parents=True)
    features = {"observation.images.wrist": {"dtype": "video", "shape": [256, 256, 3]}}
    (tmp_path / "ds" / "meta" / "info.json").write_text(json.dumps({"features": features}))
    script = f"""
from vlapol.train import camera_rename_map
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
cams = {{f"observation.images.camera{{i}}": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)) for i in (1, 2)}}
SmolVLAConfig(input_features=cams, device="cpu").save_pretrained(r"{tmp_path / "base"}")
"""
    subprocess.run([sys.executable, "-c", script], check=True)
    check = f'from vlapol.train import camera_rename_map as f; print(f(r"{tmp_path / "base"}", r"{tmp_path / "ds"}"))'
    out = subprocess.run([sys.executable, "-c", check], check=True, capture_output=True, text=True).stdout
    assert "'observation.images.wrist': 'observation.images.camera1'" in out


def test_resume_command_keeps_save_freq(tmp_path):
    import argparse

    from vlapol.train import build_command

    make_checkpoint(tmp_path / "drive", 2000)
    args = argparse.Namespace(
        local_dir=tmp_path / "local",
        run_dir=tmp_path / "drive",
        dataset_root=tmp_path / "ds",
        steps=20000,
        save_freq=500,
        num_workers=2,
    )
    cmd = build_command(args, ["--policy.device=cuda"])
    assert "--resume=true" in cmd and "--save_freq=500" in cmd and cmd[-1] == "--policy.device=cuda"
    assert (tmp_path / "local" / "checkpoints" / "002000" / "pretrained_model" / "train_config.json").is_file()
