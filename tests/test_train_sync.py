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
