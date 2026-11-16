import threading
import time

import numpy as np
import pytest

from vlapol.episode_io import Episode, EpisodeWriter, list_episodes
from vlapol.geometry import arcore_pose_to_T_world_cam, inv, relative, T_to_vec6, vec6_to_T
from vlapol.gripper import GripperTagDetector, fill_gaps
from vlapol.phone import sim
from vlapol.phone.protocol import decode, encode
from vlapol.phone.server import PhoneServer
from vlapol.process import process_episode, resample_indices
from vlapol.rig import GripperConfig, Rig
from vlapol.viz import chunk_to_tcp_poses, draw_action_chunk

RIG = Rig.from_dict({
    "T_cam_tcp": {"translation": [0.0, 0.04, 0.15], "rotvec": [0.0, 0.0, 0.0]},
    "gripper": {"tag_size_m": 0.015, "closed_dist_m": 0.02, "open_dist_m": 0.08},
    "export": {"fps": 10, "image_size": [480, 640]},
})


def test_vec6_roundtrip():
    v = np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.4])
    assert np.allclose(T_to_vec6(vec6_to_T(v)), v)


def test_protocol_roundtrip_and_pose_convention():
    header, jpeg = sim.make_frame(3, 1.5, GripperConfig(), sim._texture())
    f = decode(encode(header, jpeg))
    assert f.seq == 3 and f.tracking_ok and f.image().shape == (480, 640, 3)
    assert np.allclose(arcore_pose_to_T_world_cam(f.pose7), sim.camera_pose(1.5), atol=1e-6)


def test_gripper_tag_distance():
    cfg = GripperConfig()
    det = GripperTagDetector(cfg)
    bg = sim._texture()
    for t in [0.0, 1.0, 2.0, 3.0]:
        obs = det.detect(sim.render(t, cfg, bg), sim.K)
        assert obs.dist_m is not None
        assert abs(obs.dist_m - sim.gripper_dist(t, cfg)) < 0.002  # < 2 mm


def test_fill_gaps():
    v = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    assert np.allclose(fill_gaps(v), [1, 1, 2, 3, 3])


def test_resample():
    t = (np.arange(0, 3.0, 1 / 30) * 1e9).astype(np.int64)
    idx = resample_indices(t, 10)
    assert len(idx) == 30 and np.all(np.diff(idx) == 3)


def _write_sim_episode(tmp_path, seconds=4.0, fps=30, lost=()):
    cfg = RIG.gripper
    bg = sim._texture()
    w = EpisodeWriter(tmp_path / "session", "pick up the cup")
    for i in range(int(seconds * fps)):
        h, j = sim.make_frame(i, i / fps, cfg, bg, tracking="PAUSED" if i in lost else "TRACKING")
        w.add(decode(encode(h, j)))
    return w.close()


def test_actions_reconstruct_trajectory(tmp_path):
    ep_dir = _write_sim_episode(tmp_path)
    pe = process_episode(ep_dir, RIG)
    assert pe.actions.shape == (len(pe.frame_idx), 7)
    # integrating the relative actions must reproduce the TCP trajectory
    poses = chunk_to_tcp_poses(pe.actions[:-1])
    for k in range(len(poses)):
        assert np.allclose(pe.T_world_tcp[0] @ poses[k], pe.T_world_tcp[k], atol=1e-6)
    # ground-truth TCP trajectory from the simulator
    gt = sim.camera_pose(pe.t[5]) @ RIG.T_cam_tcp
    assert np.allclose(relative(pe.T_world_tcp[0], pe.T_world_tcp[5]), inv(sim.camera_pose(0) @ RIG.T_cam_tcp) @ gt, atol=1e-6)
    # gripper labels match the simulated opening
    expected = [(sim.gripper_dist(t, RIG.gripper) - 0.02) / 0.06 for t in pe.t]
    assert np.max(np.abs(pe.gripper - np.clip(expected, 0, 1))) < 0.05


def test_tracking_loss_uses_longest_segment(tmp_path):
    ep_dir = _write_sim_episode(tmp_path, lost=set(range(30, 40)))
    pe = process_episode(ep_dir, RIG)
    assert pe.frame_idx.min() >= 40


def test_draw_does_not_crash(tmp_path):
    ep_dir = _write_sim_episode(tmp_path, seconds=2)
    pe = process_episode(ep_dir, RIG)
    ep = Episode(ep_dir)
    img = ep.image(0)
    before = img.copy()
    draw_action_chunk(img, pe.K, RIG.T_cam_tcp, pe.actions[:10], label="x")
    assert (img != before).any()


def test_server_end_to_end(tmp_path):
    port = 8799
    with PhoneServer(host="127.0.0.1", port=port) as srv:
        th = threading.Thread(target=sim.run, args=(f"ws://127.0.0.1:{port}", 1.0, 30.0))
        th.start()
        th.join()
        time.sleep(0.2)
        frames = srv.drain()
    assert len(frames) == 30
    assert [f.seq for f in frames] == list(range(30))


def test_record_headless(tmp_path):
    from vlapol import record

    port = 8798
    out = tmp_path / "rec"
    th = threading.Thread(target=record.main, args=([
        "--out", str(out), "--task", "wave", "--port", str(port),
        "--headless", "--episodes", "2", "--episode-seconds", "1.0"],))
    th.start()
    time.sleep(0.5)
    sim.run(f"ws://127.0.0.1:{port}", 3.0, 30.0)
    th.join(timeout=10)
    eps = list_episodes(out)
    assert len(eps) == 2
    for e in eps:
        ep = Episode(e)
        assert ep.task == "wave" and 20 <= len(ep) <= 40
        process_episode(e, RIG)


def test_export_lerobot(tmp_path):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from vlapol.export_lerobot import export

    _write_sim_episode(tmp_path, seconds=2)
    n = export(tmp_path / "session", "local/test", str(tmp_path / "lr"), RIG)
    assert n == 1
    ds = LeRobotDataset("local/test", root=str(tmp_path / "lr"))
    x = ds[3]
    assert tuple(x["observation.state"].shape) == (2,)
    assert tuple(x["action"].shape) == (7,)
    assert tuple(x["observation.images.wrist"].shape) == (3, 480, 640)
    assert x["task"] == "pick up the cup"
