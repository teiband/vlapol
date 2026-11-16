# vlapol

Testbed for VLA training and inference on limited hardware resources. It has two parts:

- [Simulation](#simulation-smolvla-on-a-mujoco-pick-and-place-task): fine-tune SmolVLA on a simulated
  pick-and-place task and evaluate it in MuJoCo.
- [Phone-based data collection](#phone-based-data-collection): record real demonstrations with an
  Android phone on a gripper mockup and export them as a LeRobot dataset.

## Simulation: SmolVLA on a MuJoCo pick-and-place task

Fine-tunes [SmolVLA](https://huggingface.co/lerobot/smolvla_base) (LeRobot) on a simulated
pick-and-place task and evaluates it closed loop in MuJoCo. There is no robot arm: the robot is a
floating parallel-jaw gripper with a wrist camera. Training and policy evaluation run on Google
Colab (free T4); the simulation and the scripted expert also run on a CPU-only machine.

### Layout

```
src/vlapol/
  assets/pick_place.xml   MuJoCo scene: table, cube, place target, floating gripper, cameras
  env.py                  PickPlaceEnv (reset / step / render)
  expert.py               scripted expert (uses privileged sim state)
  dataset.py              LeRobotDataset feature schema
  collect.py              demo collection -> LeRobotDataset v3.0
  train.py                lerobot-train wrapper: checkpoint mirroring + resume
  evaluate.py             closed-loop evaluation, success rate + videos
notebooks/smolvla_finetune_colab.ipynb   the full pipeline on Colab
tests/                    env / expert / checkpoint-sync tests
```

### Task

| | |
|---|---|
| Task string | `pick up the {red,blue,yellow} cube and place it on the green target` (one per episode) |
| Observation | `observation.state` = TCP x, y, z, yaw, gripper opening; `observation.images.wrist` 256x256 (optional `scene` overview camera) |
| Action | 5-D in [-1, 1]: dx, dy, dz, dyaw (2 cm / 0.15 rad per step at full scale) and absolute gripper command (-1 closed, +1 open) |
| Control rate | 20 Hz |
| Randomized | cube position, yaw and colour, target position, gripper start pose |
| Success | cube at rest within 3 cm of the target centre, released, gripper retreated |

### Colab

Open `notebooks/smolvla_finetune_colab.ipynb` in Colab, select a T4 runtime and run it top to
bottom. It collects demos, fine-tunes and evaluates. The dataset, the two newest checkpoints and
the eval results are kept in `MyDrive/vlapol/`; after a disconnect, re-running the cells resumes
training from the newest checkpoint.

Set `BRANCH` in the first cell to the branch that holds this code.

### Local (Windows 11, CPU)

The env and the expert need only MuJoCo:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
python -m vlapol.evaluate --expert --episodes 10 --output-dir outputs/expert
```

Collecting demos or loading a policy also needs LeRobot (`pip install -e ".[lerobot]"`, Python 3.12+):

```
python -m vlapol.collect --root data/vlapol_pickplace --episodes 100
python -m vlapol.evaluate --policy-path <checkpoint>/pretrained_model --episodes 10 --device cpu
```

### Commands

```
python -m vlapol.collect  --root DATASET --episodes N [--cameras wrist scene] [--resume]
python -m vlapol.train    --dataset-root DATASET --run-dir PERSISTENT_DIR [--steps N] [lerobot-train flags]
python -m vlapol.evaluate --policy-path CKPT/pretrained_model | --expert  [--episodes N] [--output-dir DIR]
```

`vlapol.train` trains into a local directory and copies each finished checkpoint to `--run-dir`.
It does this because Google Drive does not support the `checkpoints/last` symlink that
`lerobot-train` creates. Running the same command again resumes from the newest checkpoint.

## Phone-based data collection

This part collects demonstrations **without a robot**. An Android phone is mounted on a mockup of the
UMI gripper and you move the mockup by hand. ARCore on the phone tracks the camera in metric 6-DoF (visual-inertial
odometry, no markers in the scene). Two small ArUco tags on the mockup's fingers give the gripper
opening. The PC records the stream, converts it into relative end-effector actions and exports a
[LeRobot](https://github.com/huggingface/lerobot) dataset, e.g. for fine-tuning SmolVLA.

```
 Android phone (ARCore)            PC (python, this package)
 ┌──────────────────────┐  WS   ┌──────────────────────────────────────────────┐
 │ camera image (JPEG)  │──────▶│ vlapol-record   → data/raw/<session>/episode_*│
 │ camera pose (VIO)    │ USB or│ vlapol-replay   → overlay of GT action chunks │
 │ intrinsics, t        │ Wi-Fi │ vlapol-export-lerobot → LeRobotDataset        │
 └──────────────────────┘       └──────────────────────────────────────────────┘
```

### Layout

| Path | What |
|---|---|
| `android/` | Kotlin ARCore streamer app (open in Android Studio) |
| `configs/rig.yaml` | Camera→TCP extrinsic, finger-tag config, export rate/size |
| `src/vlapol/phone/` | Wire protocol, WebSocket receiver, phone simulator |
| `src/vlapol/record.py` | Recording UI |
| `src/vlapol/gripper.py` | Gripper opening from finger tags |
| `src/vlapol/process.py` | Poses → fixed-rate relative actions |
| `src/vlapol/viz.py` | Draws an action chunk as a 3D trajectory on the image (used for replay and later for live eval) |
| `src/vlapol/export_lerobot.py` | LeRobot export |

### Setup

```bash
pip install -e ".[phone,dev]"
pip install -e ".[lerobot]"  # only needed for the export step
```

LeRobot pulls in `opencv-python-headless`, which replaces the GUI build of OpenCV. If the recording
window then fails with `cv2.imshow ... not implemented`, restore the GUI build:

```bash
pip install --force-reinstall --no-deps opencv-python
```

#### 1. Phone app

You need an [ARCore-supported phone](https://developers.google.com/ar/devices) with
*Google Play Services for AR* installed (from the Play Store; the app prompts for it).

Build and install over USB. This needs USB debugging enabled, and Android Studio or the
Android SDK, with `local.properties` pointing at the SDK:

```bash
cd android && ./gradlew installDebug
```

Alternatively, run `./gradlew assembleDebug`, copy `app/build/outputs/apk/debug/app-debug.apk` to the phone
and open it there (you have to allow installing from unknown sources).

The app streams continuously to the URL in its text field. The default `ws://127.0.0.1:8765` is meant
for a USB connection with:

```bash
adb reverse tcp:8765 tcp:8765
```

Over Wi-Fi, enter `ws://<pc-ip>:8765` instead. Phone and PC must be on the same network, and the PC firewall has to
allow port 8765.

The streamed image is the **sensor-native landscape** CPU image (640×480 by default). Pose and intrinsics are
given in that same orientation, so phone screen rotation doesn't affect anything.

##### First test

```bash
vlapol-phone-check
```

Then tap *Connect* in the app. You should see about 30 fps and `TRACKING` once you move the phone a little.
`moved` should match the real distance you move it. To check: lay the phone down for a few seconds,
shift it 20 cm along a ruler and lay it down again. The tool then prints
`came to rest 20.0 cm from the previous rest position (turned 0.3 deg)`. Keep the lens uncovered
while it lies there, and slide the phone along the ruler without turning it.

If it stays `PAUSED`, the phone screen shows the reason (for example too little light or too few
features). The screen also shows the live camera image.

#### 2. Mount and tags

- Mount the phone **rigidly** on the mockup. The back camera should look along the gripper's approach
  direction and see both fingertips. Keep the fingertips at least about 10 cm from the lens, because phones can't
  focus closer than that.
- Print the finger tags at 100% scale (check the 50 mm bar with a ruler) and glue one onto each finger
  so they face the camera:
  ```bash
  vlapol-make-tags --out gripper_tags.png
  ```
- Calibrate the opening range: hold the mockup closed and press `c`, fully open and press `o`, then `s`:
  ```bash
  vlapol-calibrate-gripper
  ```
- Measure `T_cam_tcp` in `configs/rig.yaml`: the position of the fingertip midpoint in the camera frame
  (x right, y down, z forward, in meters). Rough millimeter accuracy is enough. Check it with step 4.

#### 3. Record

```bash
vlapol-record --out data/raw/cup_s1 --task "put the cup on the plate"
```

`SPACE` starts or stops an episode, `d` deletes the last one, `q` quits. Before each episode, move the
phone around a little so ARCore starts tracking. When tracking is lost, the preview shows a red warning.
During processing, only the longest tracked segment of an episode is kept.

#### 4. Check the labels

```bash
vlapol-replay data/raw/cup_s1/episode_000000 --chunk 10 --out replay.mp4
```

This draws the ground-truth future TCP trajectory on each frame. The white dot should sit between the
fingertips, and the line should follow where the gripper actually goes next. The line is green when the gripper
is open and red when it is closed. If the dot is off, fix `T_cam_tcp`.

#### 5. Export to LeRobot

```bash
vlapol-export-lerobot data/raw --repo-id local/umi_phone --root data/lerobot/umi_phone
```

| Feature | Shape | Meaning |
|---|---|---|
| `observation.images.wrist` | 480×640×3 video | Phone camera |
| `observation.state` | (2,) | Gripper opening (0 = closed, 1 = open) and raw finger-tag distance in m |
| `action` | (7,) | `[dx, dy, dz, drx, dry, drz, gripper]`: pose of the TCP at the next step, expressed in the current TCP frame (m, rotation vector in rad), plus the gripper opening at the next step |

Actions are **relative to the current TCP frame**, so they need no world frame or robot base. The same
prediction can be drawn as arrows on the live camera image or sent to a real arm's end-effector delta
controller. The export rate is `export.fps` in `rig.yaml` (default 10 Hz; ARCore streams at about 30 Hz).

### Testing without a phone

`vlapol.phone.sim` sends synthetic frames with a known trajectory and moving finger tags, using exactly the
app's wire format:

```bash
vlapol-record --out data/raw/sim --task test &
python -m vlapol.phone.sim --seconds 20
pytest
```
