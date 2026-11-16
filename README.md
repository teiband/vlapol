# vlapol

Testbed for VLA training and inference on limited hardware resources.

Fine-tunes [SmolVLA](https://huggingface.co/lerobot/smolvla_base) (LeRobot) on a simulated
pick-and-place task and evaluates it closed loop in MuJoCo. There is no robot arm: the robot is a
floating parallel-jaw gripper with a wrist camera. Training and policy evaluation run on Google
Colab (free T4); the simulation and the scripted expert also run on a CPU-only machine.

## Layout

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

## Task

| | |
|---|---|
| Task string | `pick up the {red,blue,yellow} cube and place it on the green target` (one per episode) |
| Observation | `observation.state` = TCP x, y, z, yaw, gripper opening; `observation.images.wrist` 256x256 (optional `scene` overview camera) |
| Action | 5-D in [-1, 1]: dx, dy, dz, dyaw (2 cm / 0.15 rad per step at full scale) and absolute gripper command (-1 closed, +1 open) |
| Control rate | 20 Hz |
| Randomized | cube position, yaw and colour, target position, gripper start pose |
| Success | cube at rest within 3 cm of the target centre, released, gripper retreated |

## Colab

Open `notebooks/smolvla_finetune_colab.ipynb` in Colab, select a T4 runtime and run it top to
bottom. It collects demos, fine-tunes and evaluates. The dataset, the two newest checkpoints and
the eval results are kept in `MyDrive/vlapol/`; after a disconnect, re-running the cells resumes
training from the newest checkpoint.

Set `BRANCH` in the first cell to the branch that holds this code.

## Local (Windows 11, CPU)

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

## Commands

```
python -m vlapol.collect  --root DATASET --episodes N [--cameras wrist scene] [--resume]
python -m vlapol.train    --dataset-root DATASET --run-dir PERSISTENT_DIR [--steps N] [lerobot-train flags]
python -m vlapol.evaluate --policy-path CKPT/pretrained_model | --expert  [--episodes N] [--output-dir DIR]
```

`vlapol.train` trains into a local directory and copies each finished checkpoint to `--run-dir`.
It does this because Google Drive does not support the `checkpoints/last` symlink that
`lerobot-train` creates. Running the same command again resumes from the newest checkpoint.
