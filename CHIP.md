# CHIP compliance task

Implemented in the installed project at `active-adaptation/projects/mimic-lite`.
The integration repository's other `mimic-lite/` checkout is independent.

## Training contract

`task=chip` uses MimicLite PPO, the mode-15 G1 model, and MJLab. It ports the
BM/CHIP Precision three-point mechanism, with the requested changes:

- Physical compliance: left/right wrist `[0, 0.02] m/N`, torso zero.
- **No displacement clipping.** At 20 N and 0.02 m/N the virtual displacement
  is 0.4 m. This is a target displacement, not a guarantee of physical motion.
- Actor compliance input is `10*c`, so its range is `[0, 0.2]`, not the old
  BM range `[0, 0.5]`. Zero is nominal tracking, not infinite physical stiffness.
- Per-environment compliance holds for 100–199 control steps. Sampling assigns
  25% probability to all-zero wrists, 25% to both maximum values, and otherwise
  uses independent `U(0,1)^2 * c_max` for each wrist.
- Force peak magnitude is uniform `[0,20) N`; each body has an independent
  isotropic world-frame direction but bodies share the sampled peak magnitude.
  Ninety percent of cycles select one body with probabilities `[.15,.75,.10]`
  for left wrist/right wrist/torso; ten percent activate all three.
- Force duration is 50–99 control steps, with 100 waiting steps. The envelope
  ramps up during the first quarter, holds during the middle half, and ramps
  down during the final quarter. The global curriculum is 500 warmup steps,
  followed by a 5000-step linear ramp. Control frequency is 50 Hz.
- Force and compliance are prepared before the corresponding actor observation;
  the same world-frame force is reapplied at every physics substep. Reset clears
  force only for reset environments, resamples their schedules, and preserves
  the global curriculum. This avoids stale-force/reset leakage.

For reference point position `p_ref_local`, actor position is
`p_ref_local - R_robot_anchor.T @ (c * F_world)`. The reference uses its own full
pelvis orientation, matching BM's convention rather than MimicLite yaw-only
coordinates. Reference and measured points use local link offsets:
left wrist `[.18,-.025,0]`, right wrist `[.0719,-.003,0]`, torso `[0,0,.35]` meters.
The right point replaces the BM 18 cm point: it lies 30.4 mm (5+25.4 mm)
along local +X beyond the wrist mounting face `[.0415,-.003,0]`, not the COM.
The physical force point, actor reference, measured point and tracking rewards
all use this same offset. Payload mass locations are independent and unchanged.
MJLab receives force at each body's center of mass plus the moment
`(point_world - COM_world) cross F_world`.

Rewards compare actual points with **unmodified reference points**, never with
the virtual targets. Normalized point weights are `[2,4,1]`. Position terms use
sigma .10/.04/.03 m and weights 2/2/2; orientation terms use sigma .40/.15/.10 rad
and weights .50/.25/.25. The original .04 m and .15 rad terms are preserved;
the .03 m and .10 rad terms are additive, not replacements. This increases the
total tracking reward weight. An additional right-hand position term uses point weights
`[0,1,0]`, sigma .03 m, and reward weight 1.0. It tracks the same offset point
in the anchor-local frame, not the wrist origin or a world-space target.
The formula is `exp(-weighted_mean(error_squared) / sigma**2)`; sigma is a
reward scale, not a guaranteed tracking tolerance. Broad terms remain to
provide learning signal at larger errors. These tighter scales and the new
right-hand weight require training validation, including compliance response.
Lower-body/root tracking and regularization remain, but there
is no competing upper-arm reference-joint reward. Critic-only inputs contain
the original points and external force.

## Actor interface

MimicLite PPO concatenates `policy` then `command`:

- `policy`: 930 values, term-major gravity/angular velocity/joint position/
  joint velocity/previous raw policy action, 10 frames **oldest to newest**.
  Joint order follows MimicLite's simulation action order.
- `command`: 54 values: 12 lower-body q, 12 lower-body dq, 9 virtual point xyz,
  12 reference point quaternions (wxyz), 6 robot-to-reference anchor rotation
  matrix entries (`matrix[..., :2].flatten()`), 3 scaled compliance values.

This totals 984 values and produces 29 actions. **It is not binary compatible
with a BM checkpoint or deployment observation vector**, despite equal size.
MimicLite action scales/delay/filter, model, PPO, and normalization are retained.
Neither raw force nor unshifted upper-body reference joints are actor inputs.
CHIP has no extra action-delta limiter. Ordinary simulator joint/actuator limits
and action smoothing still apply.

CHIP now uses `g1-mode_15-chip-payload`: the original mode-15 geometry plus
two added point masses on right_wrist_yaw_link. The URDF right_hand_palm_joint
mounting reference is [.0415,-.003,0] m. Along local +X, .237 kg is placed
2.5 mm beyond it and .280 kg at 17.5 mm: positions [.044,-.003,0] and
[.059,-.003,0] m. The original .170 kg rubber-hand mass, first moment and
inertia contribution are subtracted using the stock mode-15 URDF hand inertial
parameters. Visual/collision geometry remains unchanged; this is an inertial
replacement, not a complete mechanical/collision model of the new end effector.
Combined wrist-link mass becomes .601576 kg (was .254576 kg; net +.347 kg). The load's
own distributed inertia is unknown; only point-mass parallel-axis contributions
are modeled. Existing body inertia is rotated and recombined with the shifted
COM. No joints/bodies or payload collision geometry are added. Reference FK and
Reference body FK is unchanged; right-point offsets are applied at runtime.
The original `g1-mode_15` is untouched.
Deployment virtual-target
construction and force processing must be checked before reusing a controller;
the training formula alone does not prescribe the desired deployment force law.

## Run

From `MimicLite/active-adaptation`:

```bash
CHIP_MOTION_DIR=/absolute/path/to/any4hdmi-compatible/motions \
  bash projects/mimic-lite/scripts/train_chip.sh wandb.mode=disabled
```

Use an existing any4hdmi dataset (including its metadata), or its supported
legacy `motion.npz` + `meta.json` format. BM's loose NPZ directory does not
provide the name mapping/metadata required by this loader and must not be
silently interpreted using an assumed joint/body order. The default dataset
path is `MimicLite/any4hdmi/output/g1/lafan`; set `CHIP_MOTION_DIR` if elsewhere.

To reproduce original simultaneous-push sampling instead of Precision:

```bash
bash projects/mimic-lite/scripts/train_chip.sh \
  task.command.chip.multi_body_probability=1 \
  task.command.chip.warmup_steps=0 task.command.chip.ramp_steps=0
```

For uniform rather than Precision-biased compliance sampling, additionally set
`zero_probability=0`, `max_probability=0`, `sampling_power=1` under
`task.command.chip`. The explicit compliance and force ranges remain unchanged.

## Verification

### Accepted loco-manip dataset

Run `bash projects/mimic-lite/scripts/train_chip_loco.sh wandb.mode=disabled`
from the framework root. The launcher first runs `prepare_chip_loco.py`, then
selects generated `task=chip_loco_train`. Set `CHIP_LOCO_ROOT` for another
dataset location. The original dataset is never modified.

Preparation validates paths, duplicate/split leakage, exact coverage of the
14,038 accepted files, and common model/qpos/timestep metadata. It writes
pool-relative lists and task snapshots under `.cache/chip_loco/`. Snapshots
inherit the current `chip.yaml` values by copying that config at preparation
time; rerun preparation after changing rewards. The launcher does this every
time. Four pool weights are proportional to split clip counts (train:
4504/4085/233/2454), not equal-pool weighting or guaranteed frame proportions.
All pools use windowed loading (`full_motion: false`). This does not eliminate
the first-run FK build or CPU/disk cache cost for the full training split.

Generated `chip_loco_val` and `chip_loco_test` configs select held-out files;
they do not automatically launch evaluation or disable force/randomization.
Do not use those splits for optimization. The `_smoke` variants contain one
clip per pool and are integration fixtures, not evaluation sets.

```bash
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/prepare_chip_loco.py
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/validate_chip_loco_gt.py
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/smoke_chip.py --loco
```

Verified four real training clips: windowed MJLab execution, one PPO update,
984D actor input, force/virtual-target consistency and partial reset. An
independent CPU MuJoCo FK check at seven frames per clip matched loaded offset
points within 6e-7 m, verified joint-name mapping and quaternion conventions,
and recorded finite velocity/root-height statistics in
`.cache/chip_loco/gt_validation.json`. Sample peak joint speeds were about
11–13.5 rad/s; these are diagnostics, not a full-dataset smoothness certificate.
Only the four-clip FK caches have been built during verification. Full-data
cache building occurs on the first formal run; full training has not been run.

```bash
venv/mjlab/.venv/bin/python -m unittest discover -s projects/mimic-lite/tests -p test_chip.py -v
HF_HUB_OFFLINE=1 venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/smoke_chip.py
```

The smoke script creates a static FK fixture in a fresh temporary directory,
instantiates a real four-environment MJLab task, checks observations and forces,
and runs a PPO update. It is an integration test, not a trained compliance policy
or evidence of real-robot performance. Normal training needs real motion data.

Validated locally on RTX 4090 with Python 3.12.8, PyTorch 2.10.0+cu128,
MJLab 1.3.0, MuJoCo 3.7.0: six unit tests passed; four-environment smoke
passed with the task's dynamics randomizations and one finite PPO update.
The runtime check also verifies a full 0.4 m virtual offset, unchanged nominal
reward when force is edited, nonzero applied moment, and isolated reset clearing.

### 2026-09-08: current-framework and remote-launch fixes

CHIP components now follow the framework's deferred lifecycle: constructors
store configuration, `_initialize(env)` binds the environment and allocates
device tensors, and reset callbacks accept the reset TensorDict. This fixes
the missing `env` constructor error without changing force/reward semantics.
`tests/test_chip_lifecycle.py` covers construction and reset signatures.

The CHIP launch scripts default `ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0` to avoid
the FK-cache loader stall observed with forked workers after CUDA initialization.
This is a workaround, not a general multiprocessing fix; it does not change GT.
Direct Python/torchrun launches must set this variable too.

Launchers use `uv run --no-sync`: install the environment and editable project
packages explicitly before launch, rather than implicitly resynchronizing them
and potentially removing packages installed outside the environment lockfile.
Do not copy a virtual environment between machines: its Python links, entry
points and editable paths may be absolute. Recreate it at the destination,
refresh project discovery, rerun `prepare_chip_loco.py`, and provision the public
robot model cache before enabling offline mode. The remote `--locked` check
failed; `--frozen` was used to restore the existing lockfile, followed by editable
project installs. This does not mean the lockfile/dependency mismatch is fixed.

Remote validation used Python 3.11.15, PyTorch 2.11.0+cu128, MJLab 1.6.0,
and MuJoCo/MuJoCo-Warp 3.11.0. The original 9 CHIP tests and 2 lifecycle tests
passed. Four GPUs x 64 environments completed 50 PPO iterations on four clips;
one GPU x 4096 environments completed 12 iterations on the full training split.
The full training-split FK caches were built successfully on the remote machine.
These supersede the older small-fixture validation limits above, but do not
establish convergence, real-robot safety, or four-GPU x 8192 feasibility.
