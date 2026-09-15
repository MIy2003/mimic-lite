# CHIP compliance task

Implemented in the installed project at `active-adaptation/projects/mimic-lite`.
The integration repository's other `mimic-lite/` checkout is independent.

## Training contract

`task=chip` uses MimicLite PPO, the mode-15 G1 model, and MJLab. It ports the
BM/CHIP Precision three-point mechanism, with the requested changes:

- Physical compliance: left/right wrist `[0, 0.02] m/N`, torso zero.
- **No displacement clipping.** At 20 N along the axis and 0.02 m/N the virtual displacement
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

The default is now `compliance_mode: wrist_axis` (v2). For each point, rotate the
shared unit vector `u_local` by its **actual current link orientation**:
`u_world = R_point_link @ u_local`. The two wrists interpret the same three
numbers in their respective `left_wrist_yaw_link` / `right_wrist_yaw_link` frames,
so their world axes can differ. This is NOT a reference-wrist, pelvis, or COM
inertial-frame axis. Torso uses torso_link's frame but its default c is zero.

Only `F_parallel = dot(F_world, u_world) * u_world` enters the virtual target:
`p_ref_local - R_robot_anchor.T @ (c * F_parallel)`.
The legacy `compliance_mode: isotropic` instead uses all of `F_world`.
Both modes physically apply **all of F_world**, including perpendicular force;
neither the applied force nor its lever-arm torque is projected. Rewards still
track the unmodified reference. Perpendicular force thus trains disturbance
rejection, not compliant displacement. Zero target displacement is not a
guarantee of zero real motion or infinite stiffness. Rotation rewards are unchanged.

The reference uses its own full
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
- `command`: 57 values in wrist_axis, 54 in legacy isotropic: 12 lower-body q,
  12 lower-body dq, 9 virtual point xyz,
  12 reference point quaternions (wxyz), 6 robot-to-reference anchor rotation
  matrix entries (`matrix[..., :2].flatten()`), 3 scaled compliance values,
  then **3 unscaled local unit-axis values** (`command[54:57]`) in wrist_axis only.

This totals **987** values (legacy: 984) and produces 29 actions. The CHIP-only
privileged term also appends the same three axis values (33 to 36 values);
the critic retains full, unprojected external force. **It is not binary compatible
with a BM checkpoint or deployment observation vector**, despite equal size.
MimicLite action scales/delay/filter, model, PPO, and normalization are retained.
Neither raw force nor unshifted upper-body reference joints are actor inputs.
CHIP has no extra action-delta limiter. Ordinary simulator joint/actuator limits
and action smoothing still apply.

## Wrist-axis sampling and checkpoint compatibility (v2)

The name `stiffness_axis` means **soft along the axis, stiff perpendicular to it**.
It is an unoriented axis: u and -u yield the same projector, and negative axial
force is preserved (no rectification). Random sampling covers the +X hemisphere;
it does not need duplicate samples of the opposite hemisphere.

| Probability | Angle from wrist-local +X |
| --- | --- |
| 20% | exactly [1,0,0] |
| 60% | 0 to 30 degrees |
| 15% | 30 to 60 degrees |
| 5% | 60 to 90 degrees |

Within each band sample cos(theta) uniformly between the band cosines and phi
uniformly in [0, 2*pi); output [cos(theta), sin(theta)*cos(phi), sin(theta)*sin(phi)].
This is uniform solid angle within each band, not uniform Euler angles. The
probabilities are an initial task-specific choice, not paper defaults or an
empirically optimal distribution.

Axes are sampled on reset and at the start of each new force cycle when force
is zero, then held in the **wrist frame** through the pulse/wait cycle (150–199
steps by default). World axes still rotate with the robot. Compliance resampling
does not resample axes. Physical force direction stays world-isotropic and
independent of the axis; amplitude, body selection, c sampling and curriculum
remain unchanged.

Configuration in `cfg/task/chip.yaml`:

```yaml
compliance_mode: wrist_axis
axis_probabilities: [0.20, 0.60, 0.15, 0.05]
axis_cone_degrees: [30.0, 60.0, 90.0]
fixed_stiffness_axis: null  # random; [1,0,0] fixes each wrist's local +X
```

Fixed axes are normalized; zero/nonfinite vectors are rejected. To use different
LOCAL axes for the two hands would require an expanded interface; this version
shares three local numbers and does not sample each hand's local axis independently.

New training uses wrist_axis by default; **start with checkpoint_path=null**.
The local scripts do not migrate old weights to the larger input. A cross-mode
resume fails before constructing the simulator. To intentionally resume a legacy
checkpoint, set `task.command.chip.compliance_mode=isotropic`.

Replay, diagnostics, validation and export read the checkpoint's sibling cfg.yaml
to restore its mode; a missing mode means legacy isotropic. Thus old checkpoints
remain 54D even though chip.yaml now defaults to 57D. Keep cfg.yaml with each
checkpoint. Replay fixes the axis at local +X by default; override
`task.command.chip.fixed_stiffness_axis=null` for random axes on a v2 checkpoint.
Re-run `prepare_chip_loco.py` / `prepare_chip_episode.py` to refresh generated
task YAMLs after this change; motion data and FK need no conversion. The loco
training launcher already refreshes its task configs.

`export_chip_deploy.py` emits `mimic_lite_chip_v2` for the new interface, measured
input dimensions and axis-frame metadata. **The existing external v1 deployment
adapter is not upgraded or replaced by this change.** Do not substitute a 57D
model into a 54D adapter. `validate_chip_deploy.py` explicitly rejects v2 until
that adapter is updated. No extra force input is added to the actor.

New diagnostic metrics: `right_wrist_force_parallel/perpendicular` (N) and
`right_wrist_error_parallel/perpendicular` (m). They use the last applied axis
to avoid attributing the prior transition to a newly sampled axis. Errors remain
relative to the original reference, not an identified real-world compliance.
Like existing CHIP metrics, raw episode stats accumulate values: divide by
episode_len to obtain an episode mean. Metrics have no reward contribution.

### Reproducible checks

From the framework root, with its MJLab environment:

```bash
venv/mjlab/.venv/bin/python -m unittest discover -s projects/mimic-lite/tests -p 'test_chip*.py'
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/smoke_chip.py --mode wrist_axis
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/smoke_chip.py --mode isotropic
```

The bounded four-environment smoke checks finite observations/rewards, 57D/54D
commands, full physical wrench (including perpendicular forces and substep
reapplication), unclipped virtual targets, unchanged tracking rewards, partial
reset isolation and one PPO update. This is an integration check, not evidence
that an untrained policy has learned directional compliance.

Local validation (2026-09-14, RTX 4090): 23 unittest checks and two lifecycle
checks passed; both wrist_axis and legacy isotropic four-env smoke runs passed,
including one PPO update each. With seed 417 and 50,000 sampled axes, the four
bands contained 20.226%, 59.806%, 14.886%, 5.082%, respectively. Logs are under
`active-adaptation/records/chip_axis/`. The first smoke attempt exposed the old
test harness's tensor-only reward check; it was updated to handle the current
TensorDict reward groups before both integration runs passed. No full training,
trained v2 quality evaluation, or real-robot deployment was performed.

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

### Replay the trained CHIP controller (2026-09-09)

2026-09-11 rollback: replay defaults are restored to
`MimicLite/checkpoints/dbiroz1a-18500/checkpoint_18500.pt` and its sibling cfg.yaml.
The desktop 4k checkpoint is retained for optional comparisons. No controller
implementation or dependency rollback was needed for this model switch.

Earlier 2026-09-11 comparison used
`MimicLite/checkpoints/desktop-4000/checkpoint_4000.pt`, copied from the user's
Desktop. Its sibling `cfg.yaml` was extracted from its embedded training cfg,
not copied from 18.5k: actor hidden dimensions are [512,512,512] and critic
dimensions [1024,512,256]. Strict policy loading passed. A 12-second offline
render on episode `g1_lowstate_20260810_170525` completed its 589-step episode
with motion_timeout (no pose-error termination), no external force, and summed
right wrist error 13.3678379 m, giving a mean of 0.0226958 m (2.27 cm).
Video: `active-adaptation/20260911-125424-f455c2e8.mp4`.
Older checkpoints remain available via CHIP_CHECKPOINT. This is one replay,
not proof of improved tracking or real-robot readiness.

Previous default: the resumed 18,500-iteration checkpoint
at `MimicLite/checkpoints/dbiroz1a-18500/checkpoint_18500.pt`, with its own
`cfg.yaml`. Both files were SHA256-verified against server 5090. The old 12k
checkpoint below is retained for comparison. Restart replay to load the new
weights; an explicitly set `CHIP_CHECKPOINT` takes precedence. This changes
local simulation replay defaults, not a running real-robot controller.

The completed run `dbiroz1a` (12,000 iterations) is stored locally under
`MimicLite/checkpoints/dbiroz1a/`, with `checkpoint_12000.pt` and its sibling
`cfg.yaml`. Keep these files together: the saved configuration supplies the
policy architecture. Checkpoints are ignored by Git and must be copied separately.

From `MimicLite/active-adaptation`, run:

```bash
bash projects/mimic-lite/scripts/replay_chip_loco.sh
# Enable the training-scale random external forces:
bash projects/mimic-lite/scripts/replay_chip_loco.sh task.command.chip.max_force=20
# Use the training split instead of held-out validation:
bash projects/mimic-lite/scripts/replay_chip_loco.sh task=chip_loco_train
```

For the recorded single episode (from the framework root):

```bash
bash projects/mimic-lite/scripts/replay_chip_episode.sh \
  /home/yangmin/data/drag_collection/flip_noft/chip/g1_lowstate_20260810_170525
```

This converts `chip_motion_50hz.npz` to a local `.cache/chip_episode/` dataset,
without editing the recording. It reorders joints by name, uses the recorded
pelvis position/WXYZ quaternion, and checks FK against recorded body positions.
The episode has 589 frames at 50 Hz (11.76 seconds between first/last frames).
Its wrist-yaw origins differ from the training XML by 5 mm; reference points
are regenerated using the training XML and current CHIP point offsets.
This follows the measured `q_real_smooth` trajectory, not the separate native
30 Hz three-point commands. `full_motion: true` and `start_from_zero: true`
select the entire single clip from its start, subject to normal early-failure
termination/reset. The default model is checkpoint_18500, with zero
external force. Override `CHIP_CHECKPOINT` to use another local model plus its
sibling cfg.yaml. The interactive viewer is at http://localhost:8080.
Conversion and Hydra composition were verified; actual policy rollout on this
episode has not yet been validated.

`cfg/replay_chip_loco.yaml` runs one simulated robot with the trained policy.
This is policy-driven tracking, NOT kinematic reference playback
(`replay_motion: false`). By default it samples validation motion windows,
starts at the beginning of each sampled window, and disables external force.
It does not play every full dataset clip sequentially. Compliance sampling and
dynamics randomization remain inherited from CHIP; this is not a fixed-parameter
benchmark. Force warmup/ramp are disabled for evaluation; the normal per-environment
force pulse/wait schedule remains active when `max_force` is nonzero.

The launcher regenerates local dataset paths and defaults to offline robot-cache
access. Override `CHIP_CHECKPOINT` and `CHIP_LOCO_ROOT` to relocate the model/data.
The right-hand reference/force point and payload remain those in `chip.yaml`.

Configuration composition was checked locally. The local environment was later
upgraded to MJLab 1.6.0 and MuJoCo/MuJoCo-Warp 3.11.0 to fix the missing
`BuiltinPdActuator` import; the full `aa.init()` registration/import path passed.
Python remains 3.12 and PyTorch remains 2.10, so this is not a complete clone of
the training environment. Full checkpoint rollout has not yet been verified.
The launcher uses `--no-sync` and does not automatically upgrade dependencies.
To reproduce this dependency fix from the framework root:

```bash
uv pip install --python venv/mjlab/.venv/bin/python \
  'mjlab==1.6.0' 'mujoco==3.11.0' 'mujoco-warp==3.11.0'
uv pip check --python venv/mjlab/.venv/bin/python
```

Known remaining optional-video dependency conflict: MJLab 1.6 requires
`mjviser>=0.0.14`, which requires Pillow >=12.2, whereas installed MoviePy 2.2.1
requires Pillow <12. The environment retains Pillow 12.3 for MJLab's viewer;
`uv pip check` reports this MoviePy conflict. Do not downgrade Pillow below 12
to hide it, or blindly downgrade MoviePy to an ancient release. The project
source does not directly import MoviePy; MoviePy-dependent video workflows
still need a separate compatible environment or a future compatible release.
