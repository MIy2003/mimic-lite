# Six-dimensional compliance for native MimicLite CHIP

This extends the existing **three upper-body points plus lower-body joints**
reproduction. It does not use HELM's goal-token network or pelvis/two-hand
command. The original actor/critic architecture, 930-value proprioceptive
history, lower-body joint references, nominal-reference rewards, payload, and
29-action interface remain in use. `chip.yaml` still selects the existing
`wrist_axis` mode; the new task is `chip_6d`.

## Command contract

| Slice | Meaning |
|---|---|
| 0:12 | lower-body joint position references |
| 12:24 | lower-body joint velocity references |
| 24:33 | left/right/torso hardware-point positions, anchor-relative |
| 33:45 | left/right/torso orientations, wxyz, anchor-relative |
| 45:51 | relative anchor orientation, first two matrix columns |
| 51:57 | left hand `[cx,cy,cz,crx,cry,crz]` |
| 57:63 | right hand, same six-channel order |
| 63:69 | torso: six zeros |

This gives a 69D command and 999D combined actor input (930+69). The CHIP
privileged term is 57D: nominal point poses (21), force (9), compliance (18),
pure couple (9). Other existing privileged terms remain unchanged. Pelvis is the
anchor, **not** a replacement for the torso reference point.

Physical maxima are `[.02,.02,.02,.2,.2,.2]`, with m/N for translation and
rad/(N m) for rotation. Multipliers `[50,50,50,5,5,5]` map maxima to one.
Left-hand compliance is always zero in the current task; external disturbances
remain enabled on the left. Zero is a maximum-stiffness target, not a guarantee
of zero physical displacement.

## Reference and load geometry

For each hand, R is the **actual wrist** rotation, A is the measured pelvis
rotation, f is world force at hardware point P, and mu is world pure couple:

```
dp_world = R diag(c_xyz) R^T f
dr_world = R diag(c_rxyz) R^T mu
p_actor_local = p_reference_local - A^T dp_world
R_actor_local = Exp(-A^T dr_world) R_reference_local
```

The nominal references returned by `chip_reference()` are never modified.
Positions are already hardware-point positions, so changing orientation must
not rotate the mounting offset again. A pure couple leaves the position target
unchanged. Every physics substep applies force and `mu + (P-COM) cross f`; only
mu drives angular hindsight. Torso reference targets remain nominal.

The sampler and geometry helpers are migrated from HELM CHIP commit `ffa66b1`,
but observation packing and rewards are native MimicLite. Force/couple/mixed
event probabilities are 40/40/20%; force maximum 20 N, pure couple maximum
5 N m, with the existing envelope/curriculum. 80% of couple directions follow
signed current-wrist axes, 20% remain fixed random world directions. Force
directions remain fixed random world directions. Axes are selected independently
of compliance, exposing both soft and stiff directions to disturbances.

Translation and rotation separately choose single-x, single-y, single-z,
all-three-same, all-zero (16% each), or independent axes (20%). Structured active
axes share a positive level: 50% maximum, otherwise U^2 times maximum. Each
group has its own level; x/rx are not tied. Independent axes retain the original
25% zero / 25% max / 50% continuous distribution. A separate 25% branch makes
all hand channels zero. Probabilities above are conditional on not taking this
joint rigid override.

Metrics add `right_wrist_torque`, `right_wrist_angular_compliance`, and
`right_wrist_ori_error`. The existing compliance metric becomes the mean of
the three physical translation commands. These command metrics are not measured
closed-loop compliance. Rewards retain the current local repository settings.

## Training and validation

From `/home/yangmin/code/MimicLite`:

```bash
# Existing four-pool data, native MimicLite architecture; fresh training.
CHIP_NUM_ENVS=4096 CHIP_TOTAL_ITERS=30000 \
  bash mimic-lite/scripts/train_chip_6d_loco.sh checkpoint_path=null

# A single motion directory instead of the four-pool dataset.
CHIP_MOTION_DIR=/absolute/motion_directory \
  bash mimic-lite/scripts/train_chip_6d.sh checkpoint_path=null
```

The four-pool preparation writes separate `.cache/chip_6d_loco` configs so the
old scalar/axis prepared tasks remain available. Both the top-level
`mimic-lite` source and `active-adaptation/projects/mimic-lite` runtime copy
contain this migration. Keep them synchronized when editing either copy.

On Lambda, from `/share/ml/yangmin/mimiclite-chip/active-adaptation`:

```bash
sbatch --export=ALL,NUM_ENVS=16384,TOTAL_ITERS=30000 \
  projects/mimic-lite/scripts/train_chip_6d_lambda_4gpu.slurm
```

This allocates four GPUs, with 16,384 environments on **each** GPU (65,536
total). It first runs six full-size smoke updates, then starts fresh production
training with offline W&B logging. Logs, W&B data, and Hydra configuration go
under `../records/chip-6d-${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}`. Slurm
selects the GPU devices. `MIMICLITE_ROOT` overrides the installation root;
`RUN_SMOKE=0` skips the capacity check for an already validated configuration.

From `active-adaptation`, using its MJLab Python:

```bash
PYTHONPATH=projects/mimic-lite:.:../any4hdmi/src \
  venv/mjlab/.venv/bin/python -m unittest discover \
  -s projects/mimic-lite/tests -p 'test_chip*.py' -v

PYTHONPATH=projects/mimic-lite:.:../any4hdmi/src HF_HUB_OFFLINE=1 \
  venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/smoke_chip.py \
  --mode wrist_6d --event mixed
```

Use `--event force` or `--event couple` for pure-load checks, `--loco` for the
prepared four-pool smoke tasks, and `--mode wrist_axis` for legacy regression.
Smoke tests use four environments and a small test network; they do not measure
learned compliance quality or full-size training capacity.

## Checkpoints and export

69D models require new training. Cross-mode resumes from scalar 54D or axis 57D
checkpoints are rejected. Six-dimensional resume also checks saved physical
bounds, normalization scales, and left-hand mode. Replay restores those
semantics, while replay load/rigid overrides remain effective. Default replay
and export use no external force/couple and all-rigid commands.

ONNX export uses format `mimic_lite_chip_6d_v3` and records point order, six-axis
units/scales/limits, and left-hand training mode. The existing HIL deployment
adapter for v1/v2 is **not** a 69D adapter; this migration does not update robot
deployment or change an existing deployed controller. Validation records are
under `/home/yangmin/code/MimicLite/records/chip-6d-migration/`.

Migration validation (2026-09-27): 34 unittest cases and two component-lifecycle
checks passed. Four-environment GPU tests completed finite PPO updates for
force-only, couple-only (using prepared four-pool data), mixed loads, and the
legacy wrist-axis task. The actual four-pool shell launcher passed Hydra config
composition. A test-only checkpoint successfully reloaded and exported to ONNX
with 930/69 input shapes, 29 actions, v3 metadata, and an ONNX Runtime inference.
The small test checkpoint is not a trained controller for deployment.

## 2026-09-28: release-recovery training configuration

`chip_6d` now tracks the BM link selection: bilateral hip_roll, knee,
ankle_roll, shoulder_roll, elbow and wrist_yaw, plus pelvis and torso.
The shared lower-body selection uses hip_roll/knee/ankle_roll as well.
Whole-body position/orientation reward weights change from 1.0 to 0.25;
whole-body linear/angular velocity reward weights change from 1.0 to 0.10.
Endpoint position/orientation rewards, root and lower-joint tracking, action
regularization, compliance ranges and the MuJoCo cvel representation are unchanged.
This is a training hypothesis for load-release recovery, not a validated fix.
The scalar/axis base task is unchanged. Existing saved checkpoint configurations
and running jobs retain their original behavior; regenerate the loco config with
`train_chip_6d_loco.sh` for the new settings. The body selection also propagates
to body-based privileged observations; matching tensor dimensions alone does not
make old critic weights semantically compatible with the new selection.

## 2026-09-28: small-torque angular compliance variant

Current angular compliance range is [0, 0.8] rad/(N m), with input scale 1.25;
translation remains [0, 0.02] m/N with scale 50. This supersedes the earlier
0.2 angular limit in this document. Bounds must come from each checkpoint's
saved configuration during export/deployment, not from today's task defaults.

Peak torque norm uses uniform bands [0.1,0.6], [0.6,1.0], [1.0,2.0] N m with
probabilities 0.70/0.20/0.10. Axis/sign sampling remains 80% signed actual wrist
axes, 20% world-random. Zero loads remain covered by force-only events and the
unloaded pulse intervals. The pulse timing, motor PD and observation noise are
unchanged in this variant.

For actual wrist compliance C and unit torque direction d, peak amplitude is
sampled from the above distribution conditioned on A <= (pi/4)/||C d||.
Zero-compliance directions retain the full amplitude range. The conditional
inverse CDF preserves an event's random quantile; it is reevaluated each physics
substep as the wrist frame changes and after compliance resampling. Thus the
actual band fractions differ from 70/20/10 in compliant directions. The applied
physical torque and the hindsight computation share the same bounded value;
we do not independently clip the virtual orientation. The 45-degree bound is a
rotation-vector norm bound, not a guarantee of arm IK feasibility.

The scheduler's optional parameters default to the previous sampler so old saved
configs remain usable. This change needs retraining; do not apply 0.8 commands or
scale 1.25 to the old 538/11500 actor. The modified BM link selection and reduced
body tracking reward weights from the previous section are retained.
