# Pelvis + two hardware-offset hands (Goal–Body Transformer)

This is an independent **pure tracking** controller, adapted from
`goal-body-transformer-training-20260914`, specifically
`tracking-pelvis-two-ee-v01` / `ppo/pelvis_two_ee_v01`. It is not the bundle's
default eight-mode Omni-condition task, and does not change CHIP's configuration,
checkpoints, deployment model, force scheduling or compliance axis.

An optional **isotropic CHIP variant** is now provided separately; see below.
The pure tracking task and the older `chip.yaml` are not replaced.

## 三点 Transformer + wrist-axis CHIP

新增独立入口 `train_three_point_chip_axis` / `replay_three_point_chip_axis`，
任务为 `three_point_chip_axis`。沿用三点 CHIP 的硬件点、奖励、外力和柔顺采样。

- command 为 **321 维**：315 维三点目标、3 维 `10*c`（pelvis/左/右）、
  3 维共享局部单位轴 `u_local`。每个 goal token 接收 109 维（105+1+3）。
- 与旧 wrist-axis CHIP 相同：`u_world = R_actual_link @ u_local`，左右腕分别
  使用实际 `left_wrist_yaw_link` / `right_wrist_yaw_link` 的姿态，非参考姿态或 COM。
- 虚拟手部目标为 `p_nominal - c * dot(F_world,u_world) * u_world`，
  全部 11 个目标时刻使用当前外力，不做位移限幅。pelvis 柔顺恒为零。
- 物理施力仍是完整外力及偏移点力矩；reward 仍跟踪原始参考。
- axis 复用旧 CHIP 的采样：以局部 +X 为中心，20% 精确 +X，
  60% 在 0–30°、15% 在 30–60°、5% 在 60–90°，各区间按立体角均匀采样。
  reset 和外力周期更新时采样；`fixed_stiffness_axis` 可固定并自动归一化。
- 新增左右手 `force_parallel` / `force_perpendicular`（N）和
  `error_parallel` / `error_perpendicular`（m），例如
  `reward.chip_metrics/right_wrist_error_parallel`。分量取模长；位置误差对照
  执行帧的原始硬件点参考，投影轴缓存自最后一个物理子步，避免下一周期轴混入。
  全部仅记录，不进入奖励。`reward.chip_metrics/...` 是逐步 EMA；
  `train/stats/chip_metrics/...` 是回合累加值，不是平均误差或平均力。
- policy 为 556 维、mask 为 3 维、输出 29 个关节动作。
  新增 critic 特权量共 15 维（完整力9、柔顺3、axis3）。
- 这是新网络输入契约；315D/318D/旧 CHIP checkpoint 不能直接加载。

从 `active-adaptation` 目录运行：

```bash
bash projects/mimic-lite/scripts/train_three_point_chip_axis.sh
# 固定局部 X 轴（默认则采样）：
bash projects/mimic-lite/scripts/train_three_point_chip_axis.sh \
  'task.command.chip.fixed_stiffness_axis=[1,0,0]'
# 回放默认无外力、零柔顺：
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/play.py \
  --config-name replay_three_point_chip_axis checkpoint_path=/path/to/checkpoint.pt
# 空闲 GPU 上进行小规模仿真和一次 PPO 更新：
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/smoke_three_point.py --axis
```

## 三点 Transformer + 无 axis CHIP

入口：`cfg/train_three_point_chip.yaml`、`cfg/task/three_point_chip.yaml`、
`cfg/replay_three_point_chip.yaml`。这是新模型，不能直接加载旧 54/57D CHIP
或纯三点 315D command 的 checkpoint；启动器默认从头训练。

### 目标与 reward 的配对

- 三点顺序仍为 **pelvis、左手、右手**；硬件偏移和右手 payload 保持下文约定。
- 策略输入的手部虚拟位置为 `p_nominal - c * F_world`，不做位移限幅，
  不投影 axis。pelvis 的 c 恒为零，姿态目标不随外力改变。
- 同一时刻的外力偏移应用到输入的全部 11 个时间点（含历史）以及当前位置误差，
  然后转换为当前 pelvis heading 坐标；不使用未来外力信息。
- **reward 仍对照执行帧的原始全身参考**，没有给 reward 的参考加减 cF，
  不生成新的全身 GT，也不解 IK。不能把虚拟输入目标再当成奖励目标。
- 保留 `three_point.yaml` 的全部 tracking/loco 项、权重、sigma，包括上肢
  body/关节跟踪。与 BM/CHIP 是同类训练思路，但不是完全相同的奖励配方。
- 额外增加 `tracking/chip_right_hand_pos`，weight=1、sigma=0.03 m：
  `exp(-||p_right_actual - p_right_nominal||² / 0.03²)`。
  计算的是**硬件偏移点的世界坐标误差**，不是腕关节原点或质心。
  未额外加入右手姿态精细项；姿态仍由原有全身奖励监督。

这里的“30 mm”是指数奖励的尺度，不是误差上限或精度保证。全身/上肢监督
可能影响柔顺学习；必须通过 c=0 tracking、外力—位移关系和卸力恢复测试验证。

### 输入、采样和施力

command = 原来的 315 维三点目标 + 3 维 `10*c`，共 **318 维**。
后 3 维按 pelvis/左/右顺序，Transformer 将对应 c 加到每个 goal token，
每个 goal token 输入变为 106 维。policy 历史仍为 556 维、mask 为 3 维，
输出仍是 29 个关节动作。critic 额外获得完整施力 9 维及柔顺系数 3 维。

沿用现有 CHIP 的采样参数：

- 两手 c 范围 `[0,0.02] m/N`：25% 同时为零、25% 同时最大、50% 各自独立 `0.02*U²`。
  每 100–199 个控制步重新采样。
- 外力周期峰值幅度 `U(0,20) N`；各施力点方向由归一化三维高斯采样，世界系球面均匀。
  同一环境同一周期共享峰值幅度，方向各点独立。
- 10% 多点模式；其余 90% 单点模式，左/右/躯干概率为 15%/75%/10%。
- 力持续 50–99 个控制步，间隔 100 步；梯形包络（前后各四分之一渐入/渐出）。
- 前 500 个控制步无力，然后 5000 步线性增大到完整幅值。这是**控制步**，不是 PPO iteration；
  是同一训练中的外力课程，不是两个独立训练阶段。c 在无力阶段仍采样。
- 物理施力点顺序仍是左手/右手/躯干，与控制目标的 pelvis/左手/右手顺序不同；
  躯干仅施加扰动，不产生 pelvis 柔顺偏移。物理外力不删分量，每个物理子步重新施加，
  并补 `(施力点-质心) × F` 力矩。

### W&B 指标及单位

新增 `chip_metrics` 组，仅记录，**不进入 PPO 奖励**：

| 名称 | 每步物理量 |
| --- | --- |
| `right_wrist_error` / `left_wrist_error` | 硬件参考点对原始参考的世界系位置误差，m |
| `right_wrist_ori_error` | 右腕姿态误差，rad |
| `right_wrist_force` / `left_wrist_force` / `torso_force` | 实际施加外力的模长，N |
| `right_wrist_compliance` / `left_wrist_compliance` | 未乘 10 的 c，m/N |

`reward.chip_metrics/right_wrist_error` 等是逐步跨环境统计的 EMA；
`train/stats/chip_metrics/right_wrist_error` 等保留框架的**每回合累加值**，
不能直接把数值解读为 cm。单回合位置误差和除以该回合 `episode_len` 才是
该回合平均误差（m）。力的平均统计包括无力环境/时间，不能等同于峰值范围。
`chip_metrics/return=0` 是正常的，因为全部是 log-only 项。

旧 CHIP 的 `right_wrist_error` 是各自 anchor 局部坐标误差；新版本使用与
三点指标一致的世界坐标误差，**同名不代表可直接横向比较**。
无 axis 版本不记录沿轴/垂轴分量；原有三点、全身 tracking 和 loco 指标继续保留。

### 启动与验证

从 `active-adaptation/` 运行（先确认目标 GPU 空闲）：

```bash
# 数据划分与池权重沿用三点版：0.4 / 0.4 / 0.04 / 0.16
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/prepare_chip_loco.py --task-template three_point_chip

# 小规模 GPU 冒烟：16 个控制步、一次 PPO 更新；测试专用外力课程会加速
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/smoke_three_point.py --chip --loco

# 示例：四卡，每卡 1024 环境；默认 W&B offline，需要联网记录时显式覆盖
NPROC_PER_NODE=4 NUM_ENVS=1024 TOTAL_ITERS=4000 bash projects/mimic-lite/scripts/train_three_point_chip.sh wandb.mode=offline

# 使用本版本训练的 checkpoint，旁边需有对应 cfg.yaml
THREE_POINT_CHECKPOINT=/path/to/checkpoint.pt venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/play.py \
  --config-name replay_three_point_chip --config-dir .cache/three_point_chip_loco \
  task=three_point_chip_loco_val
```

Replay 默认 c=0、外力=0、关闭随机化（本体观测噪声仍沿用原三点 replay），
不是完全无噪声评测。评测柔顺时需显式设置 c/外力方案。
本地验证包括配置继承、原 reward 保留、无截断虚拟目标、每子步施力/力臂、
指标记录、CPU Transformer PPO 更新与 checkpoint 兼容性；GPU 短程测试和
实际训练质量仍需空闲 GPU 验证，不能据此承诺已收敛或实机可用。

### Lambda 四卡 Slurm

脚本：`scripts/train_three_point_chip_lambda_4gpu.slurm`。同步到 Lambda 后，
在 `/share/ml/yangmin/mimiclite-chip/active-adaptation` 下提交：

```bash
sbatch projects/mimic-lite/scripts/train_three_point_chip_lambda_4gpu.slurm
```

默认单节点四卡、64 CPU、`research` account、`lv0b` QoS、`HGX,DGX`
partition、两天时限、30,000 iterations，W&B offline，从头训练。
每卡环境数默认 `14336`，可显式 `NUM_ENVS=... sbatch ...` 覆盖。
设置 `NUM_ENVS=auto` 才读取单卡测试生成的
`.cache/three_point_chip_lambda_profile.json`，并检查四张 GPU 型号/容量。
axis 入口为 `scripts/train_three_point_chip_axis_lambda_4gpu.slurm`，
它选择 `three_point_chip_axis` 并调用共用启动器；auto 目前复用无 axis 的
容量测试结果，尚不是独立的 axis 容量验证。

单卡测试按用户的 `salloc -N 1 -t 8:00:00 --cpus-per-task 64 --account=research
--qos=lv0a --job-name dexhand --gres=gpu:1 -p HGX,DGX` 申请，入口为
`scripts/allocate_three_point_chip_lambda.sh`。从 16,384 环境开始，以 2,048
为步长自适应调整，最高试 20,480；每档 10 次完整 PPO 更新，统计 MuJoCo/Warp
等非 PyTorch 显存及设备实际占用，保留至少 20% 余量。此结果只代表单卡短程
容量测试通过，不能取代正式四卡启动和长期稳定性验证。

作业内先以一张分配到的 GPU 运行 `smoke_three_point.py --chip --loco`，
成功后才开始四卡训练。这个小型检查不等同于生产 batch 的显存验证。
不手工设置物理卡号，GPU 可见范围由 Slurm 分配。
日志写入 `/share/ml/yangmin/mimiclite-chip/records/three-point-chip-<jobid>/`。
此脚本本身的生成不代表已提交作业；远端同步、短程运行和训练状态应另行核实。

## Hardware contract

Ordered goals are **pelvis, left hand, right hand**. Each goal contains position
and orientation. Link-frame offsets in metres:

| Link | Offset xyz |
| --- | --- |
| pelvis | `[0, 0, 0]` |
| left_wrist_yaw_link | `[0.18, -0.025, 0]` |
| right_wrist_yaw_link | `[0.0719, -0.003, 0]` |

The right point is **30.4 mm beyond the wrist mounting face**, along local +X:
`[0.0415,-0.003,0] + [0.0304,0,0]`. It is NOT 30.4 mm from the body's COM.
Both reference and actual hand points are computed as `p_link + R_link * offset`;
all eleven target knots and current-error fields use these same points.

The robot asset is the existing `g1-mode_15-chip-payload`, unchanged:

- 237 g at mounting face + 2.5 mm, link position `[.044,-.003,0]`;
- 280 g at mounting face + 17.5 mm, link position `[.059,-.003,0]`;
- original 170 g rubber-palm mass/COM/inertia contribution removed;
- right wrist fused body mass **0.601576 kg**, net change **+0.347 kg**;
- combined COM and inertia retained, not only a scalar mass adjustment;
- original visual/collision geometry unchanged, as in CHIP.

Payload modeling remains a two-point-mass approximation, not newly identified
sensor/handle rotational inertia. The new third goal is pelvis origin, NOT
CHIP's torso offset `[0,0,.35]`.

## Observation and action contract

`command`: 315D, three link-major 105D slots. Each slot contains 11 knots of
`xyz + rotation6d` (99D), followed by current position error (3D) and orientation
rotation-vector error (3D). Knots are `[-8,-4,-2,0,1,2,3,4,8,16,32]` at 50 Hz:
past 160 ms through future 640 ms. Rotation6d flattens the **first two rows**.
Coordinates use actual pelvis yaw; the origin is `(pelvis_x,pelvis_y,0)`, so
target height is retained. Error sign is reference minus actual.

`policy`: 556D, in this exact term-major order:

1. 7 root-angular-velocity samples: 21D;
2. 7 projected-gravity samples: 21D;
3. 7 joint-position samples: 203D (current MimicLite encoder-offset convention);
4. 7 joint-velocity samples: 203D;
5. 3 previous actions: 87D;
6. 7 mocap-derived root-linear-velocity samples: 21D.

History offsets are `[0,1,2,3,4,8,16]` (newest first). The velocity estimator
uses noisy delayed positions, dropped samples and causal EMA; initialization
uses simulated velocity. A matching real-world state estimator is required.

`link_mask`: 3D, fixed `[1,1,1]`, **not statistically normalized**. This migration
intentionally trains all three goals, not single-arm or eight-mode behavior.

Actor: 3 goal tokens + joint-history token (493D) + root-history token (63D),
embedding 256, 3 Transformer layers, 4 heads, FFN 1024. Its 29D Gaussian action
is transformed into delayed/smoothed joint-position targets by existing
`JointPosition`; it is not torque control or IK. Critic uses the current
framework's privileged group and MLP `[1024,512,256]`.

No reference leg q/dq, force, compliance coefficients or stiffness axis are
supplied to the actor. **CHIP weights are not compatible**. Replay/train reject
mismatched task, geometry, robot asset and model dimensions using saved cfg.yaml.
Export metadata records the three-point contract; the external CHIP deployment
adapter has NOT been changed and cannot accept this controller unchanged.

## Rewards and migration boundaries

Preserves the source three-point baseline's full-body reference tracking:
torso world position/orientation/velocity, body-local pose, body velocities,
joint pose/velocity, action-rate, joint-velocity/limits, self-collision,
feet-air-time and survival. Uses `exp(-mean(error)/sigma)`, not CHIP's squared
point-error kernels. Root reward/termination refers to **torso_link**, despite
the commanded root being pelvis. There is no added precision or compliance
reward. Body tracking terms still supervise link origins (including wrists),
as in the source baseline; they are not silently replaced with hand-only rewards.

The new **logging-only** `three_point_metrics` reports world position errors at
the actual offset hand points and pelvis, plus orientation errors. Raw episode
stats are sums: divide by episode length for means. Metrics use the executed
reference buffers, not the newly advanced future buffer.

Current-framework adaptations:

- environment-deferred component initialization and reset signatures;
- opt-in `compute_observations_on_reset` refreshes valid goals/histories after
  reset; legacy tasks including CHIP retain their previous reset behavior;
- current PPO/DDP/checkpoint/normalization infrastructure, with an opt-in actor
  branch and identity mask normalization;
- source entropy decay at 3250–3500 / 4000 updates becomes fractions
  `.8125–.875` in current PPO, scaling with the requested run length;
- current privileged observation implementation and current train AMP defaults;
- current hardware offsets and payload instead of stock wrist origins/asset;
- existing physical-data split checker is reused, with a new task template.

This is not a bitwise reproduction of the old training environment (the source
bundle used older MJLab/MuJoCo APIs). Keep current installed dependencies; do not
overwrite the framework with the old bundle.

## Training and replay

From `active-adaptation`, generate configs and file lists without changing data:

```bash
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/prepare_chip_loco.py \
  --task-template three_point --output .cache/three_point_loco
```

Uses existing train/val/test splits, four physical pools, sampling weights
`0.4 / 0.4 / 0.04 / 0.16` as in the source bundle's physical-data configuration.
This differs from CHIP's clip-count-proportional pool sampling; CHIP is unchanged.
Default root is `../loco_manip_physical_rollout_accepted_v1`.

After GPU smoke passes on a free GPU:

```bash
bash projects/mimic-lite/scripts/train_three_point.sh
# Defaults: one process, 1024 envs, 4000 updates, W&B offline, from scratch.
# Multi-GPU example (check occupancy/memory first, not yet profiled):
NPROC_PER_NODE=4 NUM_ENVS=1024 TOTAL_ITERS=4000 \
  bash projects/mimic-lite/scripts/train_three_point.sh
```

Set `THREE_POINT_DATA_ROOT` to move data, `THREE_POINT_PYTHON` to change Python.
Extra Hydra overrides are forwarded. No training is automatically submitted to
Lambda or any other server. Checkpoint interval is 500.

Replay a **new three-point** checkpoint, preserving its cfg.yaml:

```bash
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/play.py \
  --config-name replay_three_point --config-dir .cache/three_point_loco \
  task=three_point_loco_val checkpoint_path=/absolute/path/checkpoint_4000.pt
```

For a prepared single trajectory set `THREE_POINT_MOTION_DIR` and omit the
generated `task=...` / `--config-dir` overrides. `play.py`'s viewer convention is
unchanged. This replay uses the policy to control the robot, not kinematic pose
playback. Proprioceptive noise/delay remains enabled; goal noise is disabled.

Real-time VR supplies no future 640 ms by itself. Deployment needs a consistent
future-target generator or a separately trained causal-input variant, plus
matching FK, root-state estimation, point offsets and history ordering.

## Validation

```bash
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 venv/mjlab/.venv/bin/python \
  -m unittest discover -s projects/mimic-lite/tests -p 'test_three_point.py'
# Only when the selected GPU is free:
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/smoke_three_point.py
venv/mjlab/.venv/bin/python projects/mimic-lite/scripts/smoke_three_point.py --loco
```

CPU checks cover geometry/yaw/height, the executed-step metric, fixed hardware
configuration, mask isolation, Transformer gradients, one synthetic-data Muon
PPO update, state-dict restoration, checkpoint contract rejection, opt-in reset
behavior and real MuJoCo model mass/COM/inertia repeatability. Synthetic PPO data
does not prove the simulator rollout or learned tracking quality.

On 2026-09-15 the local RTX 4090 was occupied (~23.5 GB / 24 GB), so the GPU
integration scripts were **not run** and no full training was started. The
four-env script is provided for subsequent validation. Do not treat this port
as GPU/DDP/deployment-validated until those checks pass.

The pre-existing full unittest discovery has a process-global `set_backend`
conflict between `test_multi_dataset` and `test_teacher_future_observation`;
run those modules in separate processes rather than changing production code.

## Source provenance

The migrated actor and velocity estimator derive from the supplied local
`goal-body-transformer-training-20260914` bundle, whose supplied license is
preserved in `licenses/goal_body_bundle_LICENSE.txt`. Source SHA-256:

- `mimic_lite_learning/goal_body.py`: `76f4311576432e90c08b34a94fe267e181e892b0ee8c032cc7e28ff5b576c241`
- `mimic_lite/tasks/pelvis_two_ee.py`: `f59986d909afd4bad60238608c956becf8711c83c52a5c2b55aade20795fa96f`
- `mimic_lite/tasks/observations/two_ee.py`: `b15af64f42a9d125face442badb4c7ec1615dffc7138a003cfc2382e4f5c159b`
