# MimicLite CHIP 五点控制（2026-10-23）

独立任务 `chip_five_point`：左手、右手、躯干、左脚、右脚都接收位置和姿态目标。
pelvis 仍为参考坐标系；第三点仍是 torso，不替换为 pelvis。
手部沿用 job 977 的六维 CHIP 柔顺配置，双脚只做刚性目标跟踪。
这里的“刚性”指没有足部柔顺指令或 hindsight 位移，不保证实际脚部误差为零。

## 最近训练和当前回放

本地可核实的最后一批正式训练是 2026-10-01 的 job 977/978：

- 977：`chip_6d_xyz_rot_scalar_no_upper_tracking_loco_train`。
- 978：同组实验，增加 weight=0.1 的 soft-axis tracking 惩罚。
- 两者本地都有 `checkpoint_24500.pt`，都是三点加下肢关节的 69D 指令。
- 977 的 `checkpoint_11500.pt` 虽在 10 月 5 日才下载，仍是同一轮较早的权重。
- 项目根目录 `scripts/replay_walk.py` 默认加载 862/11500，因此最近使用的回放
  默认模型与最后一批训练不是同一个概念。

证据：项目根目录 `records/no-upper-tracking-ablation/EXPERIMENT.md`、
`checkpoints/lambda-{977,978}-24500/transfer_manifest.json` 和各自 `cfg.yaml`。
这些记录不代表已经查询了远程集群的实时状态。

## 五点输入契约

| 范围 | 内容 |
|---|---|
| 0:15 | 五点 xyz，顺序 left hand / right hand / torso / left foot / right foot |
| 15:35 | 同顺序五点 wxyz 四元数 |
| 35:41 | 实际 pelvis 到目标 pelvis 的相对旋转，沿用原矩阵前两列的展平顺序 |
| 41:59 | 原 left hand / right hand / torso 三点各六项归一化柔顺 |

五点位姿均采用原 CHIP 的参考 pelvis 局部表示。脚部默认选用
`left_ankle_roll_link`、`right_ankle_roll_link` 的原点及朝向。
这不是假定的脚底中心；需要脚底控制点时，统一设置
`task.command.foot_point_offsets`，并以导出元数据中的偏移为准。

替换掉原指令中下肢 12D q + 12D dq，增加双脚 6D 位置 + 8D 四元数：
`69 - 24 + 14 = 59D`。双脚没有额外的柔顺通道。
actor 输入是 `930 + 59 = 989D`，输出仍为 29D 关节位置动作。
测量关节位置、速度、历史动作继续留在本体状态历史中。

训练继续使用原始动作数据提供足部 FK/位姿、critic 特权信息和密集奖励，
保留下肢关节跟踪奖励作为训练监督；推理接口不需要下肢参考 q/dq。
新增足部位置和姿态奖励，并使用对应 transition 的参考快照。
上肢外力及 hindsight 不改变足部目标，也不向双脚增加 CHIP 外部 wrench。
刚性足部目标可以随时间运动，不是把双脚固定在地面。

### 2026-10-27：轻量上肢姿态约束与独立 torso 跟踪

当前训练配置在保留 4045 下肢奖励的基础上，补回 862 使用的双侧
shoulder-roll / elbow / wrist-yaw link 监督。单独的上肢位置、姿态、线速度、
角速度奖励权重为 `0.15 / 0.15 / 0.05 / 0.05`，sigma 沿用 862 的
`0.30 / 0.40 / 1.0 / 3.14`。这是轻量姿态先验，不锁死关节；需要检查它对
右手柔顺的影响。原下肢跟踪强度不因加入上肢后重新平均而降低。

按用户要求，六项共享位置/姿态奖励的 `point_weights` 从默认 `2:4:1`
改为 `2:4:2`（左手/右手/torso），torso 与左手使用相同误差权重和 sigma。
按后续要求同时保留 torso-only 奖励：位置 `0.5`（sigma `0.05 m`）、
姿态 `0.75`（`0.30 rad`）、精细姿态 `0.5`（`0.10 rad`）；全身跟踪保持。
因此共享项与左手同权，但 torso 的总约束额外增强。
归一化后左手与 torso 各占 `1/4`，右手占 `1/2`；这些是共享误差内的
比例，不代表所有奖励合计后的身体部位贡献。
新增 `chip_metrics/torso_error`（米）和 `torso_ori_error`（弧度），只记录，
不加入优化奖励。姿态误差使用四元数角距离，不将欧拉角直接相减。

这针对 torso 相对 pelvis 的位姿；不等于约束世界坐标下的绝对躯干位置。
若外部给出的 torso 位置与姿态不满足三轴腰关节的几何关系，加奖励也无法
使其同时精确实现，需先检查目标可达性。实际控制点仍是 torso link 上方
`[0, 0, 0.35] m` 的导出偏移点。

这些系数是待验证的第一版，不是已证实最优参数。8000/10000/12000 的既有
权重不会因改 YAML 获得上述能力；旧模型回放仍恢复各自保存的奖励配置。
应通过新训练及同参考、同柔顺、同施力的闭环对照验证 torso 姿态精度、
左右臂串扰及右手顺应性，不能仅以训练总奖励判断效果。

该任务改变了 actor/critic 的输入契约，需要重新训练。旧 69D/75D 模型不能
直接接收五点指令，跨三点/五点续训会被拒绝；同五点任务可正常续训。
导出格式为 `mimic_lite_chip_five_point_v1`，包含五点 link/偏移和 59D 布局。
HIL_Dagger 与 GR00T 原有的三点协议尚未迁移，不能直接替换其现用模型。

## 运行

在 `/home/yangmin/code/MimicLite` 下：

```bash
# 生成独立数据配置并从头训练；正式训练前先检查 GPU 占用。
CHIP_NUM_ENVS=4096 CHIP_TOTAL_ITERS=30000 \
  bash mimic-lite/scripts/train_chip_five_point_loco.sh checkpoint_path=null

# 只验证配置，不启动训练。
bash mimic-lite/scripts/train_chip_five_point_loco.sh --cfg job --resolve
```

在 `active-adaptation` 下检查实际仿真与 PPO 更新：

```bash
export PYTHONPATH=projects/mimic-lite:.:../any4hdmi/src
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0
venv/mjlab/.venv/bin/python -m unittest discover \
  -s projects/mimic-lite/tests -p 'test_chip*.py' -v

# CPU 小规模检查，不占用训练 GPU；默认不传 --device 时使用 GPU。
CUDA_VISIBLE_DEVICES='' venv/mjlab/.venv/bin/python \
  projects/mimic-lite/scripts/smoke_chip.py --five-point --device cpu
CUDA_VISIBLE_DEVICES='' venv/mjlab/.venv/bin/python \
  projects/mimic-lite/scripts/smoke_chip.py --five-point --device cpu --loco
```

真实数据 smoke 需要先运行训练入口的配置验证命令，生成
`.cache/chip_five_point_loco`。原始 14,038 条动作和 train/val/test 划分不变。
两份源码（`mimic-lite` 和 `active-adaptation/projects/mimic-lite`）均已加入此任务。

验证产物位于项目根目录 `records/five-point-control/`。其中 smoke checkpoint
仅用于程序和导出验证，不是训练完成的控制器，不能据此判断步行或真机控制效果。
