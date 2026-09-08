# MimicLite

For the MJLab CHIP compliance task (`c=0..0.02 m/N`, no displacement clipping),
see [CHIP setup, observation contract, and verification](CHIP.md).

## CHIP 训练与部署提醒（2026-09-08）

**右手施力点已不再使用原 BM/CHIP 的 `[0.18, 0.025, 0] m`。**
新位置是从腕部末端与手掌连接的安装面，沿右腕局部 `+X` 向外
`5 + 25.4 = 30.4 mm`，不是从腕部质心量起。

- 安装基准：模型 `right_hand_palm_joint`，相对 `right_wrist_yaw_link`
  为 `[0.0415, -0.003, 0] m`。
- 新右手点：`[0.0719, -0.003, 0] m`，配置见 [chip.yaml](cfg/task/chip.yaml)。
- 物理施力点、actor 参考点、实际测量点及 tracking reward 参考点同步使用此偏移。
  左手和躯干点保持不变；即使只使用力读数，仿真仍保留 `(施力点 - 质心) × F` 的力臂力矩。
- 载荷位置独立于施力点：237 g 在安装面外 2.5 mm，280 g 在外 17.5 mm。
  已扣除原橡胶手掌 170 g 及其惯性贡献，右腕合并质量为 **601.576 g**。
  新载荷采用集中质量近似；原手掌显示与碰撞几何仍保留，不代表完整的实机安装几何。

原始动作 NPZ 和身体 FK 缓存无需重建，点偏移在运行时应用。使用生成的
`chip_loco_*` 配置时，需要从 `active-adaptation` 目录重新运行
`projects/mimic-lite/scripts/prepare_chip_loco.py`；`train_chip_loco.sh` 会自动执行此步骤。
部署端也应对齐新的右手参考点，不要继续无检查地沿用旧 18 cm 点。

**远端版本提醒：** Hugging Face 的 `releases/chip-handoff-20260907` 旧交接包
未包含本次腕部载荷、扣除 170 g 以及右手施力点变更。交给他人训练前需更新代码包；
数据集不需要因此重新上传。

## Setup

Clone the current main branches. Active Adaptation no longer embeds MimicLite
as a submodule, so clone the two repositories independently:

```bash
git clone https://github.com/Agent-3154/active-adaptation.git
cd active-adaptation
git clone https://github.com/EGalahad/mimic-lite projects/mimic-lite
```

Setup uv venv directories and install dependencies:

```bash
mkdir -p venv/mjlab
cp projects/mimic-lite/pyproject-mjlab.toml venv/mjlab/pyproject.toml
uv sync --project venv/mjlab

mkdir -p venv/isaaclab
cp projects/mimic-lite/pyproject-isaaclab.toml venv/isaaclab/pyproject.toml
uv sync --project venv/isaaclab
```

The repository should now look like this:

```text
active-adaptation/
├── venv/
│   ├── mjlab/
│   │   └── pyproject.toml
│   └── isaaclab/
│       └── pyproject.toml
├── active_adaptation/
├── projects/
│   └── mimic-lite/
└─ scripts/
```

Refresh project discovery:

```bash
uv --project venv/mjlab run aa-discover-projects
uv --project venv/mjlab run aa-project enable mimic_lite
uv --project venv/mjlab run aa-list-tasks
```

These commands refresh project discovery, enable MimicLite, and verify its tasks.

Set up environment variables:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
export HF_TOKEN=<your_huggingface_token>
```

Training motion datasets are available in the [any4hdmi Hugging Face collection](https://huggingface.co/collections/elijahgalahad/any4hdmi). Dataset conversion and validation tools are maintained in [`EGalahad/any4hdmi`](https://github.com/EGalahad/any4hdmi).

## Released Checkpoints

The public release set now exposes only the latest 16x16384 G1 mixture Huge
policies. Training compute is reported as GPU hours on RTX 4090 GPUs.

| Policy | Actor hidden dimensions | Parallel environments | Checkpoint | GPU hours |
| --- | --- | ---: | --- | ---: |
| MimicLite-PPO | `[1024, 1024, 1024]` | `16 × 16384` | [`4234dd57`](https://wandb.ai/elijahgalahad/mimic_lite/runs/4234dd57) | 92.3 |
| MimicLite-ROA | `[1024, 1024, 1024]` | `16 × 16384` (`train -> adapt -> finetune`) | [`9287d8e0`](https://wandb.ai/elijahgalahad/mimic_lite/runs/9287d8e0) | 173.2 |

Compare the released policies on the [Motion Tracking Leaderboard](https://egalahad.github.io/sim2real/leaderboard).

Download the deploy ONNX and YAML from the shared sim2real artifacts:
[MimicLite-PPO](https://drive.google.com/drive/folders/1xRmcOX0l-YIpqxUuCmW4s6Dl0HStbJSL)
and
[MimicLite-ROA](https://drive.google.com/drive/folders/1AFcvP4oDbEskx-wip5bJN-JBaUvwp8MH).
Older Huge/Base/v1.1 releases are retained only in the Drive archive.

![Canonical cross-codebase tracking evaluation](assets/mimic_lite_cross_codebase_tracking_eval.png)

## Train

Run single-stage PPO:

```bash
bash scripts/launch_ddp.sh 0,1,2,3,4,5,6,7 projects/mimic-lite/scripts/train.py venv/mjlab \
  task=tracking-base task/motion=g1/mixture +exp=ppo/train \
  algo/ppo/module=huge backend=mjlab
```

Run PPO-ROA sequential training (`train -> adapt -> finetune`) with the Huge
module on one 8-GPU node:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv --project venv/mjlab run \
  projects/mimic-lite/scripts/train_sequential.py \
  task/motion=g1/mixture \
  +task/patches=teacher_future_t16 \
  algo/ppo_roa/module=huge \
  task.num_envs=8192
```

If this runs out of memory, append `task.command.diff_future_steps=[0,1]`.

Play a PPO checkpoint:

```bash
uv --project venv/mjlab run projects/mimic-lite/scripts/play.py \
  task=tracking-base task/motion=g1/lafan \
  +exp=ppo/train algo/ppo/module=huge \
  task.num_envs=4 task.termination.root_pos_error.enabled=false \
  checkpoint_path=run:elijahgalahad/mimic_lite/4234dd57
```

Play only the student from a PPO-ROA finetune checkpoint:

```bash
uv --project venv/mjlab run \
  projects/mimic-lite/scripts/play.py \
  task/motion=g1/lafan \
  +task/patches=teacher_future_t16 \
  +exp=ppo_roa/finetune algo/ppo_roa/module=huge \
  task.num_envs=1 task.termination.root_pos_error.enabled=false \
  checkpoint_path=run:elijahgalahad/mimic_lite/9287d8e0
```

## Troubleshooting

### IsaacLab Warp cache

If IsaacLab picks up Isaac Sim's bundled Warp instead of the venv-installed
`warp-lang`, clear the cached Omni Warp extensions and retry:

```bash
rm -rf venv/isaaclab/.venv/lib/python3.11/site-packages/isaacsim/extscache/omni.warp*
rm -rf venv/isaaclab/.venv/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/exts/3/omni.warp*
```

### mjlab MuJoCo compatibility

If mjlab training fails with an error like `mujoco.mjtEnableBit.mjENBL_MULTICCD` missing while importing `mujoco_warp`, your environment likely resolved `mujoco>=3.8`. Pin `mujoco<3.8` and resync the environment:

```bash
uv --project venv/mjlab add 'mujoco<3.8'
uv --project venv/mjlab sync
```

## Citation

If you find MimicLite useful in your research, please cite:

```bibtex
@misc{mimiclite2026,
  author       = {{RoboParty Lab Team}},
  title        = {MimicLite: Efficient and Effective General Humanoid Motion Tracking},
  year         = {2026},
  howpublished = {\url{https://github.com/EGalahad/mimic-lite}},
  note         = {Technical report: \url{https://github.com/Roboparty/MimicLite/blob/main/mimic-lite.pdf}}
}
```
