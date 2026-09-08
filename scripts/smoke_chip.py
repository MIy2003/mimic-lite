"""Create a static FK fixture and test the real MJLab CHIP environment + PPO.

Not training data or a policy quality evaluation. The fixture goes to a fresh
temporary directory; framework motion caches use active-adaptation/.cache.
Run from active-adaptation with the installed MJLab Python.
"""
import json
import os
from pathlib import Path
import tempfile
import argparse

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
import mujoco
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import active_adaptation as aa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loco", action="store_true", help="Use prepared real-data train smoke split")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="mimiclite-chip-smoke-"))
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        overrides = ["task=chip", "task.num_envs=4", "wandb.mode=disabled"]
        if args.loco:
            prepared = config_dir.parents[2] / ".cache/chip_loco"
            overrides = [f"hydra.searchpath=[file://{prepared}]", "task=chip_loco_train_smoke",
                         "task.num_envs=4", "wandb.mode=disabled"]
        cfg = compose(config_name="train", overrides=overrides)
    if not args.loco:
        cfg.task.command.motion_cfgs.chip_motion.path = str(root)
    cfg.task.command.chip.warmup_steps = 0
    cfg.task.command.chip.ramp_steps = 0
    cfg.task.command.chip.force_interval = 0
    cfg.task.command.chip.force_duration = [8, 12]
    cfg.task.command.chip.zero_probability = 0.0
    cfg.task.command.chip.max_probability = 1.0
    cfg.task.command.init_joint_pos_noise = 0
    cfg.task.command.init_joint_vel_noise = 0
    cfg.algo.actor_hidden_dims = [64, 64]
    cfg.algo.critic_hidden_dims = [64, 64]
    cfg.algo.opt = "adam"
    cfg.algo.train_every = 8
    cfg.algo.num_minibatches = 1
    cfg.algo.ppo_epochs = 1
    OmegaConf.resolve(cfg)
    aa.init(cfg, auto_rank=True)
    from mimic_lite.assets.g1 import G1_MODE_15_CFG
    model = mujoco.MjModel.from_xml_path(str(G1_MODE_15_CFG.mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    joints = [i for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    joint_names = [model.joint(i).name for i in joints]
    frames = 256
    def repeat(x):
        return np.repeat(x[None], frames, axis=0).astype(np.float32)
    arrays = dict(body_pos_w=repeat(data.xpos[1:]), body_quat_w=repeat(data.xquat[1:]),
                  body_lin_vel_w=np.zeros((frames, len(body_names), 3), np.float32),
                  body_ang_vel_w=np.zeros((frames, len(body_names), 3), np.float32),
                  joint_pos=repeat(data.qpos[model.jnt_qposadr[joints]]),
                  joint_vel=np.zeros((frames, len(joints)), np.float32))
    np.savez(root / "motion.npz", **arrays)
    (root / "meta.json").write_text(json.dumps(dict(body_names=body_names, joint_names=joint_names, fps=50)))
    from active_adaptation.helpers import make_env_policy
    env, policy = make_env_policy(cfg)
    base = env.base_env
    c = base.command_manager
    td = env.reset()
    saw_force = False
    rollout = []
    actor = policy.get_rollout_policy()
    for i in range(24):
        # Environment reset returns zero observations; first transition primes buffers.
        with torch.no_grad():
            td = actor(td)
            transition, carry = env.step_and_maybe_reset(td)
        nxt = transition["next"]
        assert nxt["command"].shape == (4, 54), nxt["command"].shape
        assert nxt["policy"].shape == (4, 930), nxt["policy"].shape
        assert torch.isfinite(nxt["command"]).all()
        ref_p, _ = c.chip_reference()
        actual_target = nxt["command"][:, 24:33].reshape(4, 3, 3)
        displacement = (ref_p - actual_target).norm(dim=-1)
        # step_and_maybe_reset has already reset terminal environments' schedules;
        # their transition still correctly holds the pre-reset terminal observation.
        live = ~nxt["done"].flatten()
        torch.testing.assert_close(displacement[live], (c.chip.force.norm(dim=-1) * c.chip.compliance)[live], atol=2e-6, rtol=2e-5)
        if (~live).any():
            assert (c.chip.force[~live] == 0).all()
        saw_force |= bool((c.chip_applied_force.norm(dim=-1) > 0).any())
        assert torch.isfinite(nxt["reward"]).all()
        rollout.append(transition.exclude("stats", "discount").clone())
        td = carry
        if i == 7:
            batch = torch.stack(rollout, dim=1)
            with torch.no_grad():
                policy.compute_rollout_values(batch, carry.copy())
            info = policy.train_op(batch)
            print("PPO update passed:", {k: v for k, v in info.items() if "value_loss" in k or "grad_norm" in k})
            rollout.clear()
    assert saw_force
    # Changing the force changes the actor target, not the original-reference reward.
    from mimic_lite.tasks.chip import chip_command
    reward = base.reward_groups["tracking"].funcs["chip_pos"]
    reward_before = reward._compute().clone()
    c.chip.force.zero_()
    c.chip.force[:, 1, 0] = 20.0
    c.chip.compliance[:, 1] = 0.02
    obs = chip_command(base).compute()
    nominal, _ = c.chip_reference()
    shift = (nominal[:, 1] - obs[:, 27:30]).norm(dim=-1)
    torch.testing.assert_close(shift, torch.full_like(shift, .4), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(reward_before, reward._compute())
    c.pre_step(0)
    actual_wrench = c.asset.data.body_external_wrench[:, c.chip_body_ids]
    torch.testing.assert_close(actual_wrench[..., :3], c.chip.force)
    assert (actual_wrench[:, 1, 3:].norm(dim=-1) > 0).all()
    other_forces = c.chip.force[1:].clone()
    c.reset(torch.tensor([0], device=base.device))
    assert (c.chip.force[0] == 0).all()
    assert (c.chip_applied_force[0] == 0).all()
    assert (c.asset.data.body_external_wrench[0, c.chip_body_ids] == 0).all()
    torch.testing.assert_close(c.chip.force[1:], other_forces)
    print("CHIP MJLab smoke passed: actor 984D, action 29D, unclipped cF, real external forces, PPO update, partial reset")
    print("Fixture/cache:", root)
    env.close()


if __name__ == "__main__":
    main()
