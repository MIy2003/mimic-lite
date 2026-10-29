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
    parser.add_argument("--mode", choices=("wrist_axis", "isotropic", "wrist_6d"), default="wrist_axis")
    parser.add_argument("--event", choices=("force", "couple", "mixed"), default="mixed")
    parser.add_argument("--checkpoint-output", type=Path, help="Save test-only checkpoint for export validation")
    parser.add_argument("--five-point", action="store_true", help="Use five poses with wrist_6d compliance")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="CPU allows small smoke checks without occupying a training GPU")
    args = parser.parse_args()
    if args.five_point:
        args.mode = "wrist_6d"
    root = Path(tempfile.mkdtemp(prefix="mimiclite-chip-smoke-"))
    config_dir = Path(__file__).resolve().parents[1] / "cfg"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        overrides = [f"task={'chip_five_point' if args.five_point else 'chip_wrist_joint' if args.mode == 'wrist_joint_binary' else 'chip_6d' if args.mode == 'wrist_6d' else 'chip'}", "task.num_envs=4", "wandb.mode=disabled"]
        if args.loco:
            framework = config_dir.parent.parent / 'active-adaptation'
            if not framework.is_dir():
                framework = config_dir.parents[2]
            prefix = 'chip_five_point_loco' if args.five_point else 'chip_wrist_joint_loco' if args.mode == 'wrist_joint_binary' else 'chip_6d_loco' if args.mode == 'wrist_6d' else 'chip_loco'
            prepared = framework / '.cache' / prefix
            overrides = [f"hydra.searchpath=[file://{prepared}]", f"task={prefix}_train_smoke",
                         "task.num_envs=4", "wandb.mode=disabled"]
        cfg = compose(config_name="train", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.task.command.chip.compliance_mode = args.mode
    if args.mode in ('wrist_6d', 'wrist_joint_binary'):
        cfg.task.command.chip.wrench_event_probabilities = {
            'force': [1.,0.,0.], 'couple': [0.,1.,0.], 'mixed': [0.,0.,1.]}[args.event]
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
    if args.device == "cpu":
        cfg.device = "cpu"
    from mimic_lite.assets.g1 import G1_MJCF_REF_BY_MODE, resolve_asset_reference
    from mimic_lite.tasks.chip_math import project_force
    model = mujoco.MjModel.from_xml_path(str(resolve_asset_reference(G1_MJCF_REF_BY_MODE[15])))
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
    env, policy = make_env_policy(cfg.task, cfg.algo, cfg.seed, cfg.headless, cfg.device)
    base = env.base_env
    c = base.command_manager
    td = env.reset()
    saw_force = False
    saw_couple = False
    joint_rejections = []
    joint_soft_rates = []
    rollout = []
    actor = policy.get_rollout_policy()
    for i in range(24):
        # Environment reset returns zero observations; first transition primes buffers.
        with torch.no_grad():
            td = actor(td)
            transition, carry = env.step_and_maybe_reset(td)
        nxt = transition["next"]
        command_dim = {'isotropic': 54, 'wrist_axis': 57, 'wrist_6d': 69, 'wrist_joint_binary': 75}[args.mode]
        if args.five_point:
            command_dim = 59
        point_start = 0 if args.five_point else 24
        compliance_start = 41 if args.five_point else 51
        assert nxt["command"].shape == (4, command_dim), nxt["command"].shape
        assert nxt["policy"].shape == (4, 930), nxt["policy"].shape
        assert torch.isfinite(nxt["command"]).all()
        ref_p, _ = c.chip_reference()
        actual_target = nxt["command"][:, point_start:point_start+9].reshape(4, 3, 3)
        displacement = (ref_p - actual_target).norm(dim=-1)
        # step_and_maybe_reset has already reset terminal environments' schedules;
        # their transition still correctly holds the pre-reset terminal observation.
        live = ~nxt["done"].flatten()
        force_for_target = c.chip.force
        if args.mode == "wrist_axis":
            force_for_target = project_force(force_for_target, c.chip_axis_world())
            torch.testing.assert_close(nxt["command"][live, 54:57], c.chip.stiffness_axis[live])
        if args.mode == 'wrist_joint_binary':
            torch.testing.assert_close(nxt['command'][live,compliance_start:compliance_start+18], c.chip_compliance_observation()[live])
            assert ((c.chip.compliance_6d[:,1,3:] == 0) | (c.chip.compliance_6d[:,1,3:] == .8)).all()
            joint_rejections.append(c.ik_rejected.float().mean().item())
            joint_soft_rates.append((c.chip.compliance_6d[:,1,3] > 0).float().mean().item())
            _, _, tau, _, _, _ = c.current_loads()
            assert (tau * c.chip.compliance_6d[:,1,3:]).abs().max() <= c.max_joint_shift+1e-6
        elif args.mode == 'wrist_6d':
            from mimic_lite.tasks.chip_math import wrist_compliance_shift
            quats = c.asset.data.body_link_quat_w[:, c.chip_body_ids[:2]]
            expected = wrist_compliance_shift(c.chip.force[:,:2], c.chip.compliance_6d[...,:3], quats).norm(dim=-1)
            torch.testing.assert_close(displacement[live,:2], expected[live], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(nxt['command'][live,compliance_start:compliance_start+18], c.chip_compliance_observation()[live])
            assert not c.chip.compliance_6d[:,0].any()
        else:
            torch.testing.assert_close(displacement[live], (force_for_target.norm(dim=-1) * c.chip.compliance)[live], atol=2e-6, rtol=2e-5)
        if args.five_point:
            foot_p, foot_q = c.foot_reference()
            torch.testing.assert_close(nxt['command'][live,9:15], foot_p[live].flatten(1))
            torch.testing.assert_close(nxt['command'][live,27:35], foot_q[live].flatten(1))
        if (~live).any():
            assert (c.chip.force[~live] == 0).all()
        saw_force |= bool((c.chip_applied_force.norm(dim=-1) > 0).any())
        saw_couple |= bool((c.chip_applied_couple.norm(dim=-1) > 0).any())
        assert all(torch.isfinite(value).all() for value in nxt["reward"].values(True, True))
        rollout.append(transition.exclude("stats", "discount").clone())
        td = carry
        if i == 7:
            batch = torch.stack(rollout, dim=1)
            with torch.no_grad():
                policy.compute_rollout_values(batch, carry.copy())
            info = policy.train_op(batch)
            print("PPO update passed:", {k: v for k, v in info.items() if "value_loss" in k or "grad_norm" in k})
            rollout.clear()
    assert saw_force == (args.mode not in ('wrist_6d', 'wrist_joint_binary') or args.event != 'couple')
    if args.mode in ('wrist_6d', 'wrist_joint_binary'):
        assert saw_couple == (args.event != 'force')
    if args.mode == 'wrist_joint_binary':
        from smoke_chip_wrist_joint import check_joint_contract
        check_joint_contract(base, c)
        print('Wrist fixture rates (not a trained-policy evaluation):',
              {'ik_rejected': sum(joint_rejections)/len(joint_rejections),
               'soft': sum(joint_soft_rates)/len(joint_soft_rates)})
    else:
        # Changing the force changes the actor target, not the original-reference reward.
        from mimic_lite.tasks.chip import chip_command
        reward = base.reward_groups["tracking"].funcs["chip_pos"]
        reward_before = reward._compute().clone()
        c.chip.force.zero_()
        c.chip.stiffness_axis[:] = torch.tensor([1., 0, 0], device=base.device)
        axes = c.chip_axis_world()
        c.chip.force[:, 1] = axes[:, 1] * 20
        c.chip.compliance[:, 1] = 0.02
        if args.mode in ('wrist_6d', 'wrist_joint_binary'):
            c.chip.compliance_6d[:,1,:3] = .02
            c.chip.couple_amplitude.zero_()
            c.chip.couple.zero_()
        if args.five_point:
            from mimic_lite.tasks.chip_five_point import chip_five_point_command
            term = chip_five_point_command()
        else:
            term = chip_command()
        term._initialize(base)
        obs = term.compute()
        nominal, _ = c.chip_reference()
        shift = (nominal[:, 1] - obs[:, point_start+3:point_start+6]).norm(dim=-1)
        torch.testing.assert_close(shift, torch.full_like(shift, .4), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(reward_before, reward._compute())
        if args.mode == "wrist_axis":
            # Physical force perpendicular to wrist +X: actor target is unshifted.
            from active_adaptation.utils.math import quat_rotate
            wrist_q = c.asset.data.body_link_quat_w[:, c.chip_body_ids[1]]
            c.chip.force[:, 1] = 20 * quat_rotate(wrist_q, torch.tensor([0., 1, 0], device=base.device).expand(4, -1))
            obs = term.compute()
            torch.testing.assert_close(obs[:, point_start+3:point_start+6], nominal[:, 1], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(reward_before, reward._compute())
        c.pre_step(0)
        actual_wrench = c.asset.data.body_external_wrench[:, c.chip_body_ids]
        torch.testing.assert_close(actual_wrench[..., :3], c.chip.force)
        assert (actual_wrench[:, 1, 3:].norm(dim=-1) > 0).all()
        # Reapply on every substep, even after MJLab clears the wrench buffer.
        c.asset.write_external_wrench_to_sim(torch.zeros_like(c.chip.force), torch.zeros_like(c.chip.force), body_ids=c.chip_body_ids)
        c.pre_step(1)
        torch.testing.assert_close(c.asset.data.body_external_wrench[:, c.chip_body_ids, :3], c.chip.force)
        other_forces = c.chip.force[1:].clone()
        c.reset(torch.tensor([0], device=base.device))
        assert (c.chip.force[0] == 0).all()
        assert (c.chip_applied_force[0] == 0).all()
        assert (c.chip_applied_couple[0] == 0).all()
        assert (c.asset.data.body_external_wrench[0, c.chip_body_ids] == 0).all()
        torch.testing.assert_close(c.chip.force[1:], other_forces)
    print(f"CHIP MJLab smoke passed: five_point={args.five_point}, {args.mode}, actor {930+command_dim}D, action 29D, "
          "physical wrench/substeps, PPO update, partial reset")
    print("Fixture/cache:", root)
    if args.checkpoint_output:
        args.checkpoint_output.mkdir(parents=True, exist_ok=True)
        torch.save({'policy': policy.state_dict(), 'env': env.state_dict(), 'cfg': cfg},
                   args.checkpoint_output / 'checkpoint.pt')
        OmegaConf.save(cfg, args.checkpoint_output / 'cfg.yaml')
        print('Test-only checkpoint:', args.checkpoint_output)
    env.close()


if __name__ == "__main__":
    main()
