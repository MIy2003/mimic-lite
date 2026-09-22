"""Four-env MJLab check: hardware points, reset state and one Transformer PPO update.

Run only on a free GPU. This is not a trained-policy quality evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import mujoco
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import active_adaptation as aa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loco', action='store_true', help='Use prepared four-pool smoke data')
    parser.add_argument('--chip', action='store_true', help='Validate the isotropic CHIP variant')
    parser.add_argument('--axis', action='store_true', help='Validate wrist-axis CHIP')
    args = parser.parse_args()
    args.chip = args.chip or args.axis
    project = Path(__file__).resolve().parents[1]
    root = Path(tempfile.mkdtemp(prefix='mimiclite-three-point-smoke-'))
    variant = 'three_point_chip_axis' if args.axis else ('three_point_chip' if args.chip else 'three_point')
    with initialize_config_dir(config_dir=str(project / 'cfg'), version_base=None):
        overrides = ['task.num_envs=4', 'wandb.mode=disabled']
        if args.loco:
            prepared = project.parents[1] / f'.cache/{variant}_loco'
            overrides += [f'hydra.searchpath=[file://{prepared}]', f'task={variant}_loco_train_smoke']
        cfg = compose(config_name=f'train_{variant}', overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    if not args.loco:
        cfg.task.command.motion_cfgs.motion.path = str(root)
    cfg.task.command.init_joint_pos_noise = 0
    cfg.task.command.init_joint_vel_noise = 0
    cfg.task.observation.command.three_point_targets.position_noise_std = 0
    cfg.task.observation.command.three_point_targets.orientation_noise_std = 0
    cfg.algo.goal_body_embed_dim = 64
    cfg.algo.goal_body_ff_dim = 128
    cfg.algo.critic_hidden_dims = [64,64]
    cfg.algo.train_every = 8
    cfg.algo.num_minibatches = 1
    cfg.algo.ppo_epochs = 1
    cfg.task.total_iters = 1
    if args.chip:
        # Exercise nonzero physical force within 16 steps; production schedule
        # remains 500-step warmup, 5000-step ramp and 100-step intervals.
        cfg.task.command.chip.warmup_steps = 0
        cfg.task.command.chip.ramp_steps = 0
        cfg.task.command.chip.force_interval = 0
    OmegaConf.resolve(cfg)
    aa.init(cfg, auto_rank=True)
    from mimic_lite.assets.g1 import G1_MJCF_REF_BY_MODE, resolve_asset_reference
    model = mujoco.MjModel.from_xml_path(str(resolve_asset_reference(G1_MJCF_REF_BY_MODE[15])))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    if not args.loco:
        frames = 256
        names = [model.body(i).name for i in range(1,model.nbody)]
        joints = [i for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
        def repeat(x):
            return np.repeat(x[None],frames,axis=0).astype(np.float32)
        np.savez(root / 'motion.npz',body_pos_w=repeat(data.xpos[1:]),body_quat_w=repeat(data.xquat[1:]),
                 body_lin_vel_w=np.zeros((frames,len(names),3),np.float32),
                 body_ang_vel_w=np.zeros((frames,len(names),3),np.float32),
                 joint_pos=repeat(data.qpos[model.jnt_qposadr[joints]]),
                 joint_vel=np.zeros((frames,len(joints)),np.float32))
        (root / 'meta.json').write_text(json.dumps(dict(body_names=names,
            joint_names=[model.joint(i).name for i in joints],fps=50)))
    from active_adaptation.helpers import make_env_policy
    from mimic_lite.tasks.three_point import pack_goals
    env, policy = make_env_policy(cfg.task,cfg.algo,cfg.seed,cfg.headless,cfg.device)
    base = env.base_env
    c = base.command_manager
    try:
        td = env.reset()
        assert td['policy'].shape == (4,556)
        assert td['command'].shape == (4,321 if args.axis else (318 if args.chip else 315))
        assert (td['link_mask'] == 1).all(), 'Reset must not leave a zero goal mask'
        ref_p, ref_q = c.virtual_reference_points() if args.chip else c.reference_points()
        p,q = c.actual_points()
        expected = pack_goals(ref_p,ref_q,p,q,c.asset.data.root_link_pos_w,c.asset.data.root_link_quat_w).flatten(1)
        if args.chip:
            expected = torch.cat((expected,c.goal_compliance()*c.chip_compliance_scale),-1)
        if args.axis:
            expected = torch.cat((expected,c.chip.stiffness_axis),-1)
        torch.testing.assert_close(td['command'],expected)
        np.testing.assert_allclose(c.point_offsets.cpu(),[[0,0,0],[.18,-.025,0],[.0719,-.003,0]],atol=1e-7)
        mj_model = base.sim.mj_model
        # Runtime MJLab composes the robot into a namespaced scene; standalone
        # asset compilation (CPU tests) has no "robot/" prefix.
        try:
            wrist_body = mj_model.body('robot/right_wrist_yaw_link')
        except KeyError:
            wrist_body = mj_model.body('right_wrist_yaw_link')
        np.testing.assert_allclose(wrist_body.mass, .601576,atol=1e-7)
        actor = policy.get_rollout_policy()
        rollout = []
        max_applied_force = 0.
        for i in range(16):
            with torch.no_grad():
                td = actor(td)
                transition,carry = env.step_and_maybe_reset(td)
            for key in ('policy','command','link_mask'):
                assert torch.isfinite(carry[key]).all()
            assert (carry['link_mask'] == 1).all()
            assert all(torch.isfinite(v).all() for v in transition['next','reward'].values(True,True))
            if args.chip:
                max_applied_force = max(max_applied_force,c.chip_applied_force.norm(dim=-1).max().item())
                assert 'chip_metrics' not in transition['next','reward']
                assert torch.isfinite(base.stats['chip_metrics','right_wrist_error']).all()
            rollout.append(transition.exclude('stats','discount').clone())
            td = carry
            if i == 7:
                batch = torch.stack(rollout,dim=1)
                with torch.no_grad():
                    policy.compute_rollout_values(batch,carry.copy())
                info = policy.train_op(batch)
                assert all(torch.isfinite(torch.as_tensor(v)).all() for v in info.values())
                print('Transformer PPO update passed')
                rollout.clear()
        # Partial resets refresh the command/FK while preserving other histories.
        history = base.observation_groups['policy'].funcs['mocap_root_lin_vel_history']
        before = history.history_buffer[1:].clone()
        mask = torch.tensor([[True],[False],[False],[False]],device=base.device)
        # TorchRL partial reset merges reset rows into the supplied carry. A TD
        # containing only _reset has no observations to preserve for other rows.
        reset_input = td.clone().set('_reset', mask)
        carry = env.reset(reset_input)
        torch.testing.assert_close(history.history_buffer[1:],before)
        assert (carry['link_mask'] == 1).all()
        if args.chip:
            assert max_applied_force > 0, 'Smoke must exercise physical force'
            print('Max applied force during smoke:',max_applied_force)
        print(f'THREE_POINT_SMOKE_PASSED: {321 if args.axis else (318 if args.chip else 315)}+556+3 -> 29, payload, reset, finite rollout and PPO')
    finally:
        env.close()


if __name__ == '__main__':
    main()
