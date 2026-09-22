import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from active_adaptation.utils.math import quat_rotate, quat_mul
from mimic_lite.tasks.three_point import (
    POINT_OFFSETS, GOAL_STEPS, ThreePointTracking, offset_points, pack_goals, three_point_error,
)
from mimic_lite_learning.goal_body import GoalBodyFlatActor
from mimic_lite_learning.ppo import PPOConfig, PPOPolicy
from mimic_lite_learning.common import NullVecNorm

PROJECT = Path(__file__).parents[1]


class ThreePointTests(unittest.TestCase):
    def test_config_hardware_and_legacy_isolation(self):
        with initialize_config_dir(config_dir=str(PROJECT / 'cfg'), version_base=None):
            cfg = compose(config_name='train_three_point')
            replay = compose(config_name='replay_three_point')
            old = compose(config_name='train', overrides=['task=chip'])
        self.assertEqual(cfg.task.robot.name, old.task.robot.name)
        self.assertEqual(list(cfg.task.command.point_offsets)[1:], list(old.task.command.chip.point_offsets)[:2])
        self.assertEqual(list(cfg.task.command.future_steps), list(GOAL_STEPS))
        self.assertTrue(cfg.task.compute_observations_on_reset)
        self.assertFalse(old.task.get('compute_observations_on_reset', False))
        self.assertFalse(old.algo.goal_body_actor)
        self.assertEqual(replay.task.robot.name, cfg.task.robot.name)
        self.assertEqual(cfg.task.reward.tracking.root_pos._target_, 'mimic_lite.body_pos_exp')
        self.assertEqual(cfg.task.reward.tracking.joint_pos.weight, .5)
        self.assertEqual(cfg.task.reward.loco.feet_air_time.weight, 4.)
        PPOConfig(**OmegaConf.to_container(cfg.algo, resolve=True))

    def test_offset_geometry_heading_and_height(self):
        q = torch.tensor([[[1.,0,0,0],[1.,0,0,0],[2**-.5,0,0,2**-.5]]])
        pos = torch.zeros(1,3,3)
        pos[:,:,2] = .8
        points = offset_points(pos, q, torch.tensor(POINT_OFFSETS))
        torch.testing.assert_close(points[0,2], torch.tensor([.003,.0719,.8]), atol=1e-6, rtol=1e-6)
        ref_pos = points[:,None].expand(-1,11,-1,-1)
        ref_q = q[:,None].expand(-1,11,-1,-1)
        goals = pack_goals(ref_pos,ref_q,points,q,pos[:,0],q[:,0])
        self.assertEqual(goals.shape,(1,3,105))
        torch.testing.assert_close(goals[...,-6:], torch.zeros(1,3,6))
        self.assertAlmostEqual(goals[0,0,2].item(), .8, places=6)
        # Global yaw + XY translation leaves the heading-local command unchanged.
        yaw = torch.tensor([[[2**-.5,0,0,2**-.5]]]).expand_as(q)
        shift = torch.tensor([2.,3.,0.])
        p2 = quat_rotate(yaw, points) + shift
        q2 = quat_mul(yaw,q)
        root2 = quat_rotate(yaw[:,0],pos[:,0])+shift
        g2 = pack_goals(p2[:,None].expand(-1,11,-1,-1),q2[:,None].expand(-1,11,-1,-1),p2,q2,root2,yaw[:,0])
        torch.testing.assert_close(goals,g2,atol=2e-6,rtol=2e-6)

    def test_reference_metric_uses_executed_not_future_frame(self):
        c = SimpleNamespace(point_tracking_ids=[0,1,2], point_offsets=torch.tensor(POINT_OFFSETS))
        c.ref_body_pos_w = torch.zeros(1,3,3)
        c.ref_body_quat_w = torch.tensor([1.,0,0,0]).expand(1,3,4)
        c.ref_body_pos_future_w = torch.ones(1,11,3,3)*99
        c.ref_body_quat_future_w = c.ref_body_quat_w[:,None].expand(-1,11,-1,-1)
        c.reference_points = lambda reward=False: ThreePointTracking.reference_points(c,reward)
        actual, q = c.reference_points(reward=True)
        actual = actual.clone()
        actual[:,2,0] += .02
        c.actual_points = lambda: (actual,q)
        metric = three_point_error(weight=1., enabled=False)
        metric._initialize(SimpleNamespace(command_manager=c,num_envs=1,device='cpu'))
        self.assertAlmostEqual(metric._compute().item(),.02,places=6)

    def test_actor_mask_gradient_and_roundtrip(self):
        torch.manual_seed(2)
        model = GoalBodyFlatActor(num_links=3,embed_dim=32,num_layers=1,ff_dim=64)
        p,c,m = torch.randn(2,556),torch.randn(2,315),torch.tensor([[1.,1,0],[1.,1,0]])
        mean,scale = model(p,c,m)
        changed = c.clone(); changed[:,210:] += 100
        torch.testing.assert_close(mean,model(p,changed,m)[0])
        self.assertTrue(torch.isfinite(model(p,c,torch.zeros_like(m))[0]).all())
        (mean.square().mean()+scale.mean()).backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()))
        other = GoalBodyFlatActor(num_links=3,embed_dim=32,num_layers=1,ff_dim=64)
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(mean,other(p,c,m)[0])

    def test_ppo_cpu_update_and_checkpoint(self):
        self._check_ppo(False)

    def test_chip_ppo_cpu_update_and_checkpoint(self):
        self._check_ppo(True)

    def test_axis_ppo_cpu_update_and_checkpoint(self):
        self._check_ppo(True, axis=True)

    def _check_ppo(self, compliance, axis=False):
        torch.manual_seed(5)
        env = SimpleNamespace(cfg=OmegaConf.create({'total_iters':4,'reward':{'tracking':{},'loco':{}}}),
                              action_manager=SimpleNamespace(action_dim=29,joint_names=[f'j{i}' for i in range(29)]))
        spec = Composite({k:Unbounded((2,n)) for k,n in [('policy',556),('command',321 if axis else (318 if compliance else 315)),('priv',47 if axis else (44 if compliance else 32)),('link_mask',3)]},shape=(2,))
        cfg = PPOConfig(goal_body_actor=True,goal_body_compliance=compliance,goal_body_axis=axis,in_keys=('policy','command','priv','link_mask'),
                        goal_body_embed_dim=32,goal_body_num_layers=1,goal_body_ff_dim=64,
                        critic_hidden_dims=(32,),ppo_epochs=1,num_minibatches=1,opt='muon',train_amp_dtype=None)
        policy = PPOPolicy(cfg,spec,None,None,'cpu',env)
        self.assertIsInstance(policy.vecnorms['link_mask'],NullVecNorm)
        data = spec.rand().unsqueeze(1).expand(2,4).clone()
        data['link_mask'] = torch.ones(2,4,3)
        data['is_init'] = torch.zeros(2,4,1,dtype=torch.bool)
        with torch.no_grad():
            policy.get_rollout_policy()(data)
        nxt = spec.rand().unsqueeze(1).expand(2,4).clone()
        nxt['link_mask'] = torch.ones(2,4,3)
        nxt['reward'] = TensorDict({'tracking':torch.rand(2,4,1),'loco':torch.rand(2,4,1)},[2,4])
        nxt['done'] = torch.zeros(2,4,1,dtype=torch.bool)
        nxt['terminated'] = torch.zeros_like(nxt['done'])
        nxt['discount'] = torch.ones(2,4,1)
        data['next'] = nxt
        before = [x.clone() for x in policy.actor.parameters()]
        info = policy.train_op(data)
        self.assertTrue(all(torch.isfinite(torch.as_tensor(v)).all() for v in info.values()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,policy.actor.parameters())))
        clone = PPOPolicy(cfg,spec,None,None,'cpu',env)
        clone.load_state_dict(policy.state_dict())
        for a,b in zip(policy.actor.parameters(),clone.actor.parameters()):
            torch.testing.assert_close(a,b)

    def test_real_asset_payload_mass_com_inertia(self):
        import numpy as np
        import mujoco
        from mimic_lite.assets.g1 import G1_MJCF_REF_BY_MODE, resolve_asset_reference
        from mimic_lite.assets.chip_payload import add_wrist_payload
        spec = mujoco.MjSpec.from_file(str(resolve_asset_reference(G1_MJCF_REF_BY_MODE[15])))
        base = spec.compile()
        payload = add_wrist_payload(spec.copy()).compile()
        repeated = add_wrist_payload(spec.copy()).compile()
        wrist = base.body('right_wrist_yaw_link').id
        self.assertAlmostEqual(payload.body_mass[wrist] - base.body_mass[wrist], .347, places=6)
        self.assertAlmostEqual(payload.body_mass[wrist], .601576, places=6)
        np.testing.assert_allclose(payload.body_mass, repeated.body_mass)
        np.testing.assert_allclose(payload.body_ipos, repeated.body_ipos)
        np.testing.assert_allclose(payload.body_inertia, repeated.body_inertia)
        self.assertEqual(base.nq, payload.nq)
        print('Right wrist payload body mass:', payload.body_mass[wrist])

    def test_checkpoint_contract_rejects_chip_and_geometry_changes(self):
        spec = importlib.util.spec_from_file_location('three_point_checkpoint', PROJECT / 'scripts/three_point_checkpoint.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from unittest.mock import patch
        with initialize_config_dir(config_dir=str(PROJECT / 'cfg'),version_base=None):
            cfg = compose(config_name='train_three_point')
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.pt'
            cfg.checkpoint_path = str(checkpoint)
            OmegaConf.save(cfg,Path(directory) / 'cfg.yaml')
            with patch('active_adaptation.utils.wandb.parse_checkpoint_path',side_effect=lambda p:p):
                module.check_three_point_checkpoint(cfg)
                cfg.task.command.point_offsets[2][0] = .18
                with self.assertRaisesRegex(ValueError,'point_offsets'):
                    module.check_three_point_checkpoint(cfg)

    def test_reset_observations_are_opt_in_and_do_not_change_chip(self):
        from active_adaptation.envs.env_base import _EnvBase
        for enabled in (False,True):
            spec = Composite({'policy':Unbounded((2,3))},shape=(2,))
            c = SimpleNamespace(num_envs=2,device='cpu',episode_length_buf=torch.ones(2),
                episode_id=torch.zeros(2,dtype=torch.long),episode_count=0,
                cfg={'compute_observations_on_reset':enabled},observation_spec=spec,
                _reset_idx=lambda ids,td:None,scene=SimpleNamespace(reset=lambda ids:None),
                _reset_callbacks=[],_compute_observation=lambda td:td.set('policy',torch.ones(2,3)))
            result = _EnvBase._reset(c)
            torch.testing.assert_close(result['policy'],torch.full((2,3),float(enabled)))

    def test_velocity_estimator_history_and_partial_reset(self):
        from mimic_lite.tasks.observations.goal_body import goal_root_lin_vel_history
        data = SimpleNamespace(root_link_pos_w=torch.zeros(2,3),
            root_link_quat_w=torch.tensor([[1.,0,0,0]]).expand(2,4),root_com_lin_vel_w=torch.zeros(2,3))
        env = SimpleNamespace(num_envs=2,device='cpu',step_dt=.02,command_manager=None,
                              scene=SimpleNamespace(articulations={'robot':SimpleNamespace(data=data)}))
        term = goal_root_lin_vel_history(history_steps=[0,1,2,3,4,8,16],
            position_noise_std=0.,output_noise_std=0.,timestamp_jitter_std=0.,dropout_prob=0.,
            latency_steps_range=[0,0],ema_alpha=1.)
        term._initialize(env)
        data.root_link_pos_w[:,0] += .02
        term.update()
        torch.testing.assert_close(term.compute()[:,:3],torch.tensor([[1.,0,0]]).expand(2,3))
        other = term.history_buffer[1].clone()
        term.reset(torch.tensor([0]))
        torch.testing.assert_close(term.history_buffer[1],other)
        torch.testing.assert_close(term.compute()[0],torch.zeros(21))


if __name__ == '__main__':
    unittest.main()
