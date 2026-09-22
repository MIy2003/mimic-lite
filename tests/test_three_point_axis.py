"""Wrist-frame projection, observation and checkpoint contracts for axis CHIP."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
from types import SimpleNamespace
import torch
import test_three_point_chip as chip_tests
from test_three_point_chip import config, script_module
from mimic_lite.tasks.three_point_chip import ThreePointChipTracking, three_point_chip_targets
from mimic_lite_learning.goal_body import GoalBodyFlatActor, LinkCommandMask


class AxisTests(unittest.TestCase):
    def test_config_and_checkpoint_contract(self):
        check = script_module('three_point_checkpoint').check_three_point_checkpoint
        for name in ('train_three_point_chip_axis', 'replay_three_point_chip_axis'):
            cfg = config(name)
            self.assertTrue(cfg.algo.goal_body_axis)
            self.assertEqual(cfg.task.command.chip.compliance_mode, 'wrist_axis')
            check(cfg)
            cfg.algo.goal_body_axis = False
            with self.assertRaisesRegex(ValueError, 'goal_body_axis'):
                check(cfg)
        replay = config('replay_three_point_chip_axis')
        self.assertEqual(replay.task.command.chip.max_force, 0)
        prepare = script_module('prepare_chip_loco')
        template = prepare.load_task_template('three_point_chip_axis')
        self.assertEqual(template['command']['chip']['compliance_mode'], 'wrist_axis')

    def test_saved_checkpoint_rejects_axis_mismatch(self):
        check = script_module('three_point_checkpoint').check_three_point_checkpoint
        cfg = config('train_three_point_chip_axis')
        with tempfile.TemporaryDirectory() as directory:
            cfg.checkpoint_path = str(Path(directory) / 'checkpoint.pt')
            OmegaConf.save(cfg, Path(directory) / 'cfg.yaml')
            with patch('active_adaptation.utils.wandb.parse_checkpoint_path', side_effect=lambda p: p):
                check(cfg)
                cfg.task.command.chip.fixed_stiffness_axis = [0, 1, 0]
                check(cfg)
                other = config('train_three_point_chip')
                other.checkpoint_path = cfg.checkpoint_path
                with self.assertRaisesRegex(ValueError, 'axis input differs'):
                    check(other)

    def test_actual_wrist_projection_and_axis_observation(self):
        c = chip_tests.ThreePointChipTests().make_command()
        c.chip.compliance_mode = 'wrist_axis'
        c.chip.stiffness_axis[:] = torch.tensor([1.,0,0])
        c.chip.compliance[:] = torch.tensor([.02,.02,0])
        c.chip.force[:] = torch.tensor([10.,20.,30.])
        c.chip_body_ids = [0,1,2]
        c.asset.data.body_link_quat_w = torch.tensor([[1.,0,0,0], [2**-.5,0,0,2**-.5], [1.,0,0,0]]).expand(2,3,4)
        c.chip_axis_world = lambda: ThreePointChipTracking.chip_axis_world(c)
        force_before = c.chip.force.clone()
        ref,_ = c.virtual_reference_points()
        torch.testing.assert_close(ref[:,:,0], torch.zeros(2,11,3))
        torch.testing.assert_close(ref[:,:,1], torch.tensor([-.2,0,0]).expand(2,11,3))
        torch.testing.assert_close(ref[:,:,2], torch.tensor([0.,-.4,0]).expand(2,11,3), atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(c.chip.force,force_before)
        term=three_point_chip_targets(position_noise_std=0,orientation_noise_std=0)
        term._initialize(SimpleNamespace(command_manager=c,num_envs=2,device='cpu'))
        obs=term.compute()
        self.assertEqual(obs.shape,(2,321))
        torch.testing.assert_close(obs[:,-3:],c.chip.stiffness_axis)
        torch.testing.assert_close(obs[:,315:318],torch.tensor([0.,.2,.2]).expand(2,3))

    def test_actor_axis_gradients_and_mask(self):
        actor=GoalBodyFlatActor(num_links=3,compliance=True,axis=True,embed_dim=32,num_heads=4,num_layers=1,ff_dim=64)
        command=torch.randn(2,321,requires_grad=True)
        mask=torch.tensor([[1.,1,1],[1.,0,1]])
        loc,scale=actor(torch.randn(2,556),command,mask)
        self.assertEqual(loc.shape,(2,29))
        loc.square().sum().backward()
        self.assertTrue(torch.isfinite(command.grad).all())
        self.assertGreater(command.grad[:,-3:].abs().sum().item(),0)
        torch.testing.assert_close(command.grad[1,105:210],torch.zeros(105))
        self.assertEqual(command.grad[1,316],0)
        masked=LinkCommandMask(3,True,True)(command,torch.zeros(2,3))
        torch.testing.assert_close(masked,torch.zeros_like(masked))

    def test_axis_metrics_use_applied_axis_and_nominal_reference(self):
        from collections import OrderedDict
        from tensordict import TensorDict
        from active_adaptation.envs.env_base import RewardGroup
        from mimic_lite.tasks.three_point_chip import three_point_chip_metric
        c = chip_tests.ThreePointChipTests().make_command()
        c.chip_applied_axis[:] = torch.tensor([1.,0,0])
        c.chip.stiffness_axis[:] = torch.tensor([0.,1,0])
        c.chip_applied_force[:] = torch.tensor([3.,4,0])
        nominal, quat = c.reference_points(reward=True)
        actual = nominal.clone()
        actual[:,1:] += torch.tensor([.03,.04,0])
        c.actual_points = lambda: (actual,quat)
        metrics = OrderedDict()
        expected = dict(force_parallel=3.,force_perpendicular=4.,error_parallel=.03,error_perpendicular=.04)
        cfg = config('train_three_point_chip_axis')
        for body,hand in enumerate(('left','right')):
            for quantity in expected:
                name = f'{hand}_wrist_{quantity}'
                self.assertFalse(cfg.task.reward.chip_metrics[name].enabled)
                metrics[name] = three_point_chip_metric(body=body,quantity=quantity,weight=1.,enabled=False)
        self.assertFalse(cfg.task.reward.chip_metrics._enabled_)
        env = SimpleNamespace(command_manager=c,num_envs=2,device='cpu',
            stats=TensorDict({('chip_metrics',k):torch.zeros(2,1) for k in metrics},[2]))
        group = RewardGroup('chip_metrics',metrics,enabled=False)
        group._initialize(env)
        torch.testing.assert_close(group.compute(),torch.zeros(2,1))
        for name,term in metrics.items():
            torch.testing.assert_close(env.stats['chip_metrics',name],torch.full((2,1),expected[term.quantity]))

    def test_physical_axis_cache_and_reset(self):
        from mimic_lite.tasks.three_point import ThreePointTracking
        c = chip_tests.ThreePointChipTests().make_command()
        c.chip.compliance_mode = 'wrist_axis'
        c.chip_body_ids = [0,1,2]
        c.chip_offsets = torch.zeros(3,3)
        c.asset.data.body_link_pos_w = torch.zeros(2,3,3)
        c.asset.data.body_com_pos_w = torch.zeros(2,3,3)
        c.asset.data.body_link_quat_w = torch.tensor([1.,0,0,0]).expand(2,3,4)
        c.asset.write_external_wrench_to_sim = lambda *args,**kwargs: None
        instance = object.__new__(ThreePointChipTracking)
        instance.__dict__.update(c.__dict__)
        with patch.object(ThreePointTracking,'pre_step'):
            instance.pre_step(0)
        before = instance.chip_applied_axis.clone()
        instance.chip.stiffness_axis[:] = torch.tensor([0.,1,0])
        torch.testing.assert_close(instance.chip_applied_axis,before)
        with patch.object(ThreePointTracking,'reset'):
            instance.reset(torch.tensor([0]))
        torch.testing.assert_close(instance.chip_applied_axis[0],torch.zeros(3,3))
        torch.testing.assert_close(instance.chip_applied_axis[1],before[1])
