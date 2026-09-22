"""CHIP/Transformer contracts without allocating a GPU or running real robots."""
import importlib.util
import math
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict

from mimic_lite.tasks.chip_math import ChipSchedule
from mimic_lite.tasks.three_point import ThreePointTracking
from mimic_lite.tasks.three_point_chip import (
    ThreePointChipTracking, three_point_chip_targets, three_point_chip_metric,
    three_point_precision_reward,
)
from mimic_lite_learning.goal_body import GoalBodyFlatActor

PROJECT = Path(__file__).resolve().parents[1]


def config(name):
    with initialize_config_dir(config_dir=str(PROJECT / 'cfg'), version_base=None):
        return compose(config_name=name)


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / f'scripts/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThreePointChipTests(unittest.TestCase):
    def test_inherited_rewards_sampling_hardware_and_replay(self):
        plain, cfg = config('train_three_point'), config('train_three_point_chip')
        old = OmegaConf.load(PROJECT / 'cfg/task/chip.yaml')
        for key, value in plain.task.reward.tracking.items():
            self.assertEqual(value, cfg.task.reward.tracking[key])
        self.assertEqual(cfg.task.reward.loco, plain.task.reward.loco)
        self.assertEqual(set(cfg.task.reward.tracking) - set(plain.task.reward.tracking), {'chip_right_hand_pos'})
        self.assertEqual(cfg.task.reward.tracking.chip_right_hand_pos.sigma, .03)
        self.assertEqual(cfg.task.robot, plain.task.robot)
        self.assertEqual(cfg.task.command.point_offsets, plain.task.command.point_offsets)
        self.assertEqual(cfg.task.command.chip.point_offsets[:2], cfg.task.command.point_offsets[1:])
        for key, value in cfg.task.command.chip.items():
            if key != 'compliance_mode':
                self.assertEqual(value, old.command.chip[key])
        self.assertTrue(cfg.algo.goal_body_compliance)
        self.assertFalse(plain.algo.goal_body_compliance)
        replay = config('replay_three_point_chip')
        self.assertEqual(replay.task.command._target_, 'mimic_lite.ThreePointChipTracking')
        self.assertEqual(replay.task.command.chip.max_force, 0)
        self.assertEqual(list(replay.task.command.chip.compliance_max), [0, 0, 0])
        self.assertEqual(replay.task.num_envs, 1)
        self.assertEqual(replay.task.observation.command.three_point_targets.position_noise_std, 0)
        self.assertEqual(replay.task.observation.command.three_point_targets._target_, 'mimic_lite.three_point_chip_targets')
        self.assertEqual(list(replay.task.randomization), [])
        for key, term in cfg.task.reward.chip_metrics.items():
            if not key.startswith('_'):
                self.assertFalse(term.enabled)
        self.assertFalse(cfg.task.reward.chip_metrics._enabled_)

    def make_command(self):
        c = SimpleNamespace(chip=ChipSchedule(2, 'cpu'), chip_compliance_scale=10.)
        pos = torch.zeros(2, 11, 3, 3)
        quat = torch.tensor([1., 0, 0, 0]).expand(2, 11, 3, 4)
        c.reference_points = lambda reward=False: (pos[:, 3], quat[:, 3]) if reward else (pos, quat)
        c.actual_points = lambda: (pos[:, 3].clone(), quat[:, 3])
        c.asset = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.zeros(2,3),
            root_link_quat_w=quat[:, 3, 0]))
        c.goal_compliance = lambda: ThreePointChipTracking.goal_compliance(c)
        c.virtual_reference_points = lambda: ThreePointChipTracking.virtual_reference_points(c)
        c.chip_applied_force = torch.zeros(2,3,3)
        c.chip_applied_axis = torch.zeros(2,3,3)
        return c

    def test_unclipped_shift_all_knots_error_and_compliance_order(self):
        c = self.make_command()
        c.chip.compliance[:] = torch.tensor([.01, .02, 0])
        c.chip.force[:, 0, 1] = -5.
        c.chip.force[:, 1, 0] = 20.
        ref, _ = c.virtual_reference_points()
        torch.testing.assert_close(ref[:,:,2,0], torch.full((2,11), -.4))
        torch.testing.assert_close(ref[:,:,1,1], torch.full((2,11), .05))
        torch.testing.assert_close(c.reference_points()[0], torch.zeros_like(ref))
        term = three_point_chip_targets(position_noise_std=0., orientation_noise_std=0.)
        term._initialize(SimpleNamespace(command_manager=c,num_envs=2,device='cpu'))
        obs = term.compute()
        self.assertEqual(obs.shape, (2,318))
        torch.testing.assert_close(obs[:,-3:], torch.tensor([[0,.1,.2]]).expand(2,3))
        # Last 6 values in each 105D slot are xyz/orientation error.
        torch.testing.assert_close(obs[:,210+99], torch.full((2,), -.4))
        # Rotate actual pelvis heading 90deg: world +X displacement -> local -Y.
        c.asset.data.root_link_quat_w = torch.tensor([[2**-.5,0,0,2**-.5]]).expand(2,4)
        rotated = term.compute()
        torch.testing.assert_close(rotated[:,210+100], torch.full((2,), .4), atol=1e-6, rtol=1e-6)
        c.chip.compliance.zero_()
        torch.testing.assert_close(c.virtual_reference_points()[0], c.reference_points()[0])

    def test_reward_is_nominal_offset_point_and_metrics_are_log_only(self):
        from active_adaptation.envs.env_base import RewardGroup
        c = self.make_command()
        c.chip.force[:,1,0] = 20
        c.chip.compliance[:,1] = .02
        ref, q = c.reference_points(reward=True)
        actual = ref.clone(); actual[:,2,0] += .03
        c.actual_points = lambda: (actual,q)
        term = three_point_precision_reward(point=2,sigma=.03,weight=1.)
        env = SimpleNamespace(command_manager=c,num_envs=2,device='cpu')
        term._initialize(env)
        torch.testing.assert_close(term._compute(), torch.full((2,1),math.exp(-1)))
        # Applied force, not next-step force. Both are still SI units.
        c.chip_applied_force[:,1,:] = torch.tensor([3.,4.,0.])
        metrics = OrderedDict(force=three_point_chip_metric(body=1,quantity='force',weight=1.,enabled=False),
                              compliance=three_point_chip_metric(body=1,quantity='compliance',weight=1.,enabled=False))
        env.stats = TensorDict({('chip_metrics',k):torch.zeros(2,1) for k in metrics}, [2])
        group = RewardGroup('chip_metrics',metrics,enabled=False)
        group._initialize(env)
        torch.testing.assert_close(group.compute(), torch.zeros(2,1))
        torch.testing.assert_close(env.stats['chip_metrics','force'], torch.full((2,1),5.))
        torch.testing.assert_close(env.stats['chip_metrics','compliance'], torch.full((2,1),.02))

    def test_force_each_substep_com_lever_arm_and_partial_reset(self):
        c = self.make_command()
        c.chip_body_ids = [0,1,2]
        c.chip_offsets = torch.tensor([[.18,-.025,0],[.0719,-.003,0],[0,0,.35]])
        c.asset.data.body_link_pos_w = torch.zeros(2,3,3)
        c.asset.data.body_link_quat_w = torch.tensor([1.,0,0,0]).expand(2,3,4)
        c.asset.data.body_com_pos_w = torch.zeros(2,3,3)
        c.asset.data.body_com_pos_w[:,:,0] = .01
        writes = []
        c.asset.write_external_wrench_to_sim = lambda f,t,**kw:writes.append((f.clone(),t.clone(),kw))
        c.chip.force[:,1,2] = 10.
        c.chip.steps = 77
        # Invoke actual class methods on a lightweight instance (no GPU model).
        instance = object.__new__(ThreePointChipTracking)
        instance.__dict__.update(c.__dict__)
        with patch.object(ThreePointTracking,'pre_step'):
            for i in range(4):
                instance.pre_step(i)
        self.assertEqual(len(writes),4)
        torch.testing.assert_close(writes[-1][0],c.chip.force)
        torch.testing.assert_close(writes[-1][1][:,1],torch.tensor([[-.03,-.619,0]]).expand(2,3))
        force1 = c.chip.force[1].clone()
        with patch.object(ThreePointTracking,'reset'):
            instance.reset(torch.tensor([0]))
        self.assertEqual(c.chip.steps,77)
        torch.testing.assert_close(c.chip.force[0],torch.zeros(3,3))
        torch.testing.assert_close(c.chip.force[1],force1)
        self.assertEqual(writes[-1][2]['env_ids'].tolist(),[0])

    def test_actor_mask_includes_suffix_and_compliance_has_gradient(self):
        torch.manual_seed(9)
        actor = GoalBodyFlatActor(num_links=3,compliance=True,embed_dim=32,num_layers=1,ff_dim=64)
        p,c,m = torch.randn(2,556),torch.randn(2,318,requires_grad=True),torch.tensor([[1.,1,0]]).expand(2,3)
        mean,_ = actor(p,c,m)
        changed = c.detach().clone(); changed[:,210:315] += 99; changed[:,317] += 99
        torch.testing.assert_close(mean,actor(p,changed,m)[0])
        mean.square().sum().backward()
        self.assertGreater(c.grad[:,315:317].abs().sum().item(),0)
        torch.testing.assert_close(c.grad[:,317],torch.zeros(2))
        with self.assertRaises(ValueError):
            actor(p,c[:,:315],m)

    def test_checkpoint_guard_and_old_chip_isolation(self):
        validator = script_module('three_point_checkpoint')
        legacy = script_module('chip_checkpoint')
        cfg = config('train_three_point_chip')
        with tempfile.TemporaryDirectory() as directory:
            cfg.checkpoint_path = str(Path(directory)/'checkpoint.pt')
            OmegaConf.save(cfg,Path(directory)/'cfg.yaml')
            with patch('active_adaptation.utils.wandb.parse_checkpoint_path',side_effect=lambda p:p):
                validator.check_three_point_checkpoint(cfg)
                self.assertIsNone(legacy.configure_chip_checkpoint(cfg))
                cfg.task.command.chip.max_force = 0
                validator.check_three_point_checkpoint(cfg)
                plain = config('train_three_point'); plain.checkpoint_path = cfg.checkpoint_path
                with self.assertRaisesRegex(ValueError,'goal_body_compliance'):
                    validator.check_three_point_checkpoint(plain)
                old = config('train'); old.task = OmegaConf.load(PROJECT/'cfg/task/chip.yaml')
                old.checkpoint_path = cfg.checkpoint_path
                with self.assertRaisesRegex(ValueError,'Transformer'):
                    legacy.configure_chip_checkpoint(old)
                cfg.task.command.chip.point_offsets[1][0] = .18
                with self.assertRaisesRegex(ValueError,'point_offsets'):
                    validator.check_three_point_checkpoint(cfg)

    def test_generated_dataset_template_has_no_inherited_motion_pool(self):
        generator = script_module('prepare_chip_loco')
        cfg = generator.load_task_template('three_point_chip')
        self.assertNotIn('defaults',cfg)
        cfg['command']['motion_cfgs'] = {pool:{'path':'/test','weight':1.} for pool in generator.POOLS}
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)/'task'; task_dir.mkdir()
            OmegaConf.save(OmegaConf.create(cfg),task_dir/'generated.yaml')
            with initialize_config_dir(config_dir=str(PROJECT/'cfg'),version_base=None):
                loaded = compose(config_name='train_three_point_chip',overrides=[
                    f'hydra.searchpath=[file://{directory}]','task=generated'])
            self.assertEqual(set(loaded.task.command.motion_cfgs),set(generator.POOLS))
            self.assertIn('chip_right_hand_pos',loaded.task.reward.tracking)


if __name__ == '__main__':
    unittest.main()
