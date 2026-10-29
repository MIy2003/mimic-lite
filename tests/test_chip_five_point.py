"""Five-point geometry, sparse command, reward timing and checkpoint contracts."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import mujoco
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import torch

from mimic_lite.assets.g1 import G1_MJCF_REF_BY_MODE, resolve_asset_reference
from mimic_lite.tasks.chip import ChipTracking, chip_command
from mimic_lite.tasks.chip_five_point import (
    ChipFivePointTracking, chip_five_point_command,
    chip_five_point_privileged, chip_feet_tracking_reward,
)
from mimic_lite.tasks.chip_math import ChipSchedule

PROJECT = Path(__file__).resolve().parents[1]


class FivePointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(resolve_asset_reference(G1_MJCF_REF_BY_MODE[15])))

    def setUp(self):
        data = mujoco.MjData(self.model)
        data.qpos[:3] = [.7, -.3, .8]
        data.qpos[3:7] = Rotation.from_euler('xyz', [.1, -.2, .7]).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(self.model, data)
        self.mjdata = data
        pos = torch.tensor(data.xpos, dtype=torch.float32)[None]
        quat = torch.tensor(data.xquat, dtype=torch.float32)[None]
        ids = lambda names: [self.model.body(n).id for n in names]
        anchor = self.model.body('pelvis').id
        chip_ids = ids(['left_wrist_yaw_link', 'right_wrist_yaw_link', 'torso_link'])
        foot_ids = ids(['left_ankle_roll_link', 'right_ankle_roll_link'])
        schedule = ChipSchedule(1, 'cpu', compliance_mode='wrist_6d',
                                compliance_6d_max=[.02]*3 + [1.]*3, left_compliance_mode='rigid')
        schedule.compliance_6d.zero_()
        schedule.force.zero_()
        schedule.couple.zero_()
        schedule.couple_amplitude.zero_()
        c = SimpleNamespace(num_envs=1, chip=schedule, chip_body_ids=chip_ids,
            chip_tracking_ids=chip_ids, foot_body_ids=foot_ids, foot_tracking_ids=foot_ids,
            chip_offsets=torch.tensor([[.18, -.025, 0], [.0719, -.003, 0], [0, 0, .35]]),
            foot_offsets=torch.tensor([[.02, 0, -.03], [.02, 0, -.03]]),
            chip_compliance_6d_scale=torch.tensor([50.]*3 + [1.]*3),
            anchor_body_idx_asset=anchor, obs_current_step_index=0,
            ref_body_pos_future_w=pos[:, None].clone(), ref_body_quat_future_w=quat[:, None].clone(),
            ref_anchor_pos_future_w=pos[:, anchor:anchor+1].clone(),
            ref_anchor_quat_future_w=quat[:, anchor:anchor+1].clone(),
            ref_body_pos_w=pos.clone(), ref_body_quat_w=quat.clone(),
            ref_anchor_pos_w=pos[:, anchor].clone(), ref_anchor_quat_w=quat[:, anchor].clone(),
            asset=SimpleNamespace(data=SimpleNamespace(body_link_pos_w=pos.clone(), body_link_quat_w=quat.clone())))
        for cls, names in ((ChipTracking, ('chip_reference', 'chip_virtual_reference', 'chip_compliance_observation')),
                           (ChipFivePointTracking, ('foot_reference', 'foot_actual'))):
            for name in names:
                setattr(c, name, getattr(cls, name).__get__(c))
        self.c = c

    def term(self, cls, **kwargs):
        term = cls(**kwargs)
        term._initialize(SimpleNamespace(command_manager=self.c, num_envs=1, device='cpu'))
        return term

    def test_five_poses_match_independent_mujoco_fk_without_leg_commands(self):
        c, data = self.c, self.mjdata
        command = self.term(chip_five_point_command).compute()
        self.assertEqual(command.shape, (1, 59))
        # No reference q/dq attributes exist in this fixture.
        ids = c.chip_body_ids + c.foot_body_ids
        offsets = torch.cat((c.chip_offsets, c.foot_offsets)).numpy()
        rotations = Rotation.from_quat(data.xquat[ids][:, [1, 2, 3, 0]])
        anchor = Rotation.from_quat(data.xquat[c.anchor_body_idx_asset][[1, 2, 3, 0]])
        world = data.xpos[ids] + rotations.apply(offsets)
        expected_p = anchor.inv().apply(world - data.xpos[c.anchor_body_idx_asset])
        expected_q = (anchor.inv() * rotations).as_quat()[:, [3, 0, 1, 2]]
        np.testing.assert_allclose(command[0, :15].reshape(5, 3), expected_p, atol=2e-6)
        np.testing.assert_allclose(command[0, 15:35].reshape(5, 4), expected_q, atol=2e-6)
        self.assertEqual(self.term(chip_five_point_privileged).compute().shape, (1, 71))
        # Replacing the old prefix preserves the original upper pose semantics.
        c.chip_leg_ids = list(range(12))
        c.ref_joint_pos_future_ = torch.randn(1, 1, 12)
        c.ref_joint_vel_future_ = torch.randn(1, 1, 12)
        old = self.term(chip_command).compute()
        torch.testing.assert_close(old[:, 24:33], command[:, :9])
        torch.testing.assert_close(old[:, 33:45], command[:, 15:27])
        torch.testing.assert_close(self.term(chip_five_point_command).compute(), command)

    def test_hand_hindsight_never_moves_foot_targets_or_nominal_rewards(self):
        term = self.term(chip_five_point_command)
        before = term.compute()
        reward = self.term(chip_feet_tracking_reward, weight=1.)
        self.c.chip.compliance_6d[:, 1, :3] = .02
        self.c.chip.force[:, 1] = torch.tensor([5., 0., 0.])
        after = term.compute()
        self.assertGreater((before[:, 3:6] - after[:, 3:6]).norm().item(), .09)
        torch.testing.assert_close(before[:, 9:15], after[:, 9:15])
        torch.testing.assert_close(before[:, 27:35], after[:, 27:35])
        torch.testing.assert_close(reward._compute(), torch.ones(1, 1))
        self.c.chip.force.zero_()
        torch.testing.assert_close(term.compute()[:, :35], before[:, :35])

    def test_reward_uses_snapshot_and_quaternion_sign_is_irrelevant(self):
        reward = self.term(chip_feet_tracking_reward, weight=1.)
        orientation = self.term(chip_feet_tracking_reward, orientation=True, weight=1.)
        self.c.ref_body_pos_future_w[:, :, self.c.foot_tracking_ids] += .2
        torch.testing.assert_close(reward._compute(), torch.ones(1, 1))
        self.c.asset.data.body_link_quat_w[:, self.c.foot_body_ids] *= -1
        torch.testing.assert_close(orientation._compute(), torch.ones(1, 1))
        self.c.asset.data.body_link_pos_w[:, self.c.foot_body_ids] += .1
        self.assertLess(reward._compute().item(), .1)

    def test_shared_world_frame_change_preserves_command(self):
        before = self.term(chip_five_point_command).compute()
        from active_adaptation.utils.math import quat_mul, quat_rotate
        rotation = torch.tensor([2**-.5, 0., 0., 2**-.5])
        translation = torch.tensor([2., -1., .4])
        for obj, pname, qname in (
            (self.c.asset.data, 'body_link_pos_w', 'body_link_quat_w'),
            (self.c, 'ref_body_pos_future_w', 'ref_body_quat_future_w'),
            (self.c, 'ref_anchor_pos_future_w', 'ref_anchor_quat_future_w')):
            p, q = getattr(obj, pname), getattr(obj, qname)
            rotation_batch = rotation.expand_as(q)
            setattr(obj, pname, quat_rotate(rotation_batch, p) + translation)
            setattr(obj, qname, quat_mul(rotation_batch, q))
        torch.testing.assert_close(self.term(chip_five_point_command).compute(), before, atol=2e-6, rtol=1e-5)

    def test_task_and_checkpoint_contract(self):
        with initialize_config_dir(config_dir=str(PROJECT/'cfg'), version_base=None):
            cfg = compose(config_name='train', overrides=['task=chip_five_point'])
            old = compose(config_name='train', overrides=['task=chip'])
        self.assertEqual(cfg.task.observation.command.chip_command._target_, 'mimic_lite.chip_five_point_command')
        self.assertEqual(cfg.task.command.chip.left_compliance_mode, 'rigid')
        spec = importlib.util.spec_from_file_location('five_point_checkpoint', PROJECT/'scripts/chip_checkpoint.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp, patch('active_adaptation.utils.wandb.parse_checkpoint_path', side_effect=lambda p: p):
            cfg.checkpoint_path = str(Path(tmp)/'checkpoint.pt')
            OmegaConf.save(cfg, Path(tmp)/'cfg.yaml')
            module.configure_chip_checkpoint(cfg)
            old.checkpoint_path = cfg.checkpoint_path
            with self.assertRaisesRegex(ValueError, 'three-point and five-point'):
                module.configure_chip_checkpoint(old)
            module.configure_chip_checkpoint(old, restore_mode=True)
            self.assertEqual(old.task.command._target_, cfg.task.command._target_)
            self.assertEqual(old.task.observation.command, cfg.task.observation.command)
            self.assertEqual(old.task.observation.priv.chip_privileged, cfg.task.observation.priv.chip_privileged)
            self.assertEqual(old.task.reward, cfg.task.reward)
            self.assertEqual(old.task.termination, cfg.task.termination)
            cfg.task.command.foot_point_offsets[0][2] = -.03
            with self.assertRaisesRegex(ValueError, 'changed five-point contract'):
                module.configure_chip_checkpoint(cfg)


if __name__ == '__main__':
    unittest.main()
