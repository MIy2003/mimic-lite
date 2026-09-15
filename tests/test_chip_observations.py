"""Numeric CHIP command/critic frame and ABI checks without a simulator."""
from types import SimpleNamespace
import unittest

import torch

from mimic_lite.tasks.chip import ChipTracking, chip_command, chip_privileged, chip_metric
from mimic_lite.tasks.chip_math import ChipSchedule


class ObservationTests(unittest.TestCase):
    def setUp(self):
        # Left wrist identity, right wrist +90 yaw; pelvis identity.
        q = torch.tensor([[[1., 0, 0, 0], [2**-.5, 0, 0, 2**-.5],
                           [1., 0, 0, 0], [1., 0, 0, 0]]])
        schedule = ChipSchedule(1, "cpu", compliance_mode="wrist_axis", fixed_stiffness_axis=[1, 0, 0])
        schedule.compliance[:] = torch.tensor([.02, .02, 0])
        schedule.force[:] = torch.tensor([3., 4., 0])
        self.c = SimpleNamespace(
            chip=schedule, chip_compliance_scale=10., chip_body_ids=[0, 1, 2],
            anchor_body_idx_asset=3, asset=SimpleNamespace(data=SimpleNamespace(body_link_quat_w=q)),
            chip_leg_ids=list(range(12)), obs_current_step_index=0,
            ref_joint_pos_future_=torch.zeros(1, 1, 12), ref_joint_vel_future_=torch.zeros(1, 1, 12),
            ref_anchor_quat_future_w=q[:, 3:4],
            chip_reference=lambda **kwargs: (torch.zeros(1, 3, 3), q[:, [0, 0, 0]]),
        )
        self.c.chip_axis_world = lambda: ChipTracking.chip_axis_world(self.c)
        self.env = SimpleNamespace(command_manager=self.c, num_envs=1, device="cpu")

    def term(self, cls, **kwargs):
        term = cls(**kwargs)
        term._initialize(self.env)
        return term

    def test_actual_wrist_frames_and_axis_suffix(self):
        torch.testing.assert_close(self.c.chip_axis_world(), torch.tensor([[[1., 0, 0], [0, 1., 0], [1., 0, 0]]]), atol=1e-6, rtol=1e-6)
        obs = self.term(chip_command).compute()
        self.assertEqual(obs.shape, (1, 57))
        torch.testing.assert_close(obs[:, 24:33], torch.tensor([[-.06, 0, 0, 0, -.08, 0, 0, 0, 0]]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(obs[:, 51:54], torch.tensor([[.2, .2, 0]]))
        torch.testing.assert_close(obs[:, 54:57], torch.tensor([[1., 0, 0]]))

    def test_critic_keeps_full_force_and_legacy_abi(self):
        priv = self.term(chip_privileged).compute()
        self.assertEqual(priv.shape, (1, 36))
        torch.testing.assert_close(priv[:, 21:30], self.c.chip.force.flatten(1))
        torch.testing.assert_close(priv[:, 33:36], self.c.chip.stiffness_axis)
        self.c.chip.compliance_mode = "isotropic"
        obs = self.term(chip_command).compute()
        self.assertEqual(obs.shape, (1, 54))
        self.assertEqual(self.term(chip_privileged).compute().shape, (1, 33))
        torch.testing.assert_close(obs[:, 24:33], torch.tensor([[-.06, -.08, 0, -.06, -.08, 0, 0, 0, 0]]))

    def test_metrics_use_applied_axis_not_newly_sampled_axis(self):
        self.c.chip_applied_force = self.c.chip.force.clone()
        self.c.chip_applied_axis = self.c.chip_axis_world().clone()
        self.c.chip.stiffness_axis[:] = torch.tensor([0., 0, 1.])
        self.c.chip_actual = lambda: (torch.tensor([[[0., 0, 0], [.03, .04, 0], [0, 0, 0]]]), None)
        for quantity, expected in (("right_wrist_force_parallel", 4.), ("right_wrist_force_perpendicular", 3.),
                                   ("right_wrist_error_parallel", .04), ("right_wrist_error_perpendicular", .03)):
            value = self.term(chip_metric, weight=1., quantity=quantity)._compute()
            self.assertAlmostEqual(value.item(), expected, places=5)


if __name__ == "__main__":
    unittest.main()
