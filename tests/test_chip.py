"""Run with python -m unittest discover -s projects/mimic-lite/tests -p test_chip.py."""
import importlib.util
from pathlib import Path
import unittest
import torch
import yaml

spec = importlib.util.spec_from_file_location("chip_math", Path(__file__).parents[1] / "mimic_lite/tasks/chip_math.py")
chip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chip)


class ChipTests(unittest.TestCase):
    def test_unclipped_and_zero_compliance(self):
        p = torch.zeros(1, 3, 3)
        f = torch.zeros_like(p)
        f[:, :, 0] = 20
        c = torch.tensor([[0.02, 0, 0]])
        result = chip.virtual_target(p, f, c, torch.tensor([[1., 0, 0, 0]]))
        self.assertAlmostEqual(result[0, 0, 0].item(), -0.4, places=6)
        torch.testing.assert_close(result[:, 1:], p[:, 1:])
        torch.testing.assert_close(p, torch.zeros_like(p))

    def test_world_force_to_robot_frame(self):
        # Robot +90 degree yaw: world +x is robot -y; subtracting gives +y.
        q = torch.tensor([[2**-0.5, 0, 0, 2**-0.5]])
        f = torch.tensor([[[5., 0, 0]]]).expand(1, 3, 3)
        out = chip.virtual_target(torch.zeros_like(f), f, torch.ones(1, 3) * .02, q)
        torch.testing.assert_close(out, torch.tensor([[[0., .1, 0]]]).expand_as(out), atol=1e-6, rtol=1e-6)

    def test_axis_projection_signed_orthogonal_and_unclipped(self):
        f = torch.tensor([[[20., 0, 0], [0, 12., 0], [-3., 4., 0]]])
        original = f.clone()
        axis = torch.tensor([[[1., 0, 0]]]).expand_as(f)
        q = torch.tensor([[1., 0, 0, 0]])
        c = torch.full((1, 3), .02)
        out = chip.virtual_target(torch.zeros_like(f), f, c, q, axis)
        torch.testing.assert_close(out, torch.tensor([[[-.4, 0, 0], [0, 0, 0], [.06, 0, 0]]]))
        torch.testing.assert_close(out, chip.virtual_target(torch.zeros_like(f), f, c, q, -axis))
        torch.testing.assert_close(f, original)
        torch.testing.assert_close(chip.virtual_target(torch.zeros_like(f), f, c*0, q, axis), torch.zeros_like(f))

    def test_distinct_world_wrist_axes_and_rotated_anchor(self):
        # Same local +X: left wrist points world +X, right wrist world +Y.
        f = torch.tensor([[[3., 4, 0]]]).expand(1, 3, 3)
        axes = torch.tensor([[[1., 0, 0], [0, 1., 0], [0, 0, 1.]]])
        q = torch.tensor([[2**-.5, 0, 0, 2**-.5]])
        out = chip.virtual_target(torch.zeros_like(f), f, torch.full((1, 3), .02), q, axes)
        torch.testing.assert_close(out, torch.tensor([[[0., .06, 0], [-.08, 0, 0], [0, 0, 0]]]), atol=1e-6, rtol=1e-6)

    def test_axis_distribution(self):
        torch.manual_seed(417)
        s = chip.ChipSchedule(50000, "cpu", compliance_mode="wrist_axis")
        axis = s.stiffness_axis
        torch.testing.assert_close(axis.norm(dim=-1), torch.ones(len(axis)), atol=2e-6, rtol=1e-6)
        self.assertTrue((axis[:, 0] >= 0).all())
        x = axis[:, 0]
        exact = x == 1
        bands = [exact, (~exact) & (x >= 3**.5/2), (x < 3**.5/2) & (x >= .5), x < .5]
        for mask, expected in zip(bands, (.20, .60, .15, .05)):
            self.assertAlmostEqual(mask.float().mean().item(), expected, delta=.008)
        # Cos(theta) is uniform inside each band; Y/Z azimuth is symmetric.
        for mask, expected in zip(bands[1:], ((1+3**.5/2)/2, (3**.5/2+.5)/2, .25)):
            self.assertAlmostEqual(x[mask].mean().item(), expected, delta=.01)
        self.assertLess(axis[:, 1:].mean(0).abs().max().item(), .008)
        # Physical forces are still world-isotropic, NOT axis-aligned.
        self.assertLess(s.direction.mean(0).abs().max().item(), .015)

    def test_axis_holds_until_force_cycle_and_partial_reset(self):
        s = chip.ChipSchedule(16, "cpu", compliance_mode="wrist_axis", warmup_steps=0, ramp_steps=0)
        before = s.stiffness_axis.clone()
        s.c_remaining[:] = 1  # Compliance resampling must NOT change the axis.
        s.age[:] = s.interval + s.duration//2
        s.prepare()
        torch.testing.assert_close(s.stiffness_axis, before)
        s.age[0] = s.interval + s.duration[0]
        s.prepare()
        torch.testing.assert_close(s.stiffness_axis[1:], before[1:])
        self.assertTrue((s.force[0] == 0).all())
        other = s.stiffness_axis[1:].clone()
        s.reset(torch.tensor([0]))
        torch.testing.assert_close(s.stiffness_axis[1:], other)
        s._sample_axis(torch.empty(0, dtype=torch.long))

    def test_fixed_axis_normalization_and_config_validation(self):
        s = chip.ChipSchedule(5, "cpu", compliance_mode="wrist_axis", fixed_stiffness_axis=[2, 0, 0])
        torch.testing.assert_close(s.stiffness_axis, torch.tensor([[1., 0, 0]]).expand(5, 3))
        for kwargs in ({"fixed_stiffness_axis": [0, 0, 0]}, {"fixed_stiffness_axis": [float("nan"), 0, 0]},
                       {"fixed_stiffness_axis": [1, 0]}, {"compliance_mode": "unknown"},
                       {"axis_probabilities": [.2, .6, .15, -.05]},
                       {"axis_probabilities": [.2, .6, .15, .5]},
                       {"axis_cone_degrees": [30, 20, 90]}, {"axis_cone_degrees": [30, 60, 100]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                chip.ChipSchedule(1, "cpu", **kwargs)

    def test_training_axis_defaults(self):
        config = yaml.safe_load((Path(__file__).parents[1] / "cfg/task/chip.yaml").read_text())["command"]["chip"]
        self.assertEqual(config["compliance_mode"], "wrist_axis")
        self.assertEqual(config["axis_probabilities"], [.20, .60, .15, .05])
        self.assertEqual(config["axis_cone_degrees"], [30, 60, 90])
        self.assertIsNone(config["fixed_stiffness_axis"])

    def test_force_at_offset_has_moment(self):
        f = torch.tensor([[[0., 10, 0]]])
        _, torque = chip.point_wrench(f, torch.tensor([[[.18, 0, 0]]]), torch.zeros_like(f))
        torch.testing.assert_close(torque, torch.tensor([[[0., 0, 1.8]]]))

    def test_right_force_point_from_mount_not_com(self):
        config = yaml.safe_load((Path(__file__).parents[1] / "cfg/task/chip.yaml").read_text())
        points = torch.tensor(config["command"]["chip"]["point_offsets"])
        mount = torch.tensor([.0415, -.003, 0.])
        torch.testing.assert_close(points[1] - mount, torch.tensor([.0304, 0., 0.]))
        torch.testing.assert_close(points[0], torch.tensor([.18, -.025, 0.]))
        torch.testing.assert_close(points[2], torch.tensor([0., 0., .35]))
        force = torch.tensor([0., 10., 0.])
        # Moving COM changes the equivalent torque, not the link-fixed point.
        for com_x in (.02, .05):
            _, torque = chip.point_wrench(force, points[1], torch.tensor([com_x, 0., 0.]))
            torch.testing.assert_close(torque, torch.tensor([0., 0., (.0719 - com_x) * 10]))

    def test_sampling_and_reset_isolation(self):
        torch.manual_seed(42)
        s = chip.ChipSchedule(10000, "cpu", warmup_steps=0, ramp_steps=0)
        self.assertTrue((s.compliance >= 0).all() and (s.compliance <= .02).all())
        self.assertTrue((s.compliance[:, 2] == 0).all())
        self.assertAlmostEqual((s.compliance[:, 0] == 0).float().mean().item(), .25, delta=.03)
        self.assertAlmostEqual((s.compliance[:, 0] == .02).float().mean().item(), .25, delta=.03)
        self.assertAlmostEqual(s.amplitude.mean().item(), 10, delta=.3)
        self.assertTrue((s.amplitude < 20).all())
        self.assertLess(s.direction.mean(0).abs().max().item(), .03)
        single = s.mask.sum(-1) == 1
        self.assertAlmostEqual(single.float().mean().item(), .9, delta=.03)
        self.assertAlmostEqual(s.mask[single, 1].float().mean().item(), .75, delta=.03)
        c_other = s.compliance[1:].clone()
        s.force.fill_(7)
        s.reset(torch.tensor([0]))
        self.assertTrue((s.force[0] == 0).all())
        self.assertTrue((s.force[1:] == 7).all())
        torch.testing.assert_close(c_other, s.compliance[1:])

    def test_envelope_and_curriculum(self):
        s = chip.ChipSchedule(1, "cpu", warmup_steps=2, ramp_steps=2, force_interval=0)
        s.duration[:] = 80
        s.amplitude[:] = 20
        s.mask[:] = True
        s.direction[:] = torch.tensor([1., 0, 0])
        for step, expected in [(0, 0), (2, 0), (3, 10), (4, 20)]:
            s.steps = step
            s.age[:] = 40
            s.prepare()
            self.assertAlmostEqual(s.force[0, 0, 0].item(), expected)
        for age, expected in [(0, 0), (10, 10), (20, 20), (60, 20), (70, 10)]:
            s.age[:] = age
            s.prepare()
            self.assertAlmostEqual(s.force[0, 0, 0].item(), expected)

    def test_weighted_reward_is_original_reference_error(self):
        error = torch.tensor([[0., .04**2, 0.]])
        weights = torch.tensor([2., 4., 1.])
        expected = torch.exp(torch.tensor(-4/7))
        torch.testing.assert_close(chip.weighted_tracking(error, weights, .04).squeeze(), expected)

    def test_precision_config_and_right_hand_reward(self):
        config = yaml.safe_load((Path(__file__).parents[1] / "cfg/task/chip.yaml").read_text())
        rewards = config["reward"]["tracking"]
        expected = {
            "chip_pos": (.10, 2.0, False),
            "chip_pos_precision": (.04, 2.0, False),
            "chip_pos_fine": (.03, 2.0, False),
            "chip_ori": (.40, .50, True),
            "chip_ori_precision": (.15, .25, True),
            "chip_ori_fine": (.10, .25, True),
        }
        for name, (sigma, weight, orientation) in expected.items():
            self.assertEqual(rewards[name]["sigma"], sigma)
            self.assertEqual(rewards[name]["weight"], weight)
            self.assertEqual(rewards[name].get("orientation", False), orientation)
        right = rewards["chip_right_hand_pos"]
        self.assertEqual(right["weight"], 1.0)
        self.assertEqual(right["sigma"], .03)
        weights = torch.tensor(right["point_weights"], dtype=torch.float32)
        # Other points cannot suppress the standalone right-hand reward.
        errors = torch.tensor([[0., .03**2, 0.], [1., .03**2, 4.]])
        result = chip.weighted_tracking(errors, weights, right["sigma"])
        torch.testing.assert_close(result, torch.full((2, 1), torch.exp(torch.tensor(-1.)).item()))
        self.assertTrue((result < chip.weighted_tracking(errors, weights, .04)).all())
        torch.testing.assert_close(chip.weighted_tracking(torch.zeros(1, 3), weights, right["sigma"]),
                                   torch.ones(1, 1))


if __name__ == "__main__":
    unittest.main()
