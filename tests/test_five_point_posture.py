"""Regression checks for posture supervision and independent torso tracking."""
from pathlib import Path
from types import SimpleNamespace
import math
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from mimic_lite.tasks.chip import chip_metric, chip_tracking_reward
from mimic_lite.tasks.rewards.track import body_tracking_squared_exp


class FivePointPostureTests(unittest.TestCase):
    def setUp(self):
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/"cfg"), version_base=None):
            self.cfg = compose(config_name="train", overrides=["task=chip_five_point"])
        self.ref_p = torch.zeros(2, 3, 3)
        self.actual_p = self.ref_p.clone()
        self.ref_q = torch.zeros(2, 3, 4)
        self.ref_q[..., 0] = 1
        self.actual_q = self.ref_q.clone()
        names = ["pelvis", "torso_link"] + [f"{side}_{part}_link"
            for side in ("left", "right")
            for part in ("hip_roll", "knee", "ankle_roll", "shoulder_roll", "elbow", "wrist_yaw")]
        self.command = SimpleNamespace(tracking_body_names=names,
            chip_reference=lambda reward=False: (self.ref_p, self.ref_q),
            chip_actual=lambda: (self.actual_p, self.actual_q))
        self.env = SimpleNamespace(command_manager=self.command, num_envs=2, device="cpu")

    def term(self, name, cls):
        params = OmegaConf.to_container(self.cfg.task.reward.tracking[name], resolve=True)
        params.pop("_target_")
        term = cls(**params)
        term._initialize(self.env)
        return term

    def test_elbow_deviation_is_penalized_even_with_hand_pose_unchanged(self):
        for name in ("upper_body_pos", "upper_body_ori", "upper_body_linvel", "upper_body_angvel"):
            term = self.term(name, body_tracking_squared_exp)
            self.assertEqual(len(term.body_indices_tracking), 6)
            for side in ("left", "right"):
                error = torch.zeros(2, 14)
                error[1, self.command.tracking_body_names.index(f"{side}_elbow_link")] = term.sigma
                setattr(self.command, term.error_type, error)
                torch.testing.assert_close(term._compute().flatten(), torch.tensor([1., math.exp(-1/6)]))

    def test_torso_position_reward_is_independent_of_hand_errors(self):
        term = self.term("chip_torso_pos", chip_tracking_reward)
        self.actual_p[:, :2] = 10
        self.actual_p[1, 2, 0] = term.sigma
        torch.testing.assert_close(term._compute().flatten(), torch.tensor([1., math.exp(-1)]))

    def test_torso_yaw_error_reduces_both_orientation_rewards_without_position_error(self):
        for name in ("chip_torso_ori", "chip_torso_ori_precision"):
            term = self.term(name, chip_tracking_reward)
            self.actual_q.copy_(self.ref_q)
            self.actual_q[:, :2] = torch.tensor([0., 1., 0., 0.])
            self.actual_q[1, 2] = torch.tensor([math.cos(term.sigma/2), 0., 0., math.sin(term.sigma/2)])
            torch.testing.assert_close(term._compute().flatten(), torch.tensor([1., math.exp(-1)]))
            self.actual_q.neg_()
            torch.testing.assert_close(term._compute().flatten(), torch.tensor([1., math.exp(-1)]))

    def test_torso_metrics_report_metres_and_radians_at_correct_point(self):
        self.actual_p[:, 1, 0] = .9
        self.actual_p[:, 2, 0] = .03
        self.actual_q[:, 1] = torch.tensor([math.cos(.7/2), 0., 0., math.sin(.7/2)])
        self.actual_q[:, 2] = torch.tensor([math.cos(.2/2), 0., 0., math.sin(.2/2)])
        for quantity, expected in (("torso_error", .03), ("torso_ori_error", .2),
                                   ("right_wrist_error", .9), ("right_wrist_ori_error", .7)):
            term = chip_metric(quantity=quantity, weight=1.0, enabled=False)
            term._initialize(self.env)
            torch.testing.assert_close(term._compute(), torch.full((2, 1), expected))


if __name__ == "__main__":
    unittest.main()
