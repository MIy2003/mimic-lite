"""Whole-body reward coverage and per-body terrain contact regression checks."""
from pathlib import Path
from types import SimpleNamespace
import math
import re
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from mimic_lite.assets.g1 import _build_g1_chip_payload_cfg
from mimic_lite.tasks.rewards.track import body_tracking_squared_exp
from mimic_lite.tasks.rewards.common import undesired_ground_contacts


class ChipRewardTests(unittest.TestCase):
    def setUp(self):
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "cfg"), version_base=None):
            self.cfg = compose(config_name="train", overrides=["task=chip"])
        names = ["pelvis", "torso_link"] + [
            f"{side}_{part}_link" for side in ("left", "right")
            for part in ("hip_yaw", "knee", "toe", "shoulder_yaw", "elbow", "wrist_yaw")
        ]
        self.command = SimpleNamespace(tracking_body_names=names)
        self.env = SimpleNamespace(command_manager=self.command, num_envs=2, device="cpu")

    def make_term(self, name):
        params = OmegaConf.to_container(self.cfg.task.reward.tracking[name], resolve=True)
        params.pop("_target_")
        term = body_tracking_squared_exp(**params)
        term._initialize(self.env)
        setattr(self.command, term.error_type, torch.zeros(2, 14))
        return term

    def test_elbow_error_reduces_pose_rewards_with_wrist_unchanged(self):
        for name in ("body_pos", "body_ori"):
            term = self.make_term(name)
            error = getattr(self.command, term.error_type)
            error[1, self.command.tracking_body_names.index("right_elbow_link")] = term.sigma
            values = term._compute().flatten()
            self.assertAlmostEqual(values[0].item(), 1.0)
            self.assertAlmostEqual(values[1].item(), math.exp(-1 / 14), places=6)

    def test_every_tracked_body_contributes_to_velocity_rewards(self):
        for name in ("body_linvel", "body_angvel"):
            term = self.make_term(name)
            error = getattr(self.command, term.error_type)
            for body in range(14):
                error.zero_()
                error[1, body] = term.sigma * 2
                self.assertAlmostEqual(term._compute()[1].item(), math.exp(-4 / 14), places=6)

    def test_global_anchor_reward_tracks_pelvis(self):
        for name in ("root_pos", "root_ori"):
            term = self.make_term(name)
            error = getattr(self.command, term.error_type)
            error[:, 1] = 10  # Torso alone must not stand in for pelvis tracking.
            error[1, 0] = term.sigma
            torch.testing.assert_close(term._compute().flatten(), torch.tensor([1., math.exp(-1)]))

    def test_ground_penalty_counts_bodies_not_history_samples(self):
        force = torch.zeros(2, 3, 4, 3)
        force[1, 0, :, 0] = 2
        force[1, 1, 0, 1] = 3
        force[1, 2, :, 2] = .5
        self.env.scene = SimpleNamespace(sensors={
            "undesired_ground_contact": SimpleNamespace(data=SimpleNamespace(force_history=force))
        })
        term = undesired_ground_contacts(weight=-.1)
        term._initialize(self.env)
        torch.testing.assert_close(term.compute(), torch.tensor([[0.], [-.2]]))

    def test_ground_sensor_excludes_feet_hands_but_includes_elbows_and_pelvis(self):
        asset = SimpleNamespace(config=SimpleNamespace(spec_fn=lambda: None), sensors=())
        with patch("mimic_lite.assets.g1._build_g1_cfg", return_value=asset):
            sensor = _build_g1_chip_payload_cfg("mjlab").sensors[-1]
        def included(name):
            return bool(re.fullmatch(sensor.primary.pattern, name)) and not any(
                re.fullmatch(pattern, name) for pattern in sensor.primary.exclude)
        for name in ("pelvis", "torso_link", "right_elbow_link", "left_knee_link"):
            self.assertTrue(included(name), name)
        for name in ("left_ankle_roll_link", "right_toe_link", "right_wrist_yaw_link", "right_hand_palm_link"):
            self.assertFalse(included(name), name)
        self.assertEqual(sensor.secondary.pattern, "terrain")
        self.assertEqual(sensor.reduce, "netforce")


if __name__ == "__main__":
    unittest.main()
