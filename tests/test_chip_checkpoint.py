"""Checkpoint mode resolution without constructing a simulator or loading weights."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

spec = importlib.util.spec_from_file_location(
    "chip_checkpoint", Path(__file__).parents[1] / "scripts/chip_checkpoint.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CheckpointModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = OmegaConf.create({"checkpoint_path": str(self.root / "checkpoint.pt"), "task": {
            "command": {"chip": {"compliance_mode": "wrist_axis", "max_force": 0,
                                 "fixed_stiffness_axis": [1, 0, 0]}}}})
        self.patch = patch("active_adaptation.utils.wandb.parse_checkpoint_path", side_effect=lambda p: p)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def save(self, chip):
        OmegaConf.save(OmegaConf.create({"task": {"command": {"chip": chip}}}), self.root / "cfg.yaml")

    def test_old_replay_restores_54d_without_changing_overrides(self):
        self.save({"compliance_max": [.02, .02, 0]})
        module.configure_chip_checkpoint(self.cfg, restore_mode=True)
        self.assertEqual(self.cfg.task.command.chip.compliance_mode, "isotropic")
        self.assertEqual(self.cfg.task.command.chip.max_force, 0)

    def test_new_replay_restores_axis_mode(self):
        self.save({"compliance_mode": "wrist_axis"})
        self.cfg.task.command.chip.compliance_mode = "isotropic"
        module.configure_chip_checkpoint(self.cfg, restore_mode=True)
        self.assertEqual(self.cfg.task.command.chip.compliance_mode, "wrist_axis")
        self.assertEqual(list(self.cfg.task.command.chip.fixed_stiffness_axis), [1, 0, 0])

    def test_training_rejects_cross_mode_resume(self):
        self.save({})
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            module.configure_chip_checkpoint(self.cfg)
        self.cfg.task.command.chip.compliance_mode = "isotropic"
        module.configure_chip_checkpoint(self.cfg)

    def test_missing_metadata_fails_instead_of_guessing(self):
        with self.assertRaises(FileNotFoundError):
            module.configure_chip_checkpoint(self.cfg, restore_mode=True)

    def test_fresh_training_is_unchanged(self):
        self.cfg.checkpoint_path = None
        self.assertIsNone(module.configure_chip_checkpoint(self.cfg))
        self.assertEqual(self.cfg.task.command.chip.compliance_mode, "wrist_axis")

    def test_sixd_restores_units_but_preserves_eval_wrench_overrides(self):
        self.save(dict(compliance_mode='wrist_6d',compliance_6d_max=[.02,.02,.02,.2,.2,.2],
            compliance_6d_scale=[50.,50.,50.,5.,5.,5.],left_compliance_mode='rigid'))
        module.configure_chip_checkpoint(self.cfg,restore_mode=True)
        self.assertEqual(self.cfg.task.command.chip.compliance_mode,'wrist_6d')
        self.assertEqual(self.cfg.task.command.chip.compliance_6d_scale[-1],5.)
        self.assertEqual(self.cfg.task.command.chip.max_force,0)
        module.configure_chip_checkpoint(self.cfg)
        self.cfg.task.command.chip.compliance_6d_scale[-1]=10.
        with self.assertRaisesRegex(ValueError,'changed six-dimensional'):
            module.configure_chip_checkpoint(self.cfg)

    def test_binary_joint_resume_and_legacy_adapter_guard(self):
        self.cfg.task.command.chip = OmegaConf.create(dict(
            compliance_mode="wrist_joint_binary", compliance_6d_max=[.02]*3+[.8]*3,
            compliance_6d_scale=[50.]*3+[1.25]*3, left_compliance_mode="rigid"))
        self.save(OmegaConf.to_container(self.cfg.task.command.chip))
        module.configure_chip_checkpoint(self.cfg)
        with self.assertRaisesRegex(ValueError, "75D wrist-joint deployment adapter"):
            module.configure_chip_checkpoint(self.cfg,restore_mode=True)
        self.cfg.task.command.chip.compliance_mode="wrist_6d"
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            module.configure_chip_checkpoint(self.cfg)


if __name__ == "__main__":
    unittest.main()
