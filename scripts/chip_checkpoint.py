"""Keep CHIP observation semantics consistent with a checkpoint's saved config."""
from pathlib import Path

from omegaconf import OmegaConf, open_dict


def configure_chip_checkpoint(cfg, *, restore_mode=False):
    """Replay restores the saved mode; training rejects cross-mode resumes.

    Old configs without compliance_mode are isotropic (54D command). New axis
    checkpoints are 57D; wrist_6d is 69D with leg commands or 59D with five poses.
    This does not migrate weights or overwrite evaluation
    force/compliance/fixed-axis overrides.
    """
    chip = OmegaConf.select(cfg, "task.command.chip")
    if chip is None or not cfg.get("checkpoint_path"):
        return None
    from active_adaptation.utils.wandb import parse_checkpoint_path
    from active_adaptation.utils.checkpoint_cfg import find_run_cfg_yaml

    path = parse_checkpoint_path(cfg.checkpoint_path)
    if path is None:
        raise FileNotFoundError("Could not resolve CHIP checkpoint")
    path = Path(path).expanduser().resolve()
    sidecar = find_run_cfg_yaml(path)
    if sidecar is None:
        raise FileNotFoundError(f"CHIP requires cfg.yaml next to {path} to identify its observation contract")
    saved = OmegaConf.load(sidecar)
    if OmegaConf.select(saved, "task.command.chip") is None:
        raise ValueError(f"{sidecar} is not a CHIP checkpoint config")
    five_target = "mimic_lite.ChipFivePointTracking"
    saved_five = OmegaConf.select(saved, "task.command._target_") == five_target
    requested_five = OmegaConf.select(cfg, "task.command._target_") == five_target
    if not restore_mode and saved_five != requested_five:
        raise ValueError("Cannot resume between three-point and five-point CHIP contracts; start with checkpoint_path=null")
    if saved_five:
        contract_keys = ("task.command._target_", "task.command.foot_point_offsets",
                         "task.command.chip.point_offsets", "task.command.anchor_body_name",
                         "task.command.root_body_name", "task.command.tracking_body_names",
                         "task.command.obs_body_names", "task.command.tracking_joint_names",
                         "task.shared.tracking_body_names", "task.shared.lower_body_names",
                         "task.observation.command", "task.observation.priv",
                         "task.observation.policy.chip_history.history_length")
        for key in contract_keys:
            value = OmegaConf.select(saved, key)
            if not restore_mode and OmegaConf.select(cfg, key) != value:
                raise ValueError(f"Cannot resume with changed five-point contract: {key}")
            if restore_mode:
                OmegaConf.update(cfg, key, value, merge=False, force_add=True)
        if restore_mode:
            # Legacy replay tasks can contain already-resolved hip/toe names.
            # Restore the saved terms as well as the new foot observation IDs.
            for key in ("task.reward", "task.termination"):
                OmegaConf.update(cfg, key, OmegaConf.select(saved, key), merge=False, force_add=True)
    elif restore_mode and requested_five:
        raise ValueError("Cannot restore a three-point checkpoint into a five-point replay task")
    saved_mode = OmegaConf.select(saved, "task.command.chip.compliance_mode", default="isotropic")
    if saved_mode not in ("isotropic", "wrist_axis", "wrist_6d", "wrist_joint_binary"):
        raise ValueError(f"Unsupported checkpoint compliance_mode: {saved_mode}")
    if restore_mode and saved_mode == "wrist_joint_binary":
        raise ValueError("Binary wrist checkpoints need a 75D wrist-joint deployment adapter; legacy CHIP replay/export is not supported yet")
    requested_mode = chip.get("compliance_mode", "isotropic")
    if not restore_mode and requested_mode != saved_mode:
        raise ValueError(
            f"Cannot resume {saved_mode} checkpoint into {requested_mode}: CHIP command dimensions/"
            "semantics differ. Start new-mode training with checkpoint_path=null, or explicitly "
            f"set task.command.chip.compliance_mode={saved_mode} to resume the old mode."
        )
    if saved_mode == "wrist_joint_binary":
        for key in ("task.command._target_", "task.command.wrist_joint", "task.observation.command", "task.observation.priv"):
            if OmegaConf.select(cfg, key) != OmegaConf.select(saved, key):
                raise ValueError(f"Cannot resume with changed wrist-joint contract: {key}")
    with open_dict(cfg), open_dict(chip):
        cfg.checkpoint_path = str(path)
        chip.compliance_mode = saved_mode
        if saved_mode in ("wrist_6d", "wrist_joint_binary"):
            for key in ("compliance_6d_max", "compliance_6d_scale", "left_compliance_mode"):
                value = OmegaConf.select(saved, f"task.command.chip.{key}")
                if value is None:
                    raise ValueError(f"Six-dimensional checkpoint lacks {key}")
                if not restore_mode and chip.get(key) != value:
                    raise ValueError(f"Cannot resume with changed six-dimensional {key}; command semantics differ")
                chip[key] = value
    dimension = {"isotropic": 54, "wrist_axis": 57, "wrist_6d": 69, "wrist_joint_binary": 75}[saved_mode]
    if saved_five:
        dimension = 59
    print(f"CHIP checkpoint mode: {saved_mode} ({dimension}D command)")
    return saved
