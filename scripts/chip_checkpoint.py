"""Keep CHIP observation semantics consistent with a checkpoint's saved config."""
from pathlib import Path

from omegaconf import OmegaConf, open_dict


def configure_chip_checkpoint(cfg, *, restore_mode=False):
    """Replay restores the saved mode; training rejects cross-mode resumes.

    Old configs without compliance_mode are isotropic (54D command). New axis
    checkpoints are 57D. This does not migrate weights or overwrite evaluation
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
    saved_mode = OmegaConf.select(saved, "task.command.chip.compliance_mode", default="isotropic")
    if saved_mode not in ("isotropic", "wrist_axis"):
        raise ValueError(f"Unsupported checkpoint compliance_mode: {saved_mode}")
    requested_mode = chip.get("compliance_mode", "isotropic")
    if not restore_mode and requested_mode != saved_mode:
        raise ValueError(
            f"Cannot resume {saved_mode} checkpoint into {requested_mode}: CHIP command dimensions/"
            "semantics differ. Start wrist_axis training with checkpoint_path=null, or explicitly "
            f"set task.command.chip.compliance_mode={saved_mode} to resume the old mode."
        )
    with open_dict(cfg), open_dict(chip):
        cfg.checkpoint_path = str(path)
        chip.compliance_mode = saved_mode
    print(f"CHIP checkpoint mode: {saved_mode} ({57 if saved_mode == 'wrist_axis' else 54}D command)")
    return saved
