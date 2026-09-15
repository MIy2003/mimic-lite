"""Fail before loading incompatible CHIP/legacy geometry into the new controller."""
from pathlib import Path
from omegaconf import OmegaConf


def check_three_point_checkpoint(cfg):
    target = OmegaConf.select(cfg, "task.command._target_")
    if target not in ("mimic_lite.ThreePointTracking", "mimic_lite.ThreePointChipTracking"):
        return
    if not OmegaConf.select(cfg, "algo.goal_body_actor", default=False):
        raise ValueError("ThreePointTracking requires the train_three_point/replay_three_point configuration")
    compliant = target == "mimic_lite.ThreePointChipTracking"
    if OmegaConf.select(cfg, "algo.goal_body_compliance", default=False) != compliant:
        raise ValueError("goal_body_compliance must match the task's 315D/318D command contract")
    if compliant and OmegaConf.select(cfg, "task.command.chip.compliance_mode") != "isotropic":
        raise ValueError("Three-point CHIP requires isotropic compliance (no axis)")
    if not cfg.get("checkpoint_path"):
        return
    from active_adaptation.utils.wandb import parse_checkpoint_path
    from active_adaptation.utils.checkpoint_cfg import find_run_cfg_yaml
    resolved = parse_checkpoint_path(cfg.checkpoint_path)
    if resolved is None:
        raise FileNotFoundError("Could not resolve three-point checkpoint")
    path = Path(resolved).expanduser().resolve()
    sidecar = find_run_cfg_yaml(path)
    if sidecar is None:
        raise ValueError("Keep the three-point checkpoint's cfg.yaml beside its weights")
    saved = OmegaConf.load(sidecar)
    if OmegaConf.select(saved, "algo.goal_body_compliance", default=False) != compliant:
        raise ValueError("Checkpoint goal_body_compliance differs (315D vs 318D)")
    if compliant:
        for key in ("compliance_mode", "compliance_scale", "point_offsets"):
            field = f"task.command.chip.{key}"
            if OmegaConf.select(saved, field) != OmegaConf.select(cfg, field):
                raise ValueError(f"Three-point checkpoint contract mismatch: {field}")
    for key in ("task.command._target_", "task.command.point_offsets", "task.command.future_steps",
                "task.robot.name", "algo.goal_body_actor", "algo.goal_body_embed_dim",
                "algo.goal_body_num_heads", "algo.goal_body_num_layers", "algo.goal_body_ff_dim"):
        if OmegaConf.select(saved, key) != OmegaConf.select(cfg, key):
            raise ValueError(f"Three-point checkpoint contract mismatch: {key}; do not load CHIP/legacy weights")
    cfg.checkpoint_path = str(path)
