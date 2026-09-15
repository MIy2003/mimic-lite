"""Export the full deterministic CHIP actor and its measured deployment contract."""
import copy
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import hydra
import numpy as np
import onnx
import torch
from omegaconf import DictConfig, OmegaConf

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.utils.export import export_onnx


@hydra.main(config_path=str(Path(__file__).resolve().parents[1] / "cfg"),
            config_name="replay_chip_loco", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    from chip_checkpoint import configure_chip_checkpoint
    training_cfg = configure_chip_checkpoint(cfg, restore_mode=True)
    if training_cfg is None:
        raise ValueError("Export requires a trained CHIP checkpoint")
    # Export nominal robot parameters, not a random instance of the robot.
    cfg.task.randomization = {}
    aa.init(cfg, auto_rank=True)
    from active_adaptation.helpers import make_env_policy
    env, policy = make_env_policy(cfg.task, cfg.algo, cfg.seed, True, cfg.device,
                                  checkpoint_path=cfg.checkpoint_path)
    output = Path(cfg.deploy_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    actor = copy.deepcopy(policy.get_rollout_policy("deploy")).eval()
    fake = env.observation_spec.rand().to(env.device)
    with VecNorm.freeze():
        export_onnx(actor, fake, str(output / "policy.onnx"))
    a = env.base_env.action_manager
    c = env.base_env.command_manager
    names = list(a.joint_names)
    joint_kp, joint_kd = [], []
    for name in names:
        matches = [act for act in a.asset.cfg.articulation.actuators
                   if any(re.fullmatch(pattern, name) for pattern in act.target_names_expr)]
        if len(matches) != 1:
            raise ValueError(f"Expected one actuator for {name}, got {len(matches)}")
        joint_kp.append(float(matches[0].stiffness))
        joint_kd.append(float(matches[0].damping))
    contract = {
        "format": "mimic_lite_chip_v2" if c.chip.compliance_mode == "wrist_axis" else "mimic_lite_chip_v1",
        "checkpoint": str(Path(cfg.checkpoint_path).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(cfg.checkpoint_path).read_bytes()).hexdigest(),
        "policy_joint_names": names,
        "lower_joint_names": [c.tracking_joint_names[i] for i in c.chip_leg_ids],
        "default_joint_pos": a.default_joint_pos[0,a.joint_ids].cpu().tolist(),
        "default_joint_vel": a.asset.data.default_joint_vel[0,a.joint_ids].cpu().tolist(),
        "action_scale": a.action_scaling.cpu().tolist(), "joint_kp":joint_kp, "joint_kd":joint_kd,
        "control_dt":float(env.step_dt), "physics_dt":float(env.base_env.physics_dt),
        "history_length":int(cfg.task.observation.policy.chip_history.history_length),
        "history_order":["projected_gravity","omega","q","dq","last_action"],
        "history_direction":"oldest_to_newest", "anchor_body":"pelvis",
        "point_bodies":c.chip_body_names, "point_offsets":c.chip_offsets.cpu().tolist(),
        "compliance_scale":c.chip_compliance_scale,
        "physical_compliance_max":list(training_cfg.task.command.chip.compliance_max),
        "inputs":{key: [1, *fake[key].shape[1:]] for key in ("policy", "command")}, "output":"action",
        "compliance_mode": c.chip.compliance_mode,
        "stiffness_axis": ({"command_slice": [54, 57], "frame": "current_point_link",
                            "shared_local_xyz": True, "unit_vector": True, "scale": 1.0,
                            "point_frames": c.chip_body_names,
                            "meaning": "compliant along axis, resist perpendicular disturbances"}
                           if c.chip.compliance_mode == "wrist_axis" else None),
        "normalization":"frozen VecNorm embedded in ONNX",
    }
    model = onnx.load(output / "policy.onnx", load_external_data=True)
    onnx.helper.set_model_props(model, {"mimic_lite_chip_contract":json.dumps(contract)})
    onnx.save_model(model, output / "policy.onnx", save_as_external_data=False)
    (output / "contract.json").write_text(json.dumps(contract, indent=2))
    (output / "source_cfg.yaml").write_text(OmegaConf.to_yaml(cfg))
    # Include the exact nominal simulation robot for deployment-side sim2sim.
    scene = env.base_env.scene
    for field in ("timestep","iterations","ls_iterations","solver","integrator"):
        setattr(scene.spec.option, field, getattr(env.base_env.sim.mj_model.opt, field))
    scene.write(output / "scene")
    # Scene.write copies in-memory assets; this model's meshes are file-backed.
    from mimic_lite.assets.g1 import G1_MJCF_REF_BY_MODE, resolve_asset_reference
    source = Path(resolve_asset_reference(G1_MJCF_REF_BY_MODE[15]))
    source_xml = ET.parse(source).getroot()
    meshdir = source_xml.find("compiler").get("meshdir", "")
    for mesh in ET.parse(output / "scene/scene.xml").getroot().iter("mesh"):
        if not mesh.get("file"):
            continue
        destination = output / "scene/assets" / mesh.get("file")
        if not destination.exists():
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source.parent / meshdir / mesh.get("file"), destination)
    print("CHIP EXPORT " + str(output), flush=True)
    env.close()


if __name__ == "__main__":
    main()
