"""Convert one CHIP episode to named-joint any4hdmi qpos; validate via FK."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from any4hdmi.utils.mjcf import resolve_mjcf_path, qpos_names_from_model
from any4hdmi.core.format import write_manifest

PROJECT = Path(__file__).resolve().parents[1]
FRAMEWORK = PROJECT.parents[1]
MJCF = "hf://elijahgalahad/g1_xmls@a57ffbdfc0a9379a781f37f4513a82b92ea93591/g1-mode_13_15.xml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    source = args.episode.resolve() / "chip_motion_50hz.npz"
    with np.load(source, allow_pickle=False) as archive:
        d = {k: archive[k] for k in archive.files}
    names = d["joint_names"].tolist()
    bodies = d["body_names"].tolist()
    fps = float(d["fps"].item())
    if fps != 50 or len(set(names)) != 29:
        raise ValueError("Expected 50 Hz and 29 unique named joints")
    model = mujoco.MjModel.from_xml_path(str(resolve_mjcf_path(MJCF)))
    qnames = qpos_names_from_model(model)
    pelvis = bodies.index("pelvis")
    qpos = np.concatenate([
        d["body_pos_w"][:, pelvis], d["body_quat_w"][:, pelvis],
        d["joint_pos"][:, [names.index(n) for n in qnames[7:]]],
    ], axis=-1).astype(np.float32)
    if qpos.shape[1] != model.nq or not np.isfinite(qpos).all():
        raise ValueError("Invalid qpos")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-3):
        raise ValueError("Root quaternion is not normalized")
    # Verify WXYZ convention, joint ordering and root/body-frame interpretation.
    data = mujoco.MjData(model)
    errors = []
    for i in range(len(qpos)):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        for body in ("pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"):
            bid = model.body(body).id
            errors.append(np.linalg.norm(data.xpos[bid] - d["body_pos_w"][i, bodies.index(body)]))
    max_error = float(max(errors))
    # This recording's wrist-yaw origins differ by 5 mm from the training XML.
    # Keep the training model and regenerate references from qpos; do not shift
    # recorded joint angles to force the two body-origin definitions to agree.
    if max_error > 0.0051:
        raise ValueError(f"Source/model FK mismatch: {max_error:.6f} m")
    output = FRAMEWORK / ".cache/chip_episode" / args.episode.name
    (output / "motions").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "motions/episode.npz", qpos=qpos)
    write_manifest(output, dataset_name=args.episode.name, mjcf=MJCF,
                   timestep=1/fps, qpos_names=qnames, num_motions=1,
                   source={"path": str(source), "fk_max_error_m": max_error},
                   total_hours=(len(qpos)-1)/fps/3600)
    cfg = yaml.safe_load((PROJECT / "cfg/task/chip.yaml").read_text())
    cfg["name"] = "chip_episode"
    cfg["max_episode_length"] = max(1000, len(qpos)+10)
    cfg["command"]["motion_cfgs"] = {
        "episode": {"path": str(output), "weight": 1.0, "full_motion": True}}
    (output / "task").mkdir(exist_ok=True)
    (output / "task/chip_episode.yaml").write_text("# @package task\n" + yaml.safe_dump(cfg, sort_keys=False))
    print(json.dumps({"frames": len(qpos), "duration_s": (len(qpos)-1)/fps,
                      "fk_max_error_m": max_error, "config_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
