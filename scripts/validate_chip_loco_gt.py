"""Compare four real motion clips' loaded GT with independent MuJoCo FK (CPU)."""
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ANY4HDMI_CACHE_BUILD_DEVICE", "cpu")

import mujoco
import numpy as np
import yaml
from any4hdmi import load_any4hdmi_dataset
from mjhub import resolve_asset_reference


def main():
    framework = Path(__file__).resolve().parents[3]
    prepared = framework / ".cache/chip_loco"
    cfg = yaml.safe_load((prepared / "task/chip_loco_train_smoke.yaml").read_text())
    names = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
    offsets = np.asarray(cfg["command"]["chip"]["point_offsets"])
    report = {}
    for pool, config in cfg["command"]["motion_cfgs"].items():
        root = Path(config["path"])
        manifest = json.loads((root / "manifest.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(resolve_asset_reference(manifest["mjcf"])))
        data = mujoco.MjData(model)
        dataset = load_any4hdmi_dataset(root_path=str(root), target_fps=50, base_dir=framework,
                                       num_envs=1, full_motion=True, filenames_path=config["filenames_path"])
        clip = Path(config["filenames_path"]).read_text().strip()
        with np.load(root / "motions" / clip, allow_pickle=False) as archive:
            qpos = archive["qpos"]
        assert np.isfinite(qpos).all()
        np.testing.assert_allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1, atol=1e-3)
        joints = [model.joint(n).qposadr.item() for n in dataset.joint_names]
        # Explicit name mapping: never assume dataset and simulator joint order match.
        expected_names = [model.joint(n).name for n in dataset.joint_names]
        source_columns = [manifest["qpos_names"].index(n) for n in expected_names]
        np.testing.assert_allclose(dataset.data.joint_pos.cpu().numpy(), qpos[:, source_columns], atol=1e-5)
        max_pos_error, max_ori_error = 0., 0.
        for frame in np.linspace(0, len(qpos) - 1, 7, dtype=int):
            data.qpos[:7] = qpos[frame, :7]
            data.qpos[joints] = qpos[frame, source_columns]
            mujoco.mj_forward(model, data)
            for name, offset in zip(names, offsets):
                body_id, loaded_id = model.body(name).id, dataset.body_names.index(name)
                p = dataset.data.body_pos_w[frame, loaded_id].cpu().numpy()
                q = dataset.data.body_quat_w[frame, loaded_id].cpu().numpy().astype(np.float64)
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, q)
                loaded_point = p + rotation.reshape(3, 3) @ offset
                expected_point = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset
                error = float(np.linalg.norm(loaded_point - expected_point))
                ori_error = float(1 - abs(np.dot(q, data.xquat[body_id])))
                max_pos_error = max(max_pos_error, error)
                max_ori_error = max(max_ori_error, abs(ori_error))
        assert max_pos_error < 1e-4, max_pos_error
        assert max_ori_error < 1e-5, max_ori_error
        speeds = dataset.data.joint_vel.cpu().numpy()
        assert np.isfinite(speeds).all()
        report[pool] = {"clip": clip, "frames": len(qpos),
                        "max_offset_point_fk_error_m": max_pos_error,
                        "max_quaternion_dot_error": max_ori_error,
                        "max_abs_joint_velocity_rad_s": float(np.abs(speeds).max()),
                        "root_height_range_m": [float(qpos[:, 2].min()), float(qpos[:, 2].max())]}
    (prepared / "gt_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
