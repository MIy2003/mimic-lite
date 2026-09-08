"""Render CHIP application points from the actual MJCF and chip.yaml offsets."""
import argparse
import json
import os
import importlib.util
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import mujoco
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from mjhub import resolve_asset_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/chip_points"))
    parser.add_argument("--payload", action="store_true", help="Render added wrist masses and close-up views")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(Path(__file__).resolve().parents[1] / "cfg/task/chip.yaml")
    offsets = np.array(config.command.chip.point_offsets, dtype=float)
    xml = resolve_asset_reference("hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml")
    spec = mujoco.MjSpec.from_file(str(xml))
    if args.payload:
        path = Path(__file__).resolve().parents[1] / "mimic_lite/assets/chip_payload.py"
        module_spec = importlib.util.spec_from_file_location("chip_payload", path)
        payload = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(payload)
        spec = payload.add_wrist_payload(spec)
    model = spec.compile()
    model.vis.global_.offwidth = 800
    model.vis.global_.offheight = 900
    model.vis.headlight.ambient[:] = .45
    model.mat_rgba[:, 3] = .45
    model.geom_rgba[:, 3] = .45
    data = mujoco.MjData(model)
    # Display pose only: bend elbows to separate the hands from the body.
    pose = {"left_elbow_joint": .9, "right_elbow_joint": .9,
            "left_shoulder_roll_joint": .15, "right_shoulder_roll_joint": -.15}
    for name, value in pose.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)
    names = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
    labels = ["L", "R", "T"]
    colors = [[.1, .85, .35, 1], [1, .35, .12, 1], [.25, .6, 1, 1]]
    records = []
    for name, offset in zip(names, offsets):
        bid = model.body(name).id
        origin = data.xpos[bid].copy()
        point = origin + data.xmat[bid].reshape(3, 3) @ offset
        records.append(dict(body=name, offset_local_m=offset.tolist(),
                            link_origin_world_m=origin.tolist(), application_world_m=point.tolist()))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 25)
    small = ImageFont.truetype(font_path, 21)
    title = ImageFont.truetype(font_path, 38)
    canvas = Image.new("RGB", (2400, 1440 if args.payload else 1120), "#151a22")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 20), "CHIP force application points | G1 mode-15", font=title, fill="white")
    draw.text((35, 73), "Large colored sphere = application point   |   Small white sphere = link origin   |   Line = local offset", font=font, fill="#cbd5e1")
    wrist_id = model.body("right_wrist_yaw_link").id
    wrist_rotation = data.xmat[wrist_id].reshape(3, 3)
    def world(local):
        return data.xpos[wrist_id] + wrist_rotation @ np.asarray(local)
    mount = world([.0415, -.003, 0])
    if args.payload:
        draw.rectangle((0, 0, 2400, 119), fill="#151a22")
        draw.text((35, 20), "CHIP wrist payload +517 g | mounting reference = hand attachment", font=title, fill="white")
        draw.text((35, 73), "Yellow: mount  |  Magenta: 237 g (+2.5 mm)  |  Cyan: 280 g (+17.5 mm)  |  Orange R: force point (+30.4 mm)", font=font, fill="#cbd5e1")
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[2] = 1
    options.sitegroup[:] = 0
    with mujoco.Renderer(model, height=900, width=800) as renderer:
        views = [(180, -5, "Front view"), (90, -5, "Side view"), (140, -15, "Oblique view")]
        if args.payload:
            views = [(140, -15, "Overview"), (90, -5, "Right wrist close-up"), (150, -25, "Right wrist oblique")]
        for view, (azimuth, elevation, caption) in enumerate(views):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0, 0, .85]
            camera.distance = 2.1
            if args.payload and view > 0:
                camera.lookat[:] = world([.075, -.003, 0])
                camera.distance = .43
            camera.azimuth = azimuth
            camera.elevation = elevation
            renderer.update_scene(data, camera=camera, scene_option=options)
            scene = renderer.scene
            for record, color, label in zip(records, colors, labels):
                p = np.array(record["application_world_m"])
                origin = np.array(record["link_origin_world_m"])
                scale = .35 if args.payload and view > 0 else 1.
                for position, radius, rgba, text in [(p, .022*scale, color, label), (origin, .012*scale, [1, 1, 1, 1], "")]:
                    g = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.array([radius]*3), position, np.eye(3).flatten(), np.array(rgba, np.float32))
                    g.label = text
                    scene.ngeom += 1
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array(color, np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, .004*scale, origin, p)
                scene.ngeom += 1
            if args.payload:
                # Sensor envelope is visual-only, not a collision or inertia model.
                g = scene.geoms[scene.ngeom]
                cylinder_frame = wrist_rotation @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER,
                                   np.array([.041, .0127, 0]), world(payload.PAYLOAD_POSITIONS[1]),
                                   cylinder_frame.flatten(), np.array([.1, .85, 1., .22], np.float32))
                scene.ngeom += 1
                for point, rgba, label in [(mount, [1., .85, .1, 1.], "Mount"),
                    (world(payload.PAYLOAD_POSITIONS[0]), [1., .1, .8, 1.], "237g"),
                    (world(payload.PAYLOAD_POSITIONS[1]), [.1, .85, 1., 1.], "280g")]:
                    g = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, .002),
                                       point, np.eye(3).flatten(), np.array(rgba, np.float32))
                    g.label = ""  # Nearby mass centers need the separate dimension diagram below.
                    scene.ngeom += 1
            canvas.paste(Image.fromarray(renderer.render()), (800*view, 120))
            draw.text((800*view+25, 135), caption, font=font, fill="white")
    for i, (label, color, offset) in enumerate(zip(labels, colors, offsets)):
        x = 30 + i*800
        rgb = tuple(int(v*255) for v in color[:3])
        draw.text((x, 1030), f"{label}: {names[i]}", font=small, fill=rgb)
        draw.text((x, 1065), f"Local offset: {offset.tolist()} m", font=small, fill="white")
    if args.payload:
        draw.rectangle((0, 1020, 2400, 1120), fill="#151a22")
        draw.text((30, 1030), "Mount local xyz: [41.5, -3, 0] mm   |   Added masses: [44, -3, 0] / [59, -3, 0] mm", font=font, fill="white")
        draw.text((30, 1070), "Cyan cylinder: visual only. Stock hand geometry remains, but its 170 g inertia is removed. Payload: point-mass approximation.", font=small, fill="#cbd5e1")
        draw.text((35, 1130), "Axial dimensions from mounting reference (local +X; enlarged schematic, not hand geometry)", font=font, fill="white")
        x0, scale_mm, y = 300, 45, 1280
        draw.line((x0-100, y, x0+32*scale_mm, y), fill="#b0bdce", width=3)
        draw.polygon([(x0+32*scale_mm,y), (x0+32*scale_mm-16,y-9), (x0+32*scale_mm-16,y+9)], fill="white")
        for distance, color, label, label_y in [(0, "#ffda19", "Mount: 0 mm", 1185),
                (2.5, "#ff19cc", "237 g: +2.5 mm", 1350),
                (17.5, "#19d9ff", "280 g: +17.5 mm", 1185),
                (30.4, "#ff591f", "Force: +30.4 mm", 1350)]:
            x = int(x0+distance*scale_mm)
            draw.line((x, y, x, label_y+30 if label_y<y else label_y-8), fill=color, width=3)
            draw.ellipse((x-7,y-7,x+7,y+7), fill=color)
            draw.text((x-65,label_y),label,font=font,fill=color)
        draw.text((1800, 1210), "Added total: 517 g", font=font, fill="white")
        draw.text((1800, 1260), "Payload COM: +10.624 mm", font=small, fill="white")
    stem = "chip_wrist_payload" if args.payload else "chip_force_points"
    canvas.save(args.output / f"{stem}.png")
    metadata = dict(model=str(xml), display_pose=pose, points=records)
    if args.payload:
        metadata.update(payload_masses_kg=payload.PAYLOAD_MASSES.tolist(),
                        payload_local_m=payload.PAYLOAD_POSITIONS.tolist(), mount_local_m=[.0415, -.003, 0],
                        wrist_mass_kg=float(model.body_mass[wrist_id]), wrist_com_local_m=model.body_ipos[wrist_id].tolist(),
                        visual_envelope_only=True)
    (args.output / f"{stem}.json").write_text(json.dumps(metadata, indent=2))
    print(args.output.resolve() / f"{stem}.png")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
