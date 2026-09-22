"""Validate portable splits and generate CHIP task configs without editing source data."""
import argparse
import copy
import json
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
FRAMEWORK = PROJECT.parents[1]
POOLS = ("core_motiondecode", "core_sonic", "support_motiondecode", "support_sonic")
SPLIT_DIR = "physical_rollout_inherited_family_80_10_10_seed20260828_v1"


def load_task_template(name):
    if name in ("three_point_chip", "three_point_chip_axis"):
        # Flatten inheritance BEFORE replacing motion_cfgs, otherwise the base
        # task's default motion pool is silently merged into the four real pools.
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        with initialize_config_dir(config_dir=str(PROJECT / "cfg"), version_base=None):
            cfg = compose(config_name=f"task/{name}")
        return OmegaConf.to_container(cfg.task, resolve=False)
    return yaml.safe_load((PROJECT / f"cfg/task/{name}.yaml").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=FRAMEWORK.parent / "loco_manip_physical_rollout_accepted_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-template", choices=("chip", "three_point", "three_point_chip", "three_point_chip_axis"), default="chip")
    args = parser.parse_args()
    template = load_task_template(args.task_template)
    root = args.root.resolve()
    output = (args.output or FRAMEWORK / f".cache/{args.task_template}_loco").resolve()
    splits, all_paths = {}, set()
    for split in ("train", "val", "test"):
        grouped = {pool: [] for pool in POOLS}
        for line in (root / "splits" / SPLIT_DIR / f"{split}_relative_paths.txt").read_text().splitlines():
            rel = Path(line.strip())
            if not line.strip():
                continue
            if rel.is_absolute() or ".." in rel.parts or len(rel.parts) < 4:
                raise ValueError(f"Unsafe path: {rel}")
            if rel.parts[0] != "pools" or rel.parts[1] not in POOLS or rel.parts[2] != "motions":
                raise ValueError(f"Unexpected layout: {rel}")
            path = (root / rel).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.suffix != ".npz":
                raise ValueError(f"Missing/invalid motion: {path}")
            if path in all_paths:
                raise ValueError(f"Duplicate motion or split leakage: {path}")
            all_paths.add(path)
            grouped[rel.parts[1]].append(Path(*rel.parts[3:]).as_posix())
        if any(not entries for entries in grouped.values()):
            raise ValueError(f"Empty pool in {split}")
        splits[split] = grouped
    actual = set(p.resolve() for p in (root / "pools").glob("*/motions/**/*.npz"))
    if actual != all_paths:
        raise ValueError("Split union does not match physical NPZ files")
    manifests = {pool: json.loads((root / "pools" / pool / "manifest.json").read_text()) for pool in POOLS}
    first = manifests[POOLS[0]]
    for pool, manifest in manifests.items():
        for key in ("mjcf", "qpos_names", "qpos_dim", "timestep"):
            if manifest[key] != first[key]:
                raise ValueError(f"Inconsistent {key}: {pool}")
    if first["qpos_dim"] != 36 or abs(first["timestep"] - .02) > 1e-8:
        raise ValueError("Expected G1 36D qpos at 50 Hz")
    (output / "task").mkdir(parents=True, exist_ok=True)
    report = {"root": str(root), "unique_motions": len(all_paths), "splits": {}}
    for split, grouped in splits.items():
        report["splits"][split] = {pool: len(entries) for pool, entries in grouped.items()}
        for smoke in (False, True):
            name = f"{args.task_template}_loco_{split}" + ("_smoke" if smoke else "")
            cfg = copy.deepcopy(template)
            cfg["name"] = name
            cfg["command"]["motion_cfgs"] = {}
            for pool, entries in grouped.items():
                selected = entries[:1] if smoke else entries
                listing = output / f"{name}_{pool}.txt"
                listing.write_text("\n".join(selected) + "\n")
                cfg["command"]["motion_cfgs"][pool] = {
                    "path": str(root / "pools" / pool), "filenames_path": str(listing),
                    # CHIP keeps clip-count weights; Goal–Body keeps its source physical-pool recipe.
                    "weight": (dict(zip(POOLS, (.4, .4, .04, .16)))[pool]
                               if args.task_template.startswith("three_point") else len(entries)),
                    "full_motion": False,
                }
            (output / "task" / f"{name}.yaml").write_text("# @package task\n" + yaml.safe_dump(cfg, sort_keys=False))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Generated configs: {output}/task (rerun after changing {args.task_template}.yaml)")


if __name__ == "__main__":
    main()
