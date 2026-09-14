"""Evaluate every validation clip once, recording local tracking and failures."""
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm


@hydra.main(config_path=str(Path(__file__).resolve().parents[1] / "cfg"),
            config_name="replay_chip_loco", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    for motion in cfg.task.command.motion_cfgs.values():
        motion.full_motion = True
    cfg.task.command.start_from_zero = True
    cfg.task.max_episode_length = 1000000
    aa.init(cfg, auto_rank=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    from active_adaptation.helpers import make_env_policy
    from any4hdmi.dataset.base import MotionSample
    env, policy = make_env_policy(cfg.task, cfg.algo, seed=cfg.seed,
                                  headless=True, device=cfg.device,
                                  checkpoint_path=cfg.checkpoint_path)
    base = env.base_env
    base.eval()
    command = base.command_manager
    dataset = command.dataset
    total = dataset.num_motions
    lengths = dataset.lengths.to(base.device)
    # Longest first keeps the parallel queue occupied until near the end.
    order = torch.argsort(lengths, descending=True)
    assignment = torch.full((env.num_envs,), -1, dtype=torch.long, device=base.device)
    cursor = 0

    def sample_next(env_ids, **kwargs):
        nonlocal cursor
        n = min(len(env_ids), total - cursor)
        assignment[env_ids] = -1
        assignment[env_ids[:n]] = order[cursor:cursor+n]
        cursor += n
        selected = assignment[env_ids].clamp_min(0)
        return MotionSample(motion_id=selected, motion_len=lengths[selected],
                            start_t=torch.ones_like(selected))

    # Full resident datasets support direct global motion IDs. This replaces
    # random training sampling only in this evaluator and covers each clip once.
    dataset.sample_motion = sample_next
    env.set_seed(cfg.seed)
    carry = env.reset()
    rollout = policy.get_rollout_policy("eval")
    rows, samples = {}, []
    output = Path(cfg.get("validation_output", "records/chip_validation/right_c002"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "cfg.yaml").write_text(OmegaConf.to_yaml(cfg))
    (output / "motion_paths.json").write_text(json.dumps([str(p) for p in dataset.motion_paths], indent=2))
    print(f"VALIDATION START clips={total} environments={env.num_envs} max_frames={int(lengths.max())}", flush=True)
    step = 0
    with torch.inference_mode(), VecNorm.freeze(), set_exploration_type(ExplorationType.DETERMINISTIC):
        while len(rows) < total:
            ids = assignment.clone()
            valid = ids >= 0
            age = base.episode_length_buf.clone()
            motion_t = command.t.clone()
            ref_p, ref_q = command.chip_reference()
            p, q = command.chip_actual()
            error = (p-ref_p).norm(dim=-1) * 1000
            dot = (ref_q[:, 1] * q[:, 1]).sum(-1).abs().clamp(0, 1)
            angle = torch.rad2deg(2 * torch.acos(dot))
            samples.append(torch.cat((ids[valid, None].float(), age[valid, None].float(),
                                      error[valid], angle[valid, None]), dim=-1).cpu().numpy())
            carry = rollout(carry)
            td, carry = env.step_and_maybe_reset(carry)
            done = td["next", "done"].flatten() & valid
            term = td["next", "stats", "termination"]
            for e in done.nonzero().flatten().tolist():
                mid = int(ids[e])
                flags = {k: bool(v[e].item()) for k,v in term.items()}
                success = flags.get("motion_timeout", False) and not any(v for k,v in flags.items() if k != "motion_timeout")
                rows[mid] = {"motion_id": mid, "path": str(dataset.motion_paths[mid]),
                             "length_frames": int(lengths[mid]), "observed_steps": int(age[e])+1,
                             "last_reference_frame": int(motion_t[e]),
                             "completed": success, "termination": flags}
            if step % 100 == 0:
                print(f"VALIDATION step={step} finished={len(rows)}/{total} assigned={cursor}", flush=True)
            step += 1
    a = np.concatenate(samples)
    np.savez_compressed(output / "samples.npz", motion_id=a[:,0].astype(np.int32),
                        age=a[:,1].astype(np.int32), point_error_mm=a[:,2:5], right_orientation_deg=a[:,5])
    (output / "episodes.json").write_text(json.dumps([rows[i] for i in range(total)], indent=2))
    def stats(values):
        return {"frames": len(values), "mean_mm": float(values.mean()),
                "rms_mm": float(np.sqrt((values**2).mean())),
                "p95_mm": float(np.percentile(values,95)), "max_mm": float(values.max())}
    summary = {"clips": total, "completed": sum(r["completed"] for r in rows.values()),
               "right_local_all_frames": stats(a[:,3]),
               "right_local_after_2s": stats(a[a[:,1]>=100,3])}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print("VALIDATION DONE " + json.dumps(summary), flush=True)
    env.close()


if __name__ == "__main__":
    main()
