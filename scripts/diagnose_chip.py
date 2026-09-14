"""Record a bounded, headless CHIP policy rollout for jitter analysis."""
from pathlib import Path
import json

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
    aa.init(cfg, auto_rank=True)
    # Seed before construction too: mass/CoM randomization runs at startup.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    from active_adaptation.helpers import make_env_policy
    env, policy = make_env_policy(cfg.task, cfg.algo, seed=cfg.seed,
                                  headless=cfg.headless, device=cfg.device,
                                  checkpoint_path=cfg.checkpoint_path)
    base = env.base_env
    base.eval()
    env.set_seed(cfg.seed)
    carry = env.reset()
    rollout = policy.get_rollout_policy("eval")
    action = base.action_manager
    command = base.command_manager
    data = action.asset.data
    ids = action.joint_ids
    records = {}
    def record(key, value):
        records.setdefault(key, []).append(value.detach().cpu().numpy().copy())

    with torch.inference_mode(), VecNorm.freeze(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for i in range(int(cfg.get("diagnostic_steps", 550))):
            record("q", data.joint_pos[:, ids])
            record("dq", data.joint_vel[:, ids])
            torque = data.applied_torque if hasattr(data, "applied_torque") else data.actuator_force
            record("torque", torque[:, ids])
            record("root_omega", data.root_com_ang_vel_b)
            record("command", carry["command"])
            record("compliance", command.chip.compliance)
            record("force", command.chip.force)
            ref, _ = command.chip_reference()
            actual, _ = command.chip_actual()
            record("point_error", (actual-ref).norm(dim=-1))
            record("reference_q", command.ref_joint_pos_future_[:, command.obs_current_step_index])
            carry = rollout(carry)
            record("action", carry["action"])
            td, carry = env.step_and_maybe_reset(carry)
            record("done", td["next", "done"])
            record("applied_action", action.applied_action)
            if i % 100 == 0:
                print(f"Diagnostic step {i}", flush=True)
    output = Path(cfg.get("diagnostic_output", "records/chip_jitter/baseline.npz"))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{k: np.stack(v) for k, v in records.items()},
                        joint_names=np.array(action.joint_names),
                        reference_joint_names=np.array(command.tracking_joint_names),
                        action_scale=action.action_scaling.cpu().numpy(),
                        delay=action.delay.cpu().numpy(),
                        alpha=action.alpha.cpu().numpy(),
                        step_dt=np.array(env.step_dt))
    output.with_suffix(".yaml").write_text(OmegaConf.to_yaml(cfg))
    print(json.dumps({"output": str(output), "steps": len(records["q"]),
                      "resets": int(np.stack(records["done"]).sum())}), flush=True)
    env.close()


if __name__ == "__main__":
    main()
