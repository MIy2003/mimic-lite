"""Compare the deployment adapter against live training observations/actions."""
import json
import sys
from pathlib import Path
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type
from scipy.spatial.transform import Rotation
import active_adaptation as aa
from active_adaptation.learning.modules.vecnorm import VecNorm


@hydra.main(config_path=str(Path(__file__).resolve().parents[1]/"cfg"),config_name="replay_chip_loco",version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg,False)
    cfg.task.randomization = {}
    aa.init(cfg,auto_rank=True)
    from active_adaptation.helpers import make_env_policy
    repo=Path(cfg.deploy_repo)
    sys.path.insert(0,str(repo/"src"))
    from mimic_lite_chip_policy import MimicLiteChipPolicy
    from chip_reference import REAL_JOINT_NAMES, ChipReferenceFrame
    contract=json.loads((repo/"models/mimic_lite_chip/contract.json").read_text())
    env,policy=make_env_policy(cfg.task,cfg.algo,cfg.seed,True,cfg.device,checkpoint_path=cfg.checkpoint_path)
    env.base_env.eval();env.set_seed(cfg.seed)
    c=env.base_env.command_manager;a=env.base_env.action_manager;data=a.asset.data
    real_ids=[a.asset.joint_names.index(n) for n in REAL_JOINT_NAMES]
    lower_names=list(REAL_JOINT_NAMES[:12]);lower_idx=[c.tracking_joint_names.index(n) for n in lower_names]
    source_offsets=np.array([[.18,-.025,0],[.18,.025,0],[0,0,.35]],dtype=np.float32)
    shift=np.array(contract["point_offsets"])-source_offsets
    def frame():
        p,q=c.chip_reference();p=p[0].cpu().numpy();q=q[0].cpu().numpy();k=c.obs_current_step_index
        return ChipReferenceFrame(c.ref_joint_pos_future_[0,k,lower_idx].cpu().numpy(),
                                  c.ref_joint_vel_future_[0,k,lower_idx].cpu().numpy(),
                                  c.ref_anchor_quat_future_w[0,k].cpu().numpy(),
                                  p-Rotation.from_quat(q[:,[1,2,3,0]]).apply(shift),q)
    rollout=policy.get_rollout_policy("eval")
    carry=env.reset();checks=[]
    with torch.inference_mode(),VecNorm.freeze(),set_exploration_type(ExplorationType.DETERMINISTIC):
        carry=rollout(carry);previous=carry["action"][0].cpu().numpy().copy()
        _,carry=env.step_and_maybe_reset(carry)
        runtime=MimicLiteChipPolicy(onnx_model_path=repo/"models/mimic_lite_chip/policy.onnx",
                                   initial_reference=frame(),reference_lower_joint_names=lower_names,
                                   reference_point_offsets=source_offsets,
                                   compliance=(c.chip.compliance[0]*c.chip_compliance_scale).cpu().numpy(),
                                   heading_alignment="none")
        for i in range(250):
            runtime.set_reference(frame());runtime.last_action=previous.copy()
            runtime.compute_target(q_real=data.joint_pos[0,real_ids].cpu().numpy(),
                                   dq_real=data.joint_vel[0,real_ids].cpu().numpy(),
                                   imu_quat_wxyz=data.body_link_quat_w[0,c.anchor_body_idx_asset].cpu().numpy(),
                                   omega=data.root_com_ang_vel_b[0].cpu().numpy())
            obs_error={k:float(np.max(np.abs(runtime.policy_input[k]-carry[k].cpu().numpy()))) for k in ("policy","command")}
            carry=rollout(carry);previous=carry["action"][0].cpu().numpy().copy()
            if i>=10:
                checks.append({**obs_error,"action":float(np.max(np.abs(previous-runtime.last_action)))})
            td,carry=env.step_and_maybe_reset(carry)
            if td["next","done"].any():raise RuntimeError("Unexpected reset during parity check")
    result={k:max(v[k] for v in checks) for k in checks[0]}
    print('PARITY '+json.dumps(result),flush=True)
    out=repo/"models/mimic_lite_chip"/cfg.get("parity_output","parity.json");out.write_text(json.dumps(result,indent=2))
    assert result['policy']<2e-5 and result['command']<2e-5 and result['action']<2e-4,result
    env.close()


if __name__=="__main__":main()
