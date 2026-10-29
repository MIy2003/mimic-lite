"""Five pose targets: left hand, right hand, torso, left foot, right foot.

Only the existing upper-body points receive CHIP virtual displacements and
external wrenches. Foot poses stay nominal in the pelvis reference frame.
"""
import torch

from active_adaptation.envs.mdp.observations.base import Observation
from active_adaptation.envs.mdp.rewards.base import Reward
from active_adaptation.utils.math import (
    matrix_from_quat, quat_conjugate, quat_mul,
    quat_rotate as quat_apply, quat_rotate_inverse as quat_apply_inverse,
)
from .chip import ChipTracking, chip_privileged


class ChipFivePointTracking(ChipTracking, namespace="mimic_lite"):
    def __init__(self, foot_point_offsets=((0., 0., 0.), (0., 0., 0.)), **kwargs):
        super().__init__(**kwargs)
        self._foot_point_offsets = foot_point_offsets

    def _initialize(self, env):
        super()._initialize(env)
        if self.chip.compliance_mode != "wrist_6d":
            raise ValueError("Five-point CHIP requires wrist_6d hand compliance")
        self.foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.foot_body_ids = [self.asset.body_names.index(n) for n in self.foot_body_names]
        self.foot_tracking_ids = [self.tracking_body_names.index(n) for n in self.foot_body_names]
        self.foot_offsets = torch.as_tensor(self._foot_point_offsets, device=self.device, dtype=torch.float32)
        if self.foot_offsets.shape != (2, 3) or not torch.isfinite(self.foot_offsets).all():
            raise ValueError("foot_point_offsets must be two finite xyz link-frame offsets")

    def foot_reference(self, reward=False):
        if reward:
            pos, quat = self.ref_body_pos_w, self.ref_body_quat_w
            anchor_pos, anchor_quat = self.ref_anchor_pos_w, self.ref_anchor_quat_w
        else:
            k = self.obs_current_step_index
            pos, quat = self.ref_body_pos_future_w[:, k], self.ref_body_quat_future_w[:, k]
            anchor_pos, anchor_quat = self.ref_anchor_pos_future_w[:, k], self.ref_anchor_quat_future_w[:, k]
        q = quat[:, self.foot_tracking_ids]
        p = pos[:, self.foot_tracking_ids] + quat_apply(q, self.foot_offsets.expand(self.num_envs, -1, -1))
        aq = anchor_quat[:, None].expand(-1, 2, -1)
        return quat_apply_inverse(aq, p - anchor_pos[:, None]), quat_mul(quat_conjugate(aq), q)

    def foot_actual(self):
        data = self.asset.data
        q = data.body_link_quat_w[:, self.foot_body_ids]
        p = data.body_link_pos_w[:, self.foot_body_ids] + quat_apply(q, self.foot_offsets.expand(self.num_envs, -1, -1))
        aq = data.body_link_quat_w[:, self.anchor_body_idx_asset, None].expand(-1, 2, -1)
        ap = data.body_link_pos_w[:, self.anchor_body_idx_asset, None]
        return quat_apply_inverse(aq, p - ap), quat_mul(quat_conjugate(aq), q)


class chip_five_point_command(Observation, namespace="mimic_lite"):
    """59D: positions 15, wxyz 20, anchor orientation 6, upper compliance 18.

Feet have no compliance channels. Lower-body reference q/dq are not read.
The measured joints remain in the unchanged 930D proprioceptive history.
"""
    def compute(self):
        c = self.command_manager
        hand_p, hand_q = c.chip_virtual_reference()
        foot_p, foot_q = c.foot_reference()
        aq = c.asset.data.body_link_quat_w[:, c.anchor_body_idx_asset]
        anchor = quat_mul(quat_conjugate(aq), c.ref_anchor_quat_future_w[:, c.obs_current_step_index])
        return torch.cat((torch.cat((hand_p, foot_p), 1).flatten(1),
                          torch.cat((hand_q, foot_q), 1).flatten(1),
                          matrix_from_quat(anchor)[..., :2].flatten(1),
                          c.chip_compliance_observation()), -1)


class chip_five_point_privileged(chip_privileged, namespace="mimic_lite"):
    def compute(self):
        p, q = self.command_manager.foot_reference()
        return torch.cat((super().compute(), p.flatten(1), q.flatten(1)), -1)


class chip_feet_tracking_reward(Reward, namespace="mimic_lite"):
    def __init__(self, sigma=0.1, orientation=False, **kwargs):
        super().__init__(**kwargs)
        if sigma <= 0:
            raise ValueError("Foot tracking sigma must be positive")
        self.sigma = sigma
        self.orientation = orientation

    def _compute(self):
        ref_p, ref_q = self.command_manager.foot_reference(reward=True)
        p, q = self.command_manager.foot_actual()
        if self.orientation:
            delta = quat_mul(quat_conjugate(ref_q), q)
            error = (2 * torch.atan2(delta[..., 1:].norm(dim=-1), delta[..., 0].abs())).square()
        else:
            error = (p - ref_p).square().sum(-1)
        return torch.exp(-error.mean(-1, keepdim=True) / self.sigma**2)
