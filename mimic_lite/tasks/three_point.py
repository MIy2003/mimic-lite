"""Pelvis + hardware-offset hands, adapted from Goal–Body's three-link task.

Pure tracking: no external-force curriculum or compliance target shift.
"""
import math

import torch

from active_adaptation.envs.mdp.observations.base import Observation
from active_adaptation.envs.mdp.rewards.base import Reward
from active_adaptation.utils.math import (
    axis_angle_from_quat, matrix_from_quat, quat_conjugate, quat_mul,
    quat_rotate, quat_rotate_inverse, yaw_quat,
)
from .command import RobotTracking
from .observations.common import random_noise


POINT_NAMES = ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link")
POINT_OFFSETS = ((0., 0., 0.), (.18, -.025, 0.), (.0719, -.003, 0.))
GOAL_STEPS = (-8, -4, -2, 0, 1, 2, 3, 4, 8, 16, 32)


def offset_points(pos, quat, offsets):
    """Link-frame offsets, never COM-relative; supports both [B,P] and [B,T,P]."""
    return pos + quat_rotate(quat, offsets.expand_as(pos))


def pack_goals(ref_pos, ref_quat, actual_pos, actual_quat, root_pos, root_quat,
               position_noise_std=0., orientation_noise_std=0.):
    if ref_pos.shape[1:] != (11, 3, 3) or ref_quat.shape[1:] != (11, 3, 4):
        raise ValueError("Expected [B,11,3,3] positions and [B,11,3,4] quaternions")
    heading = yaw_quat(root_quat)
    origin = root_pos.clone()
    origin[:, 2] = 0
    target_pos = quat_rotate_inverse(heading[:, None, None], ref_pos - origin[:, None, None])
    target_pos = random_noise(target_pos, position_noise_std)
    target_quat = quat_mul(quat_conjugate(heading[:, None, None]).expand_as(ref_quat), ref_quat)
    if orientation_noise_std > 0:
        axis_angle = torch.randn_like(target_quat[..., 1:]).clamp(-3, 3) * orientation_noise_std
        angle = axis_angle.norm(dim=-1, keepdim=True)
        delta = torch.cat((torch.cos(angle / 2), axis_angle * .5 * torch.sinc(angle / (2 * math.pi))), -1)
        target_quat = quat_mul(delta, target_quat)
    rotation6d = matrix_from_quat(target_quat)[..., :2, :].flatten(-2)
    targets = torch.cat((target_pos, rotation6d), -1).transpose(1, 2).flatten(-2)
    current = GOAL_STEPS.index(0)
    position_error = quat_rotate_inverse(heading[:, None], ref_pos[:, current] - actual_pos)
    inv_heading = quat_conjugate(heading[:, None]).expand_as(actual_quat)
    q_ref = quat_mul(inv_heading, ref_quat[:, current])
    q_actual = quat_mul(inv_heading, actual_quat)
    orientation_error = axis_angle_from_quat(quat_mul(q_ref, quat_conjugate(q_actual)))
    return torch.cat((targets, random_noise(position_error, position_noise_std),
                      random_noise(orientation_error, orientation_noise_std)), -1)


class ThreePointTracking(RobotTracking, namespace="mimic_lite"):
    def __init__(self, point_offsets=POINT_OFFSETS, **kwargs):
        if tuple(kwargs.get("future_steps", ())) != GOAL_STEPS:
            raise ValueError(f"ThreePointTracking requires future_steps={GOAL_STEPS}")
        if any(kwargs.get(key, "pelvis") != "pelvis" for key in ("root_body_name", "anchor_body_name")):
            raise ValueError("Three-point root and heading anchor must be pelvis")
        super().__init__(**kwargs)
        offsets = torch.as_tensor(point_offsets, dtype=torch.float32)
        if offsets.shape != (3, 3) or not torch.isfinite(offsets).all() or offsets[0].any():
            raise ValueError("Offsets must be finite [pelvis,left,right] xyz; pelvis offset is zero")
        self._point_offsets = offsets

    def _initialize(self, env):
        super()._initialize(env)
        self.point_names = list(POINT_NAMES)
        self.point_offsets = self._point_offsets.to(self.device)
        self.point_ids = [self.asset.body_names.index(n) for n in POINT_NAMES]
        self.point_tracking_ids = [self.tracking_body_names.index(n) for n in POINT_NAMES]
        # Fixed three-point baseline: all goals active, every joint controllable.
        self.link_mask = torch.ones(self.num_envs, 3, device=self.device)

    def reset(self, env_ids, reset_td=None):
        # Source bundle refreshes observations after reset. Current framework's
        # opt-in reset-observation path computes them after all history resets.
        super().reset(env_ids, reset_td)
        if self.env.backend == "mjlab":
            self.env.sim.forward()
        self._read_current_robot_state()
        self._refresh_future_buffers()

    def actual_points(self):
        data = self.asset.data
        quat = data.body_link_quat_w[:, self.point_ids]
        pos = offset_points(data.body_link_pos_w[:, self.point_ids], quat, self.point_offsets)
        return pos, quat

    def reference_points(self, reward=False):
        # The executed-step buffers must not be replaced by the next command.
        pos = self.ref_body_pos_w if reward else self.ref_body_pos_future_w
        quat = self.ref_body_quat_w if reward else self.ref_body_quat_future_w
        pos, quat = pos[..., self.point_tracking_ids, :], quat[..., self.point_tracking_ids, :]
        return offset_points(pos, quat, self.point_offsets), quat


class three_point_targets(Observation, namespace="mimic_lite"):
    def __init__(self, position_noise_std=.005, orientation_noise_std=.05):
        super().__init__()
        if any(not math.isfinite(s) or s < 0 for s in (position_noise_std, orientation_noise_std)):
            raise ValueError("Goal noise must be finite and nonnegative")
        self.position_noise_std, self.orientation_noise_std = position_noise_std, orientation_noise_std

    def compute(self):
        c = self.command_manager
        ref_pos, ref_quat = c.reference_points()
        actual_pos, actual_quat = c.actual_points()
        return pack_goals(ref_pos, ref_quat, actual_pos, actual_quat,
                          c.asset.data.root_link_pos_w, c.asset.data.root_link_quat_w,
                          self.position_noise_std, self.orientation_noise_std).flatten(1)


class three_point_mask(Observation, namespace="mimic_lite"):
    def compute(self):
        return self.command_manager.link_mask


class three_point_error(Reward, namespace="mimic_lite"):
    """Offset-point world error in m/rad; log only, not a new reward objective."""
    def __init__(self, point=2, orientation=False, **kwargs):
        super().__init__(**kwargs)
        if point not in (0, 1, 2):
            raise ValueError("point must be pelvis=0, left=1 or right=2")
        self.point, self.orientation = point, orientation

    def _compute(self):
        ref_pos, ref_quat = self.command_manager.reference_points(reward=True)
        pos, quat = self.command_manager.actual_points()
        if self.orientation:
            error = axis_angle_from_quat(quat_mul(ref_quat, quat_conjugate(quat))).norm(dim=-1)
        else:
            error = (ref_pos - pos).norm(dim=-1)
        return error[:, self.point:self.point+1]
