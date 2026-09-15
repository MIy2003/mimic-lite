"""CHIP three-point compliance task for MimicLite's MJLab backend."""
import torch

from active_adaptation.envs.mdp.observations.base import Observation
from active_adaptation.envs.mdp.rewards.base import Reward
from active_adaptation.utils.math import (
    quat_rotate as quat_apply, quat_rotate_inverse as quat_apply_inverse,
    quat_conjugate, quat_mul, matrix_from_quat,
)
from .command import RobotTracking
from .chip_math import ChipSchedule, point_wrench, project_force, virtual_target, weighted_tracking


class ChipTracking(RobotTracking, namespace="mimic_lite"):
    def __init__(self, chip=None, **kwargs):
        super().__init__(**kwargs)
        self._chip_cfg = dict(chip or {})

    def _initialize(self, env):
        if env.backend != "mjlab":
            raise ValueError("CHIP currently implements the MJLab external-wrench path only")
        super()._initialize(env)
        cfg = self._chip_cfg.copy()
        self.chip_body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
        self.chip_body_ids = [self.asset.body_names.index(n) for n in self.chip_body_names]
        self.chip_tracking_ids = [self.tracking_body_names.index(n) for n in self.chip_body_names]
        self.chip_offsets = torch.tensor(cfg.pop("point_offsets", [[0.18, -0.025, 0],
                                                [0.0719, -0.003, 0], [0, 0, 0.35]]), device=self.device)
        if self.chip_offsets.shape != (3, 3):
            raise ValueError("point_offsets must be [left wrist, right wrist, torso] x xyz")
        self.chip_compliance_scale = float(cfg.pop("compliance_scale", 10.0))
        if self.chip_compliance_scale <= 0:
            raise ValueError("compliance_scale must be positive")
        self.chip = ChipSchedule(self.num_envs, self.device, **cfg)
        self.chip_applied_force = torch.zeros_like(self.chip.force)
        self.chip_applied_axis = torch.zeros_like(self.chip.force)
        self.chip_leg_ids = [i for i, n in enumerate(self.tracking_joint_names)
                             if any(s in n for s in ("hip_", "knee_", "ankle_"))]
        if len(self.chip_leg_ids) != 12:
            raise ValueError("CHIP requires 12 lower-body tracking joints")

    def reset(self, env_ids, reset_td=None):
        super().reset(env_ids, reset_td)
        self.chip.reset(env_ids)
        self.chip_applied_force[env_ids] = 0
        self.chip_applied_axis[env_ids] = 0
        self.asset.write_external_wrench_to_sim(
            self.chip.force[env_ids], torch.zeros_like(self.chip.force[env_ids]),
            env_ids=env_ids, body_ids=self.chip_body_ids)

    def step(self):
        super().step()
        self.chip.prepare()

    def pre_step(self, substep):
        super().pre_step(substep)
        data = self.asset.data
        q = data.body_link_quat_w[:, self.chip_body_ids]
        point = data.body_link_pos_w[:, self.chip_body_ids] + quat_apply(q, self.chip_offsets.expand(self.num_envs, -1, -1))
        com = data.body_com_pos_w[:, self.chip_body_ids]
        force, torque = point_wrench(self.chip.force, point, com)
        self.asset.write_external_wrench_to_sim(force, torque, body_ids=self.chip_body_ids)
        self.chip_applied_force.copy_(force)
        self.chip_applied_axis.copy_(self.chip_axis_world())

    def chip_axis_world(self):
        """Shared local xyz rotated by each actual link, not reference/COM frames.

        The third point uses torso_link's frame; its default compliance is zero.
        """
        q = self.asset.data.body_link_quat_w[:, self.chip_body_ids]
        return quat_apply(q, self.chip.stiffness_axis[:, None].expand(-1, 3, -1))

    def chip_reference(self, reward=False):
        k = self.reward_current_step_index if reward else self.obs_current_step_index
        # Reward buffers must survive command.step(), which advances future references.
        if reward:
            pos, quat = self.ref_body_pos_w, self.ref_body_quat_w
            anchor_pos, anchor_quat = self.ref_anchor_pos_w, self.ref_anchor_quat_w
        else:
            pos, quat = self.ref_body_pos_future_w[:, k], self.ref_body_quat_future_w[:, k]
            anchor_pos = self.ref_anchor_pos_future_w[:, k]
            anchor_quat = self.ref_anchor_quat_future_w[:, k]
        q = quat[:, self.chip_tracking_ids]
        p = pos[:, self.chip_tracking_ids] + quat_apply(q, self.chip_offsets.expand(self.num_envs, -1, -1))
        aq = anchor_quat[:, None].expand(-1, 3, -1)
        return (quat_apply_inverse(aq, p - anchor_pos[:, None]),
                quat_mul(quat_conjugate(aq), q))

    def chip_actual(self):
        data = self.asset.data
        q = data.body_link_quat_w[:, self.chip_body_ids]
        p = data.body_link_pos_w[:, self.chip_body_ids] + quat_apply(q, self.chip_offsets.expand(self.num_envs, -1, -1))
        aq = data.body_link_quat_w[:, self.anchor_body_idx_asset, None].expand(-1, 3, -1)
        ap = data.body_link_pos_w[:, self.anchor_body_idx_asset, None]
        return quat_apply_inverse(aq, p - ap), quat_mul(quat_conjugate(aq), q)


class chip_command(Observation, namespace="mimic_lite"):
    """54 legacy values; wrist_axis appends 3 unscaled, shared local-axis values."""
    def compute(self):
        c = self.command_manager
        pos, orn = c.chip_reference()
        aq = c.asset.data.body_link_quat_w[:, c.anchor_body_idx_asset]
        directional = c.chip.compliance_mode == "wrist_axis"
        pos = virtual_target(pos, c.chip.force, c.chip.compliance, aq,
                             c.chip_axis_world() if directional else None)
        k = c.obs_current_step_index
        anchor = quat_mul(quat_conjugate(aq), c.ref_anchor_quat_future_w[:, k])
        values = (c.ref_joint_pos_future_[:, k, c.chip_leg_ids],
                  c.ref_joint_vel_future_[:, k, c.chip_leg_ids],
                  pos.flatten(1), orn.flatten(1),
                  matrix_from_quat(anchor)[..., :2].flatten(1),
                  c.chip.compliance * c.chip_compliance_scale)
        if directional:
            values += (c.chip.stiffness_axis,)
        return torch.cat(values, dim=-1)


class chip_privileged(Observation, namespace="mimic_lite"):
    def compute(self):
        c = self.command_manager
        pos, orn = c.chip_reference()
        values = (pos.flatten(1), orn.flatten(1), c.chip.force.flatten(1),
                  c.chip.compliance * c.chip_compliance_scale)
        if c.chip.compliance_mode == "wrist_axis":
            values += (c.chip.stiffness_axis,)
        return torch.cat(values, dim=-1)


class chip_history(Observation, namespace="mimic_lite"):
    """Term-major, oldest-to-newest 10-frame history (930 values for G1)."""
    def __init__(self, history_length=10, noise=True):
        super().__init__()
        self.length = history_length
        self.noise = noise
        if history_length < 1:
            raise ValueError("history_length must be positive")

    def _initialize(self, env):
        super()._initialize(env)
        self.asset = env.scene.articulations["robot"]
        self.action = env.action_manager
        self.buffers = [torch.zeros(self.num_envs, self.length, n, device=self.device)
                        for n in (3, 3, self.action.action_dim, self.action.action_dim, self.action.action_dim)]

    def values(self):
        d = self.asset.data
        ids = self.action.joint_ids
        values = [d.projected_gravity_b, d.root_com_ang_vel_b,
                  d.joint_pos[:, ids] - self.action.default_joint_pos[:, ids],
                  d.joint_vel[:, ids] - d.default_joint_vel[:, ids], self.action.action_buf[:, 0]]
        scales = [0.01, 0.2, 0.01, 0.5, 0.0]
        return [x + torch.empty_like(x).uniform_(-s, s) if self.noise and s else x
                for x, s in zip(values, scales)]

    def reset(self, env_ids, reset_td=None):
        for b, x in zip(self.buffers, self.values()):
            b[env_ids] = x[env_ids, None]

    def update(self):
        for b, x in zip(self.buffers, self.values()):
            b[:, :-1] = b[:, 1:].clone()
            b[:, -1] = x

    def compute(self):
        return torch.cat([b.flatten(1) for b in self.buffers], dim=-1)


class chip_tracking_reward(Reward, namespace="mimic_lite"):
    def __init__(self, sigma=0.1, orientation=False, point_weights=(2, 4, 1), **kwargs):
        super().__init__(**kwargs)
        if sigma <= 0 or len(point_weights) != 3 or min(point_weights) < 0 or sum(point_weights) <= 0:
            raise ValueError("Invalid CHIP tracking reward parameters")
        self.sigma = sigma
        self.orientation = orientation
        self._point_weights = tuple(point_weights)

    def _initialize(self, env):
        super()._initialize(env)
        self.weights = torch.tensor(self._point_weights, device=self.device)

    def _compute(self):
        ref_p, ref_q = self.command_manager.chip_reference(reward=True)
        p, q = self.command_manager.chip_actual()
        if self.orientation:
            delta = quat_mul(quat_conjugate(ref_q), q)
            error = (2 * torch.atan2(delta[..., 1:].norm(dim=-1), delta[..., 0].abs())).square()
        else:
            error = (p - ref_p).square().sum(-1)
        return weighted_tracking(error, self.weights, self.sigma)


class chip_metric(Reward, namespace="mimic_lite"):
    def __init__(self, quantity="right_wrist_error", **kwargs):
        super().__init__(**kwargs)
        if quantity not in ("right_wrist_error", "right_wrist_force", "right_wrist_compliance",
                            "right_wrist_force_parallel", "right_wrist_force_perpendicular",
                            "right_wrist_error_parallel", "right_wrist_error_perpendicular"):
            raise ValueError(quantity)
        self.quantity = quantity

    def _compute(self):
        c = self.command_manager
        if self.quantity == "right_wrist_force":
            return c.chip_applied_force[:, 1].norm(dim=-1, keepdim=True)
        if self.quantity == "right_wrist_compliance":
            return c.chip.compliance[:, 1:2]
        if self.quantity.startswith("right_wrist_force_"):
            force = c.chip_applied_force[:, 1]
            parallel = project_force(force, c.chip_applied_axis[:, 1])
            value = parallel if self.quantity == "right_wrist_force_parallel" else force - parallel
            return value.norm(dim=-1, keepdim=True)
        ref_p, _ = c.chip_reference(reward=True)
        p, _ = c.chip_actual()
        error = p[:, 1] - ref_p[:, 1]
        if self.quantity != "right_wrist_error":
            aq = c.asset.data.body_link_quat_w[:, c.anchor_body_idx_asset]
            axis_local = quat_apply_inverse(aq, c.chip_applied_axis[:, 1])
            parallel = project_force(error, axis_local)
            error = parallel if self.quantity == "right_wrist_error_parallel" else error - parallel
        return error.norm(dim=-1, keepdim=True)
