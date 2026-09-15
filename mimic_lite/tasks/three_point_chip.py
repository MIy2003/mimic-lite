"""Isotropic CHIP on the Goal–Body controller; rewards stay nominal/full-body."""
import math
import torch

from active_adaptation.envs.mdp.observations.base import Observation
from active_adaptation.envs.mdp.rewards.base import Reward
from .chip_math import ChipSchedule, point_wrench
from .three_point import ThreePointTracking, offset_points, pack_goals, three_point_targets, three_point_error


class ThreePointChipTracking(ThreePointTracking, namespace="mimic_lite"):
    def __init__(self, chip=None, **kwargs):
        super().__init__(**kwargs)
        self._chip_cfg = dict(chip or {})
        if self._chip_cfg.get("compliance_mode", "isotropic") != "isotropic":
            raise ValueError("ThreePointChipTracking supports isotropic CHIP only, without an axis")
        if self._chip_cfg.get("compliance_max", [.02, .02, 0])[2] != 0:
            raise ValueError("Torso compliance must be zero: the third control goal is pelvis")

    def _initialize(self, env):
        if env.backend != "mjlab":
            raise ValueError("CHIP external wrench implementation requires MJLab")
        super()._initialize(env)
        cfg = self._chip_cfg.copy()
        self.chip_compliance_scale = float(cfg.pop("compliance_scale", 10.))
        if not math.isfinite(self.chip_compliance_scale) or self.chip_compliance_scale <= 0:
            raise ValueError("compliance_scale must be finite and positive")
        self.chip_offsets = torch.tensor(cfg.pop("point_offsets", [[.18,-.025,0], [.0719,-.003,0], [0,0,.35]]), device=self.device)
        if self.chip_offsets.shape != (3,3) or not torch.isfinite(self.chip_offsets).all():
            raise ValueError("CHIP force offsets must be finite [left,right,torso] xyz")
        if not torch.allclose(self.chip_offsets[:2], self.point_offsets[1:]):
            raise ValueError("Hand force points and control points must use the same hardware offsets")
        self.chip_body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
        self.chip_body_ids = [self.asset.body_names.index(n) for n in self.chip_body_names]
        self.chip = ChipSchedule(self.num_envs, self.device, **cfg)
        self.chip_applied_force = torch.zeros_like(self.chip.force)

    def reset(self, env_ids, reset_td=None):
        super().reset(env_ids, reset_td)
        self.chip.reset(env_ids)
        self.chip_applied_force[env_ids] = 0
        self.asset.write_external_wrench_to_sim(self.chip.force[env_ids],
            torch.zeros_like(self.chip.force[env_ids]), env_ids=env_ids, body_ids=self.chip_body_ids)

    def step(self):
        super().step()
        self.chip.prepare()

    def pre_step(self, substep):
        super().pre_step(substep)
        d = self.asset.data
        q = d.body_link_quat_w[:, self.chip_body_ids]
        p = offset_points(d.body_link_pos_w[:, self.chip_body_ids], q, self.chip_offsets)
        force, torque = point_wrench(self.chip.force, p, d.body_com_pos_w[:, self.chip_body_ids])
        self.asset.write_external_wrench_to_sim(force, torque, body_ids=self.chip_body_ids)
        self.chip_applied_force.copy_(force)

    def goal_compliance(self):
        # Schedule is L/R/torso; actor goal order is pelvis/L/R.
        return torch.cat((torch.zeros_like(self.chip.compliance[:, :1]), self.chip.compliance[:, :2]), -1)

    def virtual_reference_points(self):
        pos, quat = self.reference_points()
        displacement = torch.zeros_like(self.chip.force)
        displacement[:, 1:] = self.chip.force[:, :2] * self.chip.compliance[:, :2, None]
        # Same current-force shift for all 11 knots (including history), with no
        # future-force oracle and no clipping. Never mutate nominal reward refs.
        return pos - displacement[:, None], quat


class three_point_chip_targets(three_point_targets, namespace="mimic_lite"):
    def compute(self):
        c = self.command_manager
        ref_pos, ref_quat = c.virtual_reference_points()
        actual_pos, actual_quat = c.actual_points()
        # Current error must use the SAME virtual target, not leak nominal error.
        goals = pack_goals(ref_pos, ref_quat, actual_pos, actual_quat,
            c.asset.data.root_link_pos_w, c.asset.data.root_link_quat_w,
            self.position_noise_std, self.orientation_noise_std).flatten(1)
        return torch.cat((goals, c.goal_compliance() * c.chip_compliance_scale), -1)


class three_point_chip_privileged(Observation, namespace="mimic_lite"):
    def compute(self):
        c = self.command_manager
        return torch.cat((c.chip.force.flatten(1), c.goal_compliance() * c.chip_compliance_scale), -1)


class three_point_precision_reward(three_point_error, namespace="mimic_lite"):
    """Additional offset-point reward against the nominal executed reference.

    World-frame error, matching three_point_error. Never reward against the
    actor's force-shifted command; that would change CHIP's learning objective.
    """
    def __init__(self, sigma=.03, **kwargs):
        super().__init__(**kwargs)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        self.sigma = sigma

    def _compute(self):
        return torch.exp(-super()._compute().square() / self.sigma**2)


class three_point_chip_metric(Reward, namespace="mimic_lite"):
    def __init__(self, body=1, quantity="force", **kwargs):
        super().__init__(**kwargs)
        if body not in (0,1,2) or quantity not in ("force", "compliance"):
            raise ValueError("Expected L/R/torso body index and force/compliance quantity")
        self.body, self.quantity = body, quantity

    def _compute(self):
        c = self.command_manager
        if self.quantity == "compliance":
            return c.chip.compliance[:, self.body:self.body+1]
        return c.chip_applied_force[:, self.body].norm(dim=-1, keepdim=True)
