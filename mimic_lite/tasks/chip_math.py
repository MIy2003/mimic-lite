"""Simulator-independent CHIP sampling and geometry. SI units throughout."""
import math

import torch


def project_force(force, unit_axis):
    """Signed projection onto a unit axis in the SAME frame as force."""
    return (force * unit_axis).sum(dim=-1, keepdim=True) * unit_axis


def virtual_target(reference_local, force_world, compliance, robot_anchor_quat,
                   axis_world=None):
    """Subtract cF (or cF_parallel) in the anchor frame, without clipping.

    Projection changes the actor's virtual target only, never the physical force.
    Omitting axis_world preserves the original isotropic checkpoint convention.
    """
    q = robot_anchor_quat[:, None, :]
    if axis_world is not None:
        force_world = project_force(force_world, axis_world)
    v = force_world * compliance[..., None]
    xyz = -q[..., 1:].expand_as(v)
    t = 2 * torch.cross(xyz, v, dim=-1)
    local_displacement = v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)
    return reference_local - local_displacement


def point_wrench(force_world, point_world, com_world, couple_world=None):
    torque = torch.cross(point_world - com_world, force_world, dim=-1)
    if couple_world is not None:
        torque = torque + couple_world
    return force_world, torque


def wrist_compliance_shift(wrench_world, compliance, wrist_quat):
    """R diag(c) R^T w, using the actual wrist frame (wxyz quaternion)."""
    xyz = -wrist_quat[..., 1:]
    t = 2 * torch.cross(xyz, wrench_world, dim=-1)
    local = wrench_world + wrist_quat[..., :1] * t + torch.cross(xyz, t, dim=-1)
    local = local * compliance
    xyz = wrist_quat[..., 1:]
    t = 2 * torch.cross(xyz, local, dim=-1)
    return local + wrist_quat[..., :1] * t + torch.cross(xyz, t, dim=-1)


def rotation_vector_quat(vector):
    """Stable Exp on SO(3), returning a unit wxyz quaternion, including at zero."""
    angle = vector.norm(dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle / 2),
                      .5 * torch.sinc(angle / (2 * math.pi)) * vector), -1)


def weighted_tracking(error_squared, weights, sigma):
    return torch.exp(-(error_squared * weights).sum(-1, keepdim=True) / weights.sum() / sigma**2)


class ChipSchedule:
    """Per-environment force cycles and independent compliance resampling.

    prepare() runs once per control step, before the observation describing the
    next action. The force is held constant over all physics substeps.
    """
    def __init__(self, num_envs, device, compliance_max=(0.02, 0.02, 0.0),
                 compliance_duration=(100, 200), zero_probability=0.25,
                 max_probability=0.25, sampling_power=2.0, max_force=20.0,
                 force_duration=(50, 100), force_interval=100,
                 body_probabilities=(0.15, 0.75, 0.10), multi_body_probability=0.10,
                 warmup_steps=500, ramp_steps=5000, compliance_mode="isotropic",
                 axis_probabilities=(0.20, 0.60, 0.15, 0.05),
                 axis_cone_degrees=(30.0, 60.0, 90.0), fixed_stiffness_axis=None,
                 compliance_6d_max=(.02, .02, .02, .1, .1, .1),
                 max_torque=5., wrench_event_probabilities=(.4, .4, .2),
                 rigid_probability=0.25, translation_pattern_probabilities=None,
                 structured_max_probability=0.5, torque_axis_probability=0.,
                 left_compliance_mode="full", rotation_pattern_probabilities=None,
                 torque_amplitude_bands=None, angular_hindsight_limit=None):
        if angular_hindsight_limit is not None and (
                not math.isfinite(angular_hindsight_limit) or angular_hindsight_limit <= 0):
            raise ValueError("angular_hindsight_limit must be positive radians")
        if torque_amplitude_bands is not None:
            if (not torque_amplitude_bands or any(len(b) != 3 for b in torque_amplitude_bands)
                    or any(not all(math.isfinite(x) for x in b) or not 0 <= b[0] < b[1] <= max_torque
                           or b[2] <= 0 for b in torque_amplitude_bands)
                    or not math.isclose(sum(b[2] for b in torque_amplitude_bands), 1., abs_tol=1e-6)):
                raise ValueError("Torque bands require [low, high, probability] within max_torque")
            if angular_hindsight_limit is not None and min(b[0] for b in torque_amplitude_bands) * max(compliance_6d_max[3:]) >= angular_hindsight_limit:
                raise ValueError("Torque bands must allow amplitudes below the angular limit")
        if left_compliance_mode not in ("full", "rigid"):
            raise ValueError("left_compliance_mode must be full or rigid")
        if left_compliance_mode != "full" and compliance_mode != "wrist_6d":
            raise ValueError("Left-hand simplification requires wrist_6d mode")
        self.left_compliance_mode = left_compliance_mode
        for name, probabilities in (("Translation", translation_pattern_probabilities),
                                    ("Rotation", rotation_pattern_probabilities)):
            if probabilities is not None and (
                    len(probabilities) != 6 or
                    any(not math.isfinite(x) or x < 0 for x in probabilities) or
                    not math.isclose(sum(probabilities), 1., abs_tol=1e-6)):
                raise ValueError(f"{name} probabilities: x/y/z/xyz/zero/independent must sum to one")
        for value in (structured_max_probability, torque_axis_probability):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Structured max and torque axis probabilities must be in [0,1]")
        if len(compliance_max) != 3 or any(x < 0 for x in compliance_max):
            raise ValueError("compliance_max must contain three nonnegative SI values")
        if not (0 <= zero_probability <= 1 and 0 <= max_probability <= 1
                and zero_probability + max_probability <= 1):
            raise ValueError("Invalid compliance endpoint probabilities")
        for bounds in (compliance_duration, force_duration):
            if len(bounds) != 2 or not 0 < bounds[0] < bounds[1]:
                raise ValueError("Durations require 0 < low < high (exclusive)")
        if (len(body_probabilities) != 3 or min(body_probabilities) < 0
                or sum(body_probabilities) <= 0 or not 0 <= multi_body_probability <= 1):
            raise ValueError("Invalid force body probabilities")
        if min(max_force, force_interval, warmup_steps, ramp_steps) < 0 or sampling_power <= 0:
            raise ValueError("Invalid force schedule parameters")
        if compliance_mode not in ("isotropic", "wrist_axis", "wrist_6d"):
            raise ValueError("Unknown compliance_mode")
        if (len(compliance_6d_max) != 6 or
                any(not math.isfinite(x) or x < 0 for x in compliance_6d_max)):
            raise ValueError("compliance_6d_max requires six finite nonnegative SI values")
        if not math.isfinite(max_torque) or max_torque < 0:
            raise ValueError("max_torque must be finite and nonnegative")
        if not math.isfinite(rigid_probability) or not 0 <= rigid_probability <= 1:
            raise ValueError("rigid_probability must be in [0,1]")
        if (len(wrench_event_probabilities) != 3 or
                any(not math.isfinite(x) or x < 0 for x in wrench_event_probabilities) or
                not math.isclose(sum(wrench_event_probabilities), 1., abs_tol=1e-6)):
            raise ValueError("force/couple/mixed probabilities must sum to one")
        if compliance_mode == "wrist_6d" and sum(body_probabilities[:2]) <= 0:
            raise ValueError("Six-dimensional compliance requires hand sampling probability")
        if (len(axis_probabilities) != 4
                or any(not math.isfinite(x) or x < 0 for x in axis_probabilities)
                or not math.isclose(sum(axis_probabilities), 1.0, abs_tol=1e-6)):
            raise ValueError("axis_probabilities must contain four probabilities summing to one")
        if (len(axis_cone_degrees) != 3
                or not 0 < axis_cone_degrees[0] < axis_cone_degrees[1] < axis_cone_degrees[2] <= 90):
            raise ValueError("axis_cone_degrees require 0 < a < b < c <= 90")
        self.device = device
        self.compliance_mode = compliance_mode
        self.axis_probabilities = torch.tensor(axis_probabilities, device=device)
        cos_bounds = [1.0] + [math.cos(math.radians(x)) for x in axis_cone_degrees]
        self.axis_cos_high = torch.tensor([1.0] + cos_bounds[:-1], device=device)
        self.axis_cos_low = torch.tensor(cos_bounds, device=device)
        self.fixed_stiffness_axis = None
        if fixed_stiffness_axis is not None:
            axis = torch.tensor(fixed_stiffness_axis, dtype=torch.float32, device=device)
            if axis.shape != (3,) or not torch.isfinite(axis).all() or axis.norm() < 1e-8:
                raise ValueError("fixed_stiffness_axis must be a finite nonzero xyz vector")
            self.fixed_stiffness_axis = axis / axis.norm()
        # One shared local vector, interpreted in each CURRENT wrist link frame.
        self.stiffness_axis = torch.zeros(num_envs, 3, device=device)
        self.stiffness_axis[:, 0] = 1
        self.c_max = torch.tensor(compliance_max, device=device)
        self.c_duration = tuple(compliance_duration)
        self.zero_probability, self.max_probability = zero_probability, max_probability
        self.sampling_power = sampling_power
        self.max_force = max_force
        self.force_duration = tuple(force_duration)
        self.interval = force_interval
        self.probabilities = torch.tensor(body_probabilities, device=device)
        self.multi_probability = multi_body_probability
        self.warmup, self.ramp = warmup_steps, ramp_steps
        self.steps = 0
        self.compliance = torch.zeros(num_envs, 3, device=device)
        # Six channels per HAND: xyz translation, xyz rotation. Pelvis is rigid.
        self.compliance_6d = torch.zeros(num_envs, 2, 6, device=device)
        self.c6_max = torch.tensor(compliance_6d_max, device=device)
        self.rigid_probability = rigid_probability
        self.translation_pattern_probabilities = (None if translation_pattern_probabilities is None
            else torch.tensor(translation_pattern_probabilities, dtype=torch.float32, device=device))
        self.rotation_pattern_probabilities = (None if rotation_pattern_probabilities is None
            else torch.tensor(rotation_pattern_probabilities, dtype=torch.float32, device=device))
        self.compliance_patterns = torch.tensor([[1,0,0], [0,1,0], [0,0,1],
            [1,1,1], [0,0,0], [0,0,0]], device=device)
        self.structured_max_probability = structured_max_probability
        self.torque_axis_probability = torque_axis_probability
        self.max_torque = max_torque
        self.torque_bands = (None if torque_amplitude_bands is None else
                            torch.tensor(torque_amplitude_bands, device=device, dtype=torch.float32))
        self.angular_hindsight_limit = angular_hindsight_limit
        self.event_probabilities = torch.tensor(wrench_event_probabilities, dtype=torch.float32, device=device)
        self.couple = torch.zeros(num_envs, 3, 3, device=device)
        self.couple_direction = torch.zeros_like(self.couple)
        self.couple_local_axis = torch.zeros_like(self.couple)
        self.couple_axis_mask = torch.zeros(num_envs, 3, dtype=torch.bool, device=device)
        self.couple_envelope = torch.zeros(num_envs, device=device)
        self.couple_amplitude = torch.zeros(num_envs, 3, device=device)
        self.couple_quantile = torch.zeros_like(self.couple_amplitude)
        self.couple_effective_amplitude = torch.zeros_like(self.couple_amplitude)
        self.couple_frame = torch.zeros(num_envs, 3, 4, device=device)
        self.couple_frame[..., 0] = 1
        self.couple_mask = torch.zeros(num_envs, 3, dtype=torch.bool, device=device)
        self.force_event = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.force = torch.zeros(num_envs, 3, 3, device=device)
        self.direction = torch.zeros_like(self.force)
        self.mask = torch.zeros(num_envs, 3, device=device, dtype=torch.bool)
        self.amplitude = torch.zeros(num_envs, device=device)
        self.age = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.duration = torch.zeros_like(self.age)
        self.c_remaining = torch.zeros_like(self.age)
        self.reset(torch.arange(num_envs, device=device))

    def _sample_compliance(self, ids):
        n = len(ids)
        if self.compliance_mode == "wrist_6d":
            u = torch.rand(n, 2, 6, device=self.device)
            c = u.pow(self.sampling_power) * self.c6_max
            choice = torch.rand_like(u)
            c = torch.where(choice < self.zero_probability, 0., c)
            c = torch.where((choice >= self.zero_probability) &
                            (choice < self.zero_probability + self.max_probability),
                            self.c6_max, c)
            # Draw translation and rotation patterns/levels independently.
            for probabilities, channels in ((self.translation_pattern_probabilities, slice(0, 3)),
                                            (self.rotation_pattern_probabilities, slice(3, 6))):
                if probabilities is None or not n:
                    continue
                pattern = torch.multinomial(probabilities, n * 2,
                    replacement=True).reshape(n, 2)
                # Share a nonzero level within this group's active axes only.
                level = torch.rand(n, 2, 1, device=self.device).pow(self.sampling_power).clamp_min(1e-6)
                level = torch.where(torch.rand_like(level) < self.structured_max_probability, 1., level)
                structured = self.compliance_patterns[pattern] * level * self.c6_max[channels]
                c[..., channels] = torch.where((pattern < 5)[..., None], structured, c[..., channels])
            if self.left_compliance_mode == "rigid":
                c[:, 0] = 0
            # Explicit BOTH-hands all-rigid examples; independent zero axes alone
            # would almost never produce a completely rigid 12-channel command.
            c[torch.rand(n, device=self.device) < self.rigid_probability] = 0
            self.compliance_6d[ids] = c
            self.c_remaining[ids] = torch.randint(*self.c_duration, (n,), device=self.device)
            return
        c = torch.rand(n, 3, device=self.device).pow(self.sampling_power) * self.c_max
        choice = torch.rand(n, device=self.device)
        c[choice < self.zero_probability] = 0
        c[(choice >= self.zero_probability) &
          (choice < self.zero_probability + self.max_probability)] = self.c_max
        self.compliance[ids] = c
        self.c_remaining[ids] = torch.randint(*self.c_duration, (n,), device=self.device)

    def _sample_force(self, ids):
        n = len(ids)
        direction = torch.randn(n, 3, 3, device=self.device)
        self.direction[ids] = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.amplitude[ids] = torch.rand(n, device=self.device) * self.max_force
        self.duration[ids] = torch.randint(*self.force_duration, (n,), device=self.device)
        mask = torch.zeros(n, 3, dtype=torch.bool, device=self.device)
        if n:
            selected = torch.multinomial(self.probabilities, n, replacement=True)
            mask[torch.arange(n, device=self.device), selected] = True
            mask[torch.rand(n, device=self.device) < self.multi_probability] = True
        self.mask[ids] = mask
        if self.compliance_mode == "wrist_6d" and n:
            event = torch.multinomial(self.event_probabilities, n, replacement=True)
            self.force_event[ids] = event != 1
            couple_mask = mask.clone()
            couple_mask[:, 2] = False
            # A torso-only force selection must not erase a requested couple event.
            missing = ~couple_mask.any(-1)
            hands = torch.multinomial(self.probabilities[:2], n, replacement=True)
            couple_mask[missing, hands[missing]] = True
            self.couple_mask[ids] = couple_mask & (event != 0)[:, None]
            direction = torch.randn(n, 3, 3, device=self.device)
            self.couple_direction[ids] = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            if self.torque_axis_probability:
                axes = torch.randint(3, (n, 3), device=self.device)
                signs = 2 * torch.randint(2, (n, 3, 1), device=self.device) - 1
                self.couple_local_axis[ids] = (torch.nn.functional.one_hot(axes, 3) * signs).to(self.couple.dtype)
                self.couple_axis_mask[ids] = torch.rand(n, 3, device=self.device) < self.torque_axis_probability
            # Inverse CDF of Beta(3,1), as in LAC's couple amplitude sampler.
            self.couple_quantile[ids] = torch.rand(n, 3, device=self.device)
            self.couple_amplitude[ids] = self._couple_amplitudes(
                self.couple_quantile[ids], torch.full((n, 3), self.max_torque, device=self.device))
        self.age[ids] = 0
        if self.compliance_mode == "wrist_axis":
            self._sample_axis(ids)

    def _sample_axis(self, ids):
        n = len(ids)
        if not n:
            return
        if self.fixed_stiffness_axis is not None:
            self.stiffness_axis[ids] = self.fixed_stiffness_axis
            return
        band = torch.multinomial(self.axis_probabilities, n, replacement=True)
        # Uniform in solid angle within each band, NOT uniform in theta.
        lo, hi = self.axis_cos_low[band], self.axis_cos_high[band]
        cos_theta = lo + (hi - lo) * torch.rand(n, device=self.device)
        sin_theta = (1 - cos_theta.square()).clamp_min(0).sqrt()
        phi = torch.rand(n, device=self.device) * (2 * math.pi)
        self.stiffness_axis[ids] = torch.stack(
            (cos_theta, sin_theta * phi.cos(), sin_theta * phi.sin()), dim=-1)

    def reset(self, ids):
        self._sample_compliance(ids)
        self._sample_force(ids)
        self.force[ids] = 0
        self.couple[ids] = 0
        self.couple_envelope[ids] = 0

    def _couple_amplitudes(self, quantile, cap):
        """Inverse CDF of the amplitude distribution conditioned on A <= cap.

        Keep each event's quantile fixed as its wrist frame/compliance changes.
        Truncation changes band proportions; it never clips the hindsight alone.
        """
        if self.torque_bands is None:
            return cap * quantile.pow(1 / 3)
        lo, hi, probability = self.torque_bands.unbind(-1)
        width = (torch.minimum(cap[..., None], hi) - lo).clamp_min(0)
        mass = probability * width / (hi - lo)
        total = mass.sum(-1, keepdim=True)
        cdf = mass.cumsum(-1)
        draw = quantile[..., None] * total
        index = (draw >= cdf).sum(-1).clamp_max(len(lo) - 1)[..., None]
        start = (cdf - mass).gather(-1, index)
        fraction = (draw - start) / mass.gather(-1, index).clamp_min(1e-12)
        value = lo[index.squeeze(-1)] + fraction.squeeze(-1) * width.gather(-1, index).squeeze(-1)
        return torch.where(total.squeeze(-1) > 0, value, torch.zeros_like(cap))

    def refresh_couple_frame(self, body_quat_world):
        """Follow current wrist axes; non-axis events retain a fixed world vector.

        Call after prepare and each physics substep. Axis selection is independent
        of compliance, so both soft and stiff directions receive disturbances.
        """
        if self.compliance_mode != "wrist_6d" or (not self.torque_axis_probability and self.angular_hindsight_limit is None):
            return
        self.couple_frame.copy_(body_quat_world)
        q = body_quat_world
        v = self.couple_local_axis
        t = 2 * torch.cross(q[..., 1:], v, dim=-1)
        world = v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)
        direction = torch.where(self.couple_axis_mask[..., None], world, self.couple_direction)
        amplitude = self.couple_amplitude
        if self.angular_hindsight_limit is not None:
            gain = wrist_compliance_shift(direction[:, :2], self.compliance_6d[..., 3:], q[:, :2]).norm(dim=-1)
            cap = torch.full_like(self.couple_amplitude, self.max_torque)
            cap[:, :2] = (self.angular_hindsight_limit / gain.clamp_min(1e-12)).clamp_max(self.max_torque)
            amplitude = self._couple_amplitudes(self.couple_quantile, cap)
        self.couple_effective_amplitude.copy_(amplitude)
        self.couple.copy_(direction * amplitude[..., None]
            * self.couple_envelope[:, None, None] * self.couple_mask[..., None])

    def prepare(self):
        self.c_remaining -= 1
        self._sample_compliance((self.c_remaining <= 0).nonzero().flatten())
        finished = self.age >= self.interval + self.duration
        self._sample_force(finished.nonzero().flatten())
        phase = ((self.age - self.interval) / self.duration).clamp(0, 1)
        envelope = torch.minimum(phase * 4, (1 - phase) * 4).clamp(0, 1)
        scale = (0.0 if self.steps < self.warmup else
                 min((self.steps - self.warmup) / self.ramp, 1.0) if self.ramp else 1.0)
        self.force.copy_(self.direction * (self.amplitude * envelope * scale)[:, None, None]
                         * self.mask[..., None])
        if self.compliance_mode == "wrist_6d":
            self.force.mul_(self.force_event[:, None, None])
            self.couple_envelope.copy_(envelope * scale)
            self.couple.copy_(self.couple_direction * self.couple_amplitude[..., None]
                              * (envelope * scale)[:, None, None] * self.couple_mask[..., None])
            if self.angular_hindsight_limit is not None:
                self.refresh_couple_frame(self.couple_frame.clone())
        self.age += 1
        self.steps += 1
