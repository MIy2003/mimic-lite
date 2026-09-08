"""Simulator-independent CHIP sampling and geometry. SI units throughout."""
import torch


def virtual_target(reference_local, force_world, compliance, robot_anchor_quat):
    """BM convention: subtract cF in the robot anchor frame; never clip displacement."""
    q = robot_anchor_quat[:, None, :]
    v = force_world * compliance[..., None]
    xyz = -q[..., 1:].expand_as(v)
    t = 2 * torch.cross(xyz, v, dim=-1)
    local_displacement = v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)
    return reference_local - local_displacement


def point_wrench(force_world, point_world, com_world):
    return force_world, torch.cross(point_world - com_world, force_world, dim=-1)


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
                 warmup_steps=500, ramp_steps=5000):
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
        self.device = device
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
        self.age[ids] = 0

    def reset(self, ids):
        self._sample_compliance(ids)
        self._sample_force(ids)
        self.force[ids] = 0

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
        self.age += 1
        self.steps += 1
