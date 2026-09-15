"""Causal mocap-style velocity estimator ported from the Goal–Body bundle."""
import torch
from active_adaptation.utils.math import quat_rotate_inverse
from mimic_lite.tasks.deferred import DeferredObservation as BaseObservation
from .common import random_noise as _clipped_normal_like

class goal_root_lin_vel_history(BaseObservation, namespace="mimic_lite"):
    """Causal root velocity estimate from a noisy, delayed mocap position stream.

    The estimator differentiates accepted position samples, accounts for the
    number of elapsed control steps after dropouts, applies a causal EMA, then
    rotates the result into the current robot body frame. It intentionally does
    not expose simulator ground-truth velocity to the actor after reset.
    """

    def _initialize_impl(
        self,
        history_steps: list[int] | None = None,
        position_noise_std: float = 0.002,
        output_noise_std: float = 0.02,
        timestamp_jitter_std: float = 0.02,
        dropout_prob: float = 0.01,
        latency_steps_range: tuple[int, int] | list[int] = (0, 2),
        ema_alpha: float = 0.35,
        output_clip: float = 4.0,
    ):
        if history_steps is None:
            history_steps = [0]
        if not history_steps or min(history_steps) < 0:
            raise ValueError("history_steps must be a non-empty list of non-negative integers")
        if len(latency_steps_range) != 2:
            raise ValueError("latency_steps_range must contain [min, max]")
        latency_min, latency_max = (int(value) for value in latency_steps_range)
        if latency_min < 0 or latency_max < latency_min:
            raise ValueError("latency_steps_range must satisfy 0 <= min <= max")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0.0 <= dropout_prob < 1.0:
            raise ValueError("dropout_prob must be in [0, 1)")

        self.asset = self.env.scene.articulations["robot"]
        self.history_steps = [int(step) for step in history_steps]
        self.history_offsets = torch.as_tensor(
            self.history_steps, dtype=torch.long, device=self.device
        )
        self.history_buffer_size = max(self.history_steps) + 1
        self.history_head = 0
        self.history_buffer = torch.zeros(
            self.num_envs, self.history_buffer_size, 3, device=self.device
        )

        self.position_noise_std = max(float(position_noise_std), 0.0)
        self.output_noise_std = max(float(output_noise_std), 0.0)
        self.timestamp_jitter_std = max(float(timestamp_jitter_std), 0.0)
        self.dropout_prob = float(dropout_prob)
        self.latency_min = latency_min
        self.latency_max = latency_max
        self.ema_alpha = float(ema_alpha)
        self.output_clip = float(output_clip)

        self.pose_buffer_size = latency_max + 1
        self.pose_head = 0
        self.pose_buffer = torch.zeros(
            self.num_envs, self.pose_buffer_size, 3, device=self.device
        )
        self.latency_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.last_accepted_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.filtered_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset(self.all_env_ids)

    def _noisy_position(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        position = self.asset.data.root_link_pos_w
        if env_ids is not None:
            position = position[env_ids]
        return _clipped_normal_like(position, self.position_noise_std)

    def _body_velocity(self, velocity_w: torch.Tensor) -> torch.Tensor:
        velocity_b = quat_rotate_inverse(
            self.asset.data.root_link_quat_w, velocity_w
        )
        velocity_b = _clipped_normal_like(velocity_b, self.output_noise_std)
        return velocity_b.clamp(-self.output_clip, self.output_clip)

    def reset(self, env_ids: torch.Tensor, reset_td=None) -> None:
        if env_ids.numel() == 0:
            return
        measured_pos = self._noisy_position(env_ids)
        self.pose_buffer[env_ids] = measured_pos.unsqueeze(1)
        self.last_accepted_pos_w[env_ids] = measured_pos
        self.elapsed_steps[env_ids] = 0
        self.latency_steps[env_ids] = torch.randint(
            self.latency_min,
            self.latency_max + 1,
            (env_ids.numel(),),
            device=self.device,
        )

        initial_vel_w = self.asset.data.root_com_lin_vel_w[env_ids]
        self.filtered_vel_w[env_ids] = initial_vel_w
        initial_vel_b = quat_rotate_inverse(
            self.asset.data.root_link_quat_w[env_ids], initial_vel_w
        )
        initial_vel_b = _clipped_normal_like(initial_vel_b, self.output_noise_std)
        initial_vel_b = initial_vel_b.clamp(-self.output_clip, self.output_clip)
        self.history_buffer[env_ids] = initial_vel_b.unsqueeze(1)

    def update(self) -> None:
        self.pose_head = (self.pose_head - 1) % self.pose_buffer_size
        self.pose_buffer[:, self.pose_head] = self._noisy_position()
        delayed_indices = (
            self.pose_head + self.latency_steps
        ) % self.pose_buffer_size
        measured_pos = self.pose_buffer[self.all_env_ids, delayed_indices]

        self.elapsed_steps.add_(1)
        accepted = torch.rand(self.num_envs, device=self.device) >= self.dropout_prob
        elapsed = self.elapsed_steps.clamp_min(1).to(torch.float32)
        jitter = (
            torch.randn(self.num_envs, device=self.device).clamp(-3.0, 3.0)
            * self.timestamp_jitter_std
        )
        measured_dt = self.env.step_dt * elapsed * (1.0 + jitter).clamp_min(0.25)
        raw_vel_w = (
            measured_pos - self.last_accepted_pos_w
        ) / measured_dt.unsqueeze(-1)
        filtered_candidate = (
            self.ema_alpha * raw_vel_w
            + (1.0 - self.ema_alpha) * self.filtered_vel_w
        )
        self.filtered_vel_w[accepted] = filtered_candidate[accepted]
        self.last_accepted_pos_w[accepted] = measured_pos[accepted]
        self.elapsed_steps[accepted] = 0

        value = self._body_velocity(self.filtered_vel_w)
        self.history_head = (self.history_head - 1) % self.history_buffer_size
        self.history_buffer[:, self.history_head] = value

    def compute(self) -> torch.Tensor:
        indices = (
            self.history_offsets + self.history_head
        ) % self.history_buffer_size
        return self.history_buffer[:, indices].reshape(self.num_envs, -1)
