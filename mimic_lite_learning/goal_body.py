"""Goal–Body actor ported from the 2026-09-14 source bundle (see THREE_POINT.md)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class LinkCommandMask(nn.Module):
    """Zero inactive link-major 105D slots."""
    def __init__(self, num_links: int = 6, compliance: bool = False, axis: bool = False):
        super().__init__(); self.num_links = num_links
        self.compliance = compliance
        self.axis = axis
        if axis and not compliance:
            raise ValueError("Axis input requires compliance")
    def forward(self, command: torch.Tensor, link_mask: torch.Tensor) -> torch.Tensor:
        if command.shape[-1] != (105 + int(self.compliance)) * self.num_links + 3 * int(self.axis) or link_mask.shape != (*command.shape[:-1], self.num_links):
            raise ValueError(f"Expected {self.num_links} link command slots and mask.")
        slots = command[..., :105*self.num_links].reshape(*command.shape[:-1], self.num_links, 105)
        masked = torch.where(link_mask[..., None] > 0.5, slots, 0.0).flatten(-2)
        if self.compliance:
            masked = torch.cat((masked, torch.where(link_mask > .5, command[..., 105*self.num_links:106*self.num_links], 0.)), -1)
        if self.axis:
            shared = torch.where((link_mask > .5).any(-1, keepdim=True), command[..., -3:], 0.)
            masked = torch.cat((masked, shared), -1)
        return masked


class GoalBodyActor(nn.Module):
    """Six goal tokens + one joint-history token + one root-history token.

    Each pre-norm Transformer layer masks inactive goal keys/values. Both
    state tokens remain visible. Only the updated joint token is decoded.
    Masks come from the task's eight validated modes; zero-mask spec probes
    are numerically supported but are not an additional control mode.
    """

    def __init__(
        self,
        action_dim: int = 29,
        num_links: int = 6,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 1024,
        init_noise_scale: float = 1.0,
        activation_checkpointing: bool = False,
        activation_checkpoint_batch_size: int = 8192,
        compliance: bool = False,
        axis: bool = False,
    ) -> None:
        super().__init__()
        if action_dim != 29:
            raise ValueError("GoalBodyActor v0.1 requires 29 G1 actions.")
        if min(embed_dim, num_heads, num_layers, ff_dim) <= 0 or embed_dim % num_heads:
            raise ValueError("Positive Transformer sizes and embed_dim % num_heads == 0 required.")
        if not math.isfinite(init_noise_scale) or init_noise_scale <= 0:
            raise ValueError("init_noise_scale must be finite and positive.")
        if num_links not in (3, 6):
            raise ValueError("GoalBodyActor supports three or six links.")
        self.num_links = num_links
        self.embed_dim = embed_dim
        self.init_noise_scale = float(init_noise_scale)
        self.activation_checkpointing = bool(activation_checkpointing)
        if activation_checkpoint_batch_size <= 0:
            raise ValueError("Activation checkpoint batch size must be positive.")
        self.activation_checkpoint_batch_size = int(activation_checkpoint_batch_size)
        self.compliance = compliance
        self.axis = axis
        if axis and not compliance:
            raise ValueError("Axis input requires compliance")
        self.command_mask = LinkCommandMask(num_links, compliance, axis)
        self.goal_encoder = nn.Sequential(
            nn.Linear(105 + int(compliance) + 3 * int(axis), embed_dim), nn.LayerNorm(embed_dim), nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(493, 512), nn.LayerNorm(512), nn.Mish(),
            nn.Linear(512, embed_dim),
        )
        self.root_encoder = nn.Sequential(
            nn.Linear(63, embed_dim), nn.LayerNorm(embed_dim), nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Six link identities followed by joint-state and root-state identities.
        self.token_identity = nn.Parameter(torch.empty(num_links + 2, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )
        self.action_mean = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.Mish(),
            nn.Linear(embed_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.empty(action_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # PPO's legacy blanket 0.01 Linear initialization must not shrink
        # Transformer projections; reinitialize this actor after that pass.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)
        nn.init.normal_(self.token_identity, std=0.02)
        nn.init.orthogonal_(self.action_mean[-1].weight, gain=0.01)
        nn.init.constant_(self.log_std, math.log(self.init_noise_scale))

    @property
    def action_std(self) -> torch.Tensor:
        return self.log_std.clamp(-5.0, 1.0).exp()

    def forward(
        self,
        policy: torch.Tensor,
        command: torch.Tensor,
        root_history: torch.Tensor,
        link_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (self.activation_checkpointing and torch.is_grad_enabled()):
            return self._forward(policy, command, root_history, link_mask)
        batch_shape = policy.shape[:-1]
        if any(x.shape[:-1] != batch_shape for x in (command, root_history, link_mask)):
            raise ValueError("All actor inputs must have the same batch shape.")
        inputs = [x.reshape(-1, x.shape[-1]) for x in (policy, command, root_history, link_mask)]
        chunks = []
        for start in range(0, inputs[0].shape[0], self.activation_checkpoint_batch_size):
            stop = start + self.activation_checkpoint_batch_size
            chunks.append(checkpoint(
                self._forward, *(x[start:stop] for x in inputs),
                use_reentrant=False,
            ))
        # PPO still sees one full minibatch and takes one optimizer step.
        return tuple(
            torch.cat([chunk[index] for chunk in chunks], dim=0).reshape(*batch_shape, 29)
            for index in (0, 1)
        )

    def _forward(
        self,
        policy: torch.Tensor,
        command: torch.Tensor,
        root_history: torch.Tensor,
        link_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = policy.shape[:-1]
        if policy.shape[-1] != 493 or root_history.shape != (*batch_shape, 63):
            raise ValueError("Expected joint history [...,493] and root history [...,63].")
        if command.shape != (*batch_shape, (105 + int(self.compliance)) * self.num_links + 3 * int(self.axis)):
            raise ValueError(f"Expected {self.num_links} link-major goal slots.")
        command = self.command_mask(command, link_mask)
        slots = command[..., :105*self.num_links].reshape(-1, self.num_links, 105)
        if self.compliance:
            slots = torch.cat((slots, command[..., 105*self.num_links:106*self.num_links].reshape(-1, self.num_links, 1)), -1)
        if self.axis:
            shared = command[..., -3:].reshape(-1, 1, 3).expand(-1, self.num_links, -1)
            shared = torch.where(link_mask.reshape(-1, self.num_links, 1) > .5, shared, 0.)
            slots = torch.cat((slots, shared), -1)
        goals = self.goal_encoder(slots)
        joints = self.joint_encoder(policy.reshape(-1, 493)).unsqueeze(1)
        root = self.root_encoder(root_history.reshape(-1, 63)).unsqueeze(1)
        tokens = torch.cat((goals, joints, root), dim=1)
        tokens = tokens + self.token_identity.to(tokens.dtype)
        inactive_goals = link_mask.reshape(-1, self.num_links) <= 0.5
        padding_mask = torch.cat(
            (inactive_goals, torch.zeros_like(inactive_goals[:, :2])), dim=1,
        )
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        loc = self.action_mean(encoded[:, self.num_links]).reshape(*batch_shape, 29)
        return loc, self.action_std.expand_as(loc)


class GoalBodyFlatActor(nn.Module):
    """Adapter for MimicLite's existing 556D policy observation group.

    The legacy group stores root history around the joint fields. Keeping this
    adapter avoids changing the simulator observation contract while exposing
    the proposed 493D joint token and 63D root token to the new actor.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.actor = GoalBodyActor(**kwargs)

    @property
    def action_std(self) -> torch.Tensor:
        return self.actor.action_std

    def forward(
        self, policy: torch.Tensor, command: torch.Tensor, link_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if policy.shape[-1] != 556:
            raise ValueError("GoalBodyFlatActor expects legacy 556D policy observations.")
        joint_history = policy[..., 42:535]
        root_history = torch.cat((policy[..., :42], policy[..., 535:556]), dim=-1)
        return self.actor(joint_history, command, root_history, link_mask)
