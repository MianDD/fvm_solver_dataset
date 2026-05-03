"""Patch-transformer foundation model for varying-PDE 2D fluid prediction.

Design notes (and how this differs from FLUID-LLM and the supervisor's brief):

    * **FLUID-LLM** (Zhu, Bazaga, Lio 2024 — your supervisor's own paper)
      trains a pretrained large language model on a *single fixed* fluid
      problem. The model adapts to one task; in-context learning is
      observed only as a small (~3%) effect.

    * **This project** trains across a *family* of PDEs and asks the model
      to *infer* which member of the family generated the context
      observations. This is the foundation-model formulation studied in
      Poseidon (Herde et al. 2024), Walrus (McCabe et al. 2025), and TNT
      (Dang et al. 2022), and is analogous to TabPFN's transformer prior
      over tabular tasks (Hollmann et al. 2023).

    * For computational tractability on a coursework budget, we drop the
      pretrained LLM weights — a from-scratch Transformer encoder is
      sufficient to demonstrate the architectural ideas (patch encoding,
      spatiotemporal embedding, autoregressive next-state prediction).
      Pretrained-LLM initialisation can be added later by replacing the
      transformer block with e.g. an OPT-125m backbone; only this file
      would need to change.

Architecture
------------
    state s_t in R^(C x H x W)               # C=4: V_x, V_y, rho, T
    -> per-channel z-score normalisation
    -> partition into N = (H/P)(W/P) non-overlapping P x P patches
    -> patch encoder (linear -> linear)      -> patch tokens R^(N x d)
    -> add spatial (x, y) embedding
    -> add temporal embedding (relative t-index in context)
    -> stack tokens for tau context steps    -> seq length = tau * N
    -> Transformer encoder with **block-causal** attention mask
       (future-step tokens are masked out so the model is autoregressive
        in time, all-to-all in space within each step)
    -> patch decoder (linear -> linear -> P x P x C)
    -> reassemble into a delta-state
    -> next state = current state + delta
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Patch / un-patch helpers
# --------------------------------------------------------------------------
def to_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """x: (B, C, H, W) -> (B, N, C * P * P) with N = (H/P)(W/P)."""
    B, C, H, W = x.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, \
        f"H,W must be divisible by P (got H={H}, W={W}, P={P})"
    x = x.reshape(B, C, H // P, P, W // P, P)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.reshape(B, (H // P) * (W // P), C * P * P)
    return x


def from_patches(tokens: torch.Tensor, C: int, H: int, W: int, P: int
                 ) -> torch.Tensor:
    """tokens: (B, N, C * P * P) -> (B, C, H, W)."""
    B, N, _ = tokens.shape
    Hp, Wp = H // P, W // P
    assert N == Hp * Wp
    x = tokens.reshape(B, Hp, Wp, C, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.reshape(B, C, H, W)
    return x


# --------------------------------------------------------------------------
# Spatiotemporal embedding
# --------------------------------------------------------------------------
class SpatiotemporalEmbedding(nn.Module):
    """Adds learnable (x, y, t) embeddings to a sequence of patch tokens."""

    def __init__(self, max_x: int, max_y: int, max_t: int, d_model: int):
        super().__init__()
        d3 = d_model // 3
        self.dx = d3
        self.dy = d_model - 2 * d3   # remaining dims absorbed into y
        self.dt = d3
        self.x_emb = nn.Embedding(max_x, self.dx)
        self.y_emb = nn.Embedding(max_y, self.dy)
        self.t_emb = nn.Embedding(max_t, self.dt)

    def forward(self, tokens: torch.Tensor, n_x: int, n_y: int) -> torch.Tensor:
        B, tau, N, d = tokens.shape
        assert N == n_x * n_y, f"N={N} != n_x*n_y={n_x*n_y}"
        device = tokens.device
        xs = torch.arange(n_x, device=device).repeat_interleave(n_y)
        ys = torch.arange(n_y, device=device).repeat(n_x)
        ts = torch.arange(tau, device=device)
        ex = self.x_emb(xs)                                    # (N, dx)
        ey = self.y_emb(ys)                                    # (N, dy)
        et = self.t_emb(ts)                                    # (tau, dt)
        spatial = torch.cat([ex, ey], dim=-1)                  # (N, dx+dy)
        spatial = spatial.unsqueeze(0).unsqueeze(0)            # (1,1,N,dx+dy)
        spatial = F.pad(spatial, (0, self.dt))                 # (1,1,N,d)
        temporal = F.pad(et, (self.dx + self.dy, 0))           # (tau, d)
        temporal = temporal.unsqueeze(0).unsqueeze(2)          # (1,tau,1,d)
        return tokens + spatial + temporal


# --------------------------------------------------------------------------
# Channel normaliser (running statistics)
# --------------------------------------------------------------------------
class ChannelNormaliser(nn.Module):
    """Per-channel z-score normaliser. Critical because the four channels
    [V_x, V_y, rho, T] live on very different scales (T ~ 100, rho ~ 1, ...).
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_channels))
        self.register_buffer("std", torch.ones(n_channels))
        self.n_channels = n_channels
        self._initialised = False

    def fit(self, dataset_iter):
        """Compute statistics from a dataset iterator."""
        sums = torch.zeros(self.n_channels)
        sums2 = torch.zeros(self.n_channels)
        n = 0
        for x in dataset_iter:
            x_flat = x.reshape(-1, self.n_channels, x.shape[-2] * x.shape[-1])
            x_flat = x_flat.transpose(1, 2).reshape(-1, self.n_channels)
            sums += x_flat.sum(0)
            sums2 += (x_flat ** 2).sum(0)
            n += x_flat.shape[0]
        m = sums / max(n, 1)
        v = (sums2 / max(n, 1) - m ** 2).clamp_min(1e-6)
        self.mean.copy_(m)
        self.std.copy_(v.sqrt())
        self._initialised = True

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mean.view(1, -1, 1, 1)
        s = self.std.view(1, -1, 1, 1)
        return (x - m) / s

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mean.view(1, -1, 1, 1)
        s = self.std.view(1, -1, 1, 1)
        return x * s + m


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
class FoundationCFDModel(nn.Module):
    """Predict update fields for CFD states.

    ``n_input_channels`` can be larger than ``n_target_channels`` when
    derivative features are concatenated to the physical input channels.
    The model always predicts updates for the target physical channels only.
    """

    def __init__(self, n_channels: int = 4, H: int = 48, W: int = 72,
                 n_input_channels: int | None = None,
                 n_target_channels: int | None = None,
                 patch_size: int = 8, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_context: int = 8, dropout: float = 0.0,
                 mlp_ratio: float = 4.0):
        super().__init__()
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"H={H}, W={W} must be divisible by patch_size={patch_size}"
        self.C_in = int(n_input_channels or n_channels)
        self.C = int(n_target_channels or n_channels)
        self.H = H; self.W = W; self.P = patch_size
        self.n_x = H // patch_size
        self.n_y = W // patch_size
        self.N = self.n_x * self.n_y
        self.d_model = d_model
        self.max_context = max_context
        self.mlp_ratio = mlp_ratio

        self.normaliser = ChannelNormaliser(self.C_in)

        patch_dim_in = self.C_in * patch_size * patch_size
        patch_dim_out = self.C * patch_size * patch_size
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_dim_in, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.spatiotemporal = SpatiotemporalEmbedding(
            max_x=self.n_x, max_y=self.n_y,
            max_t=max_context, d_model=d_model,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=int(mlp_ratio * d_model), dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.patch_decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, patch_dim_out),
        )
        # Lightweight CNN refinement smooths patch boundaries
        self.refine = nn.Sequential(
            nn.Conv2d(self.C, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),         nn.GELU(),
            nn.Conv2d(32, self.C, 3, padding=1),
        )

    def encode_context(self, states: torch.Tensor) -> torch.Tensor:
        B, tau, C, H, W = states.shape
        assert C == self.C_in, f"Expected {self.C_in} input channels, got {C}"
        x = states.reshape(B * tau, C, H, W)
        tokens = to_patches(x, self.P)                  # (B*tau, N, C*P*P)
        tokens = self.patch_encoder(tokens)              # (B*tau, N, d)
        tokens = tokens.reshape(B, tau, self.N, self.d_model)
        tokens = self.spatiotemporal(tokens, self.n_x, self.n_y)
        return tokens.reshape(B, tau * self.N, self.d_model)

    def _block_causal_mask(self, tau: int, device) -> torch.Tensor:
        S = tau * self.N
        i_step = torch.arange(S, device=device) // self.N
        j_step = torch.arange(S, device=device) // self.N
        return j_step.unsqueeze(0) > i_step.unsqueeze(1)

    def predict_update(self, states: torch.Tensor, normalised: bool = False
                       ) -> torch.Tensor:
        """Return model-predicted delta or dX/dt fields.

        The interpretation is chosen by the caller via ``prediction_mode``.
        Shape: ``(B, tau, n_target_channels, H, W)``.
        """
        if not normalised:
            x = self.normaliser.normalise(states.reshape(-1, self.C_in, self.H, self.W))
            states_n = x.reshape(*states.shape)
        else:
            states_n = states
        B, tau, C, H, W = states_n.shape

        tokens = self.encode_context(states_n)
        mask = self._block_causal_mask(tau, tokens.device)
        out = self.transformer(tokens, mask=mask)
        out = out.reshape(B, tau, self.N, self.d_model)

        flat = out.reshape(B * tau, self.N, self.d_model)
        patches = self.patch_decoder(flat)
        delta = from_patches(patches, self.C, H, W, self.P)
        delta = delta + self.refine(delta)
        return delta.reshape(B, tau, self.C, H, W)

    @staticmethod
    def integrate_update(current_state: torch.Tensor, update: torch.Tensor,
                         prediction_mode: str = "delta",
                         integrator: str = "euler",
                         dt: torch.Tensor | float | None = None) -> torch.Tensor:
        """Convert a predicted update field into a next state."""
        if prediction_mode == "delta":
            return current_state + update
        if prediction_mode != "derivative":
            raise ValueError(f"Unknown prediction_mode: {prediction_mode}")
        if integrator != "euler":
            raise ValueError(f"Unsupported integrator: {integrator}")
        if dt is None:
            scale = torch.as_tensor(1.0, dtype=update.dtype, device=update.device)
        elif isinstance(dt, torch.Tensor):
            scale = dt.to(dtype=update.dtype, device=update.device)
        else:
            scale = torch.as_tensor(float(dt), dtype=update.dtype, device=update.device)
        if scale.dim() == 1:
            scale = scale.view(scale.shape[0], *([1] * (update.dim() - 1)))
        return current_state + scale * update

    def forward(self, states: torch.Tensor, normalised: bool = False,
                prediction_mode: str = "delta", integrator: str = "euler",
                dt: torch.Tensor | float | None = None,
                current_state: torch.Tensor | None = None,
                return_update: bool = False) -> torch.Tensor:
        update = self.predict_update(states, normalised=normalised)
        if return_update:
            return update
        if current_state is None:
            if states.shape[2] < self.C:
                raise ValueError("current_state is required when inputs do not contain target channels")
            current_state = states[:, :, :self.C]
        return self.integrate_update(
            current_state,
            update,
            prediction_mode=prediction_mode,
            integrator=integrator,
            dt=dt,
        )

    @torch.no_grad()
    def rollout(self, context: torch.Tensor, n_steps: int,
                prediction_mode: str = "delta", integrator: str = "euler",
                dt: torch.Tensor | float | None = None) -> torch.Tensor:
        """context: (B, tau0, C, H, W) -> predictions: (B, n_steps, C, H, W).

        This convenience method is for plain physical-channel contexts.  For
        derivative-feature checkpoints, use the evaluation code path that
        rebuilds derivative input features after each autoregressive step.
        """
        self.eval()
        states = context
        preds = []
        for _ in range(n_steps):
            context_in = states[:, -self.max_context:]
            update = self.predict_update(context_in, normalised=False)[:, -1]
            current = context_in[:, -1, :self.C]
            next_state = self.integrate_update(
                current,
                update,
                prediction_mode=prediction_mode,
                integrator=integrator,
                dt=dt,
            ).unsqueeze(1)
            preds.append(next_state)
            states = torch.cat([states, next_state], dim=1)
        return torch.cat(preds, dim=1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
