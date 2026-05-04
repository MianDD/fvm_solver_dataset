"""Patch-transformer foundation model for varying-PDE 2D fluid prediction.

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
       Optional experimental factorized attention keeps tokens as (B,T,N,D)
       and alternates spatial attention over N with causal temporal attention
       over T.
    -> patch decoder (linear -> linear -> P x P x C)
    -> reassemble into a delta-state
    -> next state = current state + delta
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ATTENTION_TYPES = ("global", "factorized")
POSITION_ENCODINGS = ("learned_absolute", "sinusoidal")


# --------------------------------------------------------------------------
# Patch / un-patch helpers
# --------------------------------------------------------------------------
def to_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """x: (B, C, H, W) -> (B, N, C * P * P) with N = (H/P)(W/P)."""
    B, C, H, W = x.shape
    P = patch_size
    if H % P != 0 or W % P != 0:
        raise ValueError(
            f"H and W must be divisible by patch_size before patchifying "
            f"(got H={H}, W={W}, patch_size={P})."
        )
    x = x.reshape(B, C, H // P, P, W // P, P)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.reshape(B, (H // P) * (W // P), C * P * P)
    return x


def from_patches(tokens: torch.Tensor, C: int, H: int, W: int, P: int
                 ) -> torch.Tensor:
    """tokens: (B, N, C * P * P) -> (B, C, H, W)."""
    B, N, _ = tokens.shape
    Hp, Wp = H // P, W // P
    if N != Hp * Wp:
        raise ValueError(f"Expected {Hp * Wp} patches for H={H}, W={W}, P={P}; got {N}.")
    x = tokens.reshape(B, Hp, Wp, C, P, P)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.reshape(B, C, H, W)
    return x


# --------------------------------------------------------------------------
# Spatiotemporal embedding
# --------------------------------------------------------------------------
class SpatiotemporalEmbedding(nn.Module):
    """Adds learnable or sinusoidal (x, y, t) embeddings to patch tokens."""

    def __init__(self, max_x: int, max_y: int, max_t: int, d_model: int,
                 mode: str = "learned_absolute"):
        super().__init__()
        if mode not in POSITION_ENCODINGS:
            allowed = ", ".join(repr(name) for name in POSITION_ENCODINGS)
            raise ValueError(f"pos_encoding must be one of {allowed}; got {mode!r}.")
        self.mode = mode
        self.d_model = d_model
        d3 = d_model // 3
        self.dx = d3
        self.dy = d_model - 2 * d3   # remaining dims absorbed into y
        self.dt = d3
        if mode == "learned_absolute":
            self.x_emb = nn.Embedding(max_x, self.dx)
            self.y_emb = nn.Embedding(max_y, self.dy)
            self.t_emb = nn.Embedding(max_t, self.dt)

    @staticmethod
    def _sinusoidal(indices: torch.Tensor, dim: int) -> torch.Tensor:
        if dim <= 0:
            return indices.new_zeros((indices.numel(), 0), dtype=torch.float32)
        pos = indices.to(dtype=torch.float32).unsqueeze(1)
        half = (dim + 1) // 2
        freq = torch.exp(
            torch.arange(half, device=indices.device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=indices.device)) / max(half, 1))
        )
        angles = pos * freq.unsqueeze(0)
        enc = torch.zeros(indices.numel(), dim, device=indices.device, dtype=torch.float32)
        enc[:, 0::2] = torch.sin(angles[:, :enc[:, 0::2].shape[1]])
        if dim > 1:
            enc[:, 1::2] = torch.cos(angles[:, :enc[:, 1::2].shape[1]])
        return enc

    def forward(self, tokens: torch.Tensor, n_x: int, n_y: int) -> torch.Tensor:
        B, tau, N, d = tokens.shape
        if N != n_x * n_y:
            raise ValueError(f"N={N} != n_x*n_y={n_x*n_y}")
        device = tokens.device
        xs = torch.arange(n_x, device=device).repeat_interleave(n_y)
        ys = torch.arange(n_y, device=device).repeat(n_x)
        ts = torch.arange(tau, device=device)
        if self.mode == "learned_absolute":
            ex = self.x_emb(xs)                                # (N, dx)
            ey = self.y_emb(ys)                                # (N, dy)
            et = self.t_emb(ts)                                # (tau, dt)
        else:
            # Sinusoidal mode is parameter-free and can extrapolate to new
            # context lengths or patch-grid sizes without learned table limits.
            ex = self._sinusoidal(xs, self.dx)                 # (N, dx)
            ey = self._sinusoidal(ys, self.dy)                 # (N, dy)
            et = self._sinusoidal(ts, self.dt)                 # (tau, dt)
        spatial = torch.cat([ex, ey], dim=-1)                  # (N, dx+dy)
        spatial = spatial.unsqueeze(0).unsqueeze(0)            # (1,1,N,dx+dy)
        spatial = F.pad(spatial, (0, self.dt))                 # (1,1,N,d)
        temporal = F.pad(et, (self.dx + self.dy, 0))           # (tau, d)
        temporal = temporal.unsqueeze(0).unsqueeze(2)          # (1,tau,1,d)
        return tokens + spatial + temporal


# --------------------------------------------------------------------------
# Factorized space-time attention
# --------------------------------------------------------------------------
class FactorizedSpaceTimeBlock(nn.Module):
    """TimeSformer/ViViT-style block over structured (B, T, N, D) tokens."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        hidden = int(mlp_ratio * d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self._temporal_mask_cache: dict[tuple[int, str, int | None], torch.Tensor] = {}

    def _temporal_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Bool mask for causal temporal attention; True entries are masked."""
        key = (T, device.type, device.index)
        mask = self._temporal_mask_cache.get(key)
        if mask is None or mask.device != device:
            mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=device),
                diagonal=1,
            )
            self._temporal_mask_cache[key] = mask
        return mask

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T, N, D = tokens.shape

        # Spatial attention: each time step attends across its N spatial patches.
        # (B, T, N, D) -> (B*T, N, D)
        spatial_in = self.spatial_norm(tokens).reshape(B * T, N, D)
        spatial_out, _ = self.spatial_attn(
            spatial_in, spatial_in, spatial_in, need_weights=False,
        )
        tokens = tokens + self.dropout(spatial_out.reshape(B, T, N, D))

        # Temporal attention: each patch index attends backward over T context frames.
        # (B, T, N, D) -> (B, N, T, D) -> (B*N, T, D)
        temporal_in = self.temporal_norm(tokens).permute(0, 2, 1, 3).reshape(B * N, T, D)
        temporal_out, _ = self.temporal_attn(
            temporal_in, temporal_in, temporal_in,
            attn_mask=self._temporal_causal_mask(T, tokens.device),
            need_weights=False,
        )
        # Restore the structured token layout for the next factorized block.
        # (B*N, T, D) -> (B, N, T, D) -> (B, T, N, D)
        temporal_out = temporal_out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        tokens = tokens + self.dropout(temporal_out)

        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        return tokens


class FactorizedSpaceTimeEncoder(nn.Module):
    """Stack of factorized space-time attention blocks."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            FactorizedSpaceTimeBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens


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
                 mlp_ratio: float = 4.0, attention_type: str = "global",
                 pos_encoding: str = "learned_absolute",
                 pde_dim: int = 0):
        super().__init__()
        if attention_type not in ATTENTION_TYPES:
            allowed = ", ".join(repr(name) for name in ATTENTION_TYPES)
            raise ValueError(f"attention_type must be one of {allowed}; got {attention_type!r}.")
        if pos_encoding not in POSITION_ENCODINGS:
            allowed = ", ".join(repr(name) for name in POSITION_ENCODINGS)
            raise ValueError(f"pos_encoding must be one of {allowed}; got {pos_encoding!r}.")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive; got {patch_size}.")
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"H and W must be divisible by patch_size before patchifying "
                f"(got H={H}, W={W}, patch_size={patch_size})."
            )
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive; got {n_heads}.")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.C_in = int(n_input_channels or n_channels)
        self.C = int(n_target_channels or n_channels)
        self.H = H; self.W = W; self.P = patch_size
        self.n_x = H // patch_size
        self.n_y = W // patch_size
        self.N = self.n_x * self.n_y
        self.d_model = d_model
        self.max_context = max_context
        self.mlp_ratio = mlp_ratio
        self.attention_type = attention_type
        self.pos_encoding = pos_encoding
        self.pde_dim = int(pde_dim or 0)

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
            mode=pos_encoding,
        )
        if attention_type == "global":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=int(mlp_ratio * d_model), dropout=dropout,
                batch_first=True, activation="gelu", norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            self.factorized_encoder = FactorizedSpaceTimeEncoder(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                mlp_ratio=mlp_ratio, dropout=dropout,
            )
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
        self.pde_head = (
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.pde_dim))
            if self.pde_dim > 0 else None
        )

    def encode_context_tokens(self, states: torch.Tensor) -> torch.Tensor:
        B, tau, C, H, W = states.shape
        if C != self.C_in:
            raise ValueError(f"Expected {self.C_in} input channels, got {C}.")
        if H % self.P != 0 or W % self.P != 0:
            raise ValueError(
                f"Grid shape H={H}, W={W} must be divisible by patch_size={self.P}."
            )
        n_x, n_y = H // self.P, W // self.P
        if self.pos_encoding == "learned_absolute" and tau > self.max_context:
            raise ValueError(f"Context length {tau} exceeds max_context={self.max_context}.")
        if self.pos_encoding == "learned_absolute" and (n_x > self.n_x or n_y > self.n_y):
            raise ValueError(
                "learned_absolute position encoding was initialised for patch grid "
                f"{self.n_x}x{self.n_y}, got {n_x}x{n_y}. Use --pos-encoding sinusoidal "
                "for parameter-free positional features."
            )
        x = states.reshape(B * tau, C, H, W)
        tokens = to_patches(x, self.P)                  # (B*tau, N, C*P*P)
        n_patches = tokens.shape[1]
        tokens = self.patch_encoder(tokens)              # (B*tau, N, d)
        tokens = tokens.reshape(B, tau, n_patches, self.d_model)  # (B, T, N, D)
        tokens = self.spatiotemporal(tokens, n_x, n_y)
        return tokens

    def encode_context(self, states: torch.Tensor) -> torch.Tensor:
        B, tau, _, _, _ = states.shape
        tokens = self.encode_context_tokens(states)
        n_patches = tokens.shape[2]
        return tokens.reshape(B, tau * n_patches, self.d_model)

    def _block_causal_mask(self, tau: int, n_patches: int, device) -> torch.Tensor:
        S = tau * n_patches
        i_step = torch.arange(S, device=device) // n_patches
        j_step = torch.arange(S, device=device) // n_patches
        return j_step.unsqueeze(0) > i_step.unsqueeze(1)

    def _encoded_tokens(self, states: torch.Tensor, normalised: bool = False
                        ) -> tuple[torch.Tensor, int, int, int, int, int, int]:
        if states.dim() != 5:
            raise ValueError(f"Expected states with shape (B, T, C, H, W); got {tuple(states.shape)}.")
        B, tau, C, H, W = states.shape
        if C != self.C_in:
            raise ValueError(f"Expected {self.C_in} input channels, got {C}.")
        if H % self.P != 0 or W % self.P != 0:
            raise ValueError(
                f"Grid shape H={H}, W={W} must be divisible by patch_size={self.P}."
            )
        if self.pos_encoding == "learned_absolute" and tau > self.max_context:
            raise ValueError(f"Context length {tau} exceeds max_context={self.max_context}.")
        if not normalised:
            x = self.normaliser.normalise(states.reshape(-1, self.C_in, H, W))
            states_n = x.reshape(*states.shape)
        else:
            states_n = states

        tokens_structured = self.encode_context_tokens(states_n)  # (B, T, N, D)
        n_patches = tokens_structured.shape[2]
        if self.attention_type == "global":
            # Legacy baseline path: flatten time and space into one block-causal
            # sequence, then decode all T context steps for checkpoint/repro
            # compatibility. Downstream code selects the final slice with [:, -1].
            tokens = tokens_structured.reshape(B, tau * n_patches, self.d_model)
            mask = self._block_causal_mask(tau, n_patches, tokens.device)
            out = self.transformer(tokens, mask=mask)
            out = out.reshape(B, tau, n_patches, self.d_model)
        else:
            # Experimental path: keep tokens structured as (B, T, N, D), use
            # factorized spatial/temporal attention.
            out = self.factorized_encoder(tokens_structured)
        return out, B, tau, H, W, C, n_patches

    def predict_update_and_pde(self, states: torch.Tensor, normalised: bool = False
                               ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return predicted update fields and optional PDE-parameter logits.

        The update interpretation is chosen by the caller via
        ``prediction_mode``. Global attention preserves the legacy update shape
        ``(B, T, C, H, W)`` by decoding every context step. Factorized attention
        decodes only the final context step and returns ``(B, 1, C, H, W)``.
        """
        out, B, tau, H, W, _, n_patches = self._encoded_tokens(states, normalised=normalised)
        if self.attention_type == "global":
            flat = out.reshape(B * tau, n_patches, self.d_model)
            out_tau = tau
        else:
            flat = out[:, -1].reshape(B, n_patches, self.d_model)
            out_tau = 1
        patches = self.patch_decoder(flat)
        delta = from_patches(patches, self.C, H, W, self.P)
        delta = delta + self.refine(delta)
        update = delta.reshape(B, out_tau, self.C, H, W)
        pde_pred = self.pde_head(out.mean(dim=(1, 2))) if self.pde_head is not None else None
        return update, pde_pred

    def predict_update(self, states: torch.Tensor, normalised: bool = False
                       ) -> torch.Tensor:
        update, _ = self.predict_update_and_pde(states, normalised=normalised)
        return update

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
            if update.shape[1] == 1:
                current_state = states[:, -1:, :self.C]
            else:
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
