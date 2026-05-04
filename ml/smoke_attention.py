"""Smoke test for global and factorized Patch Transformer attention paths."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import torch

from .checkpoint import load_model_from_checkpoint
from .model import FoundationCFDModel, count_params


B = 2
CONTEXT = 4
C_IN = 16
C_OUT = 4
H = 64
W = 96
PATCH = 8
D_MODEL = 32
HEADS = 4
LAYERS = 1


def _complexity_summary() -> None:
    n_patches = (H // PATCH) * (W // PATCH)
    global_tokens = CONTEXT * n_patches
    global_pairs = global_tokens ** 2
    factorized_pairs = CONTEXT * (n_patches ** 2) + n_patches * (CONTEXT ** 2)
    reduction = global_pairs / factorized_pairs
    print("Attention smoke configuration")
    print(f"  H={H} W={W} patch={PATCH} context={CONTEXT}")
    print(f"  patches per frame N={n_patches}")
    print(f"  total global tokens T*N={global_tokens}")
    print(f"  global attention pairs (T*N)^2={global_pairs}")
    print(f"  factorized attention pairs T*N^2 + N*T^2={factorized_pairs}")
    print(f"  pair-count reduction={reduction:.2f}x")


def _make_model(attention_type: str) -> FoundationCFDModel:
    return FoundationCFDModel(
        n_channels=C_OUT,
        n_input_channels=C_IN,
        n_target_channels=C_OUT,
        H=H,
        W=W,
        patch_size=PATCH,
        d_model=D_MODEL,
        n_heads=HEADS,
        n_layers=LAYERS,
        max_context=CONTEXT,
        dropout=0.0,
        attention_type=attention_type,
    )


def _check(attention_type: str) -> FoundationCFDModel:
    torch.manual_seed(0)
    model = _make_model(attention_type)
    x = torch.randn(B, CONTEXT, C_IN, H, W)
    update_sequence = model.predict_update(x)
    update = update_sequence[:, -1]
    expected = (B, C_OUT, H, W)
    if tuple(update.shape) != expected:
        raise RuntimeError(
            f"{attention_type}: expected output shape {expected}, got {tuple(update.shape)}"
        )
    loss = update.square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    has_grad = bool(grads)
    finite_grads = all(torch.isfinite(g).all() for g in grads)
    if not has_grad:
        raise RuntimeError(f"{attention_type}: no finite gradients found")
    if not finite_grads:
        raise RuntimeError(f"{attention_type}: non-finite gradients found")
    print(
        f"  {attention_type}: params={count_params(model):,} "
        f"sequence_shape={tuple(update_sequence.shape)} "
        f"final_output_shape={tuple(update.shape)} loss={float(loss.detach()):.4e}"
    )
    return model


def _check_checkpoint_compat(global_model: FoundationCFDModel,
                             factorized_model: FoundationCFDModel) -> None:
    base_model_config = {
        "n_channels": C_OUT,
        "n_input_channels": C_IN,
        "n_target_channels": C_OUT,
        "H": H,
        "W": W,
        "patch_size": PATCH,
        "d_model": D_MODEL,
        "n_heads": HEADS,
        "n_layers": LAYERS,
        "max_context": CONTEXT,
        "dropout": 0.0,
        "mlp_ratio": 4.0,
    }
    tmp_dir = Path("checkpoints") / f"_codex_attention_checkpoint_smoke_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        old_path = tmp_dir / "old_global_missing_attention.pt"
        torch.save({
            "model": global_model.state_dict(),
            "model_config": dict(base_model_config),
        }, old_path)
        loaded_global, _ = load_model_from_checkpoint(old_path, device="cpu")
        if loaded_global.attention_type != "global":
            raise RuntimeError("Old checkpoint without attention_type did not default to global")

        factorized_path = tmp_dir / "factorized.pt"
        factorized_config = dict(base_model_config)
        factorized_config["attention_type"] = "factorized"
        torch.save({
            "model": factorized_model.state_dict(),
            "model_config": factorized_config,
        }, factorized_path)
        loaded_factorized, _ = load_model_from_checkpoint(factorized_path, device="cpu")
        if loaded_factorized.attention_type != "factorized":
            raise RuntimeError("Factorized checkpoint did not reconstruct a factorized model")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  checkpoint compatibility: old missing attention_type -> global; factorized reload OK")


def main() -> None:
    _complexity_summary()
    global_model = _check("global")
    factorized_model = _check("factorized")
    _check_checkpoint_compat(global_model, factorized_model)
    print("OK")


if __name__ == "__main__":
    main()
