"""Activation aggregation and layer-index contracts (local-safe utilities)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

TRANSFORMER_BLOCK_INDEX_PHASE2 = 12
HOOK_MODULE_PATH_PHASE2 = "model.model.layers[12]"


@dataclass(frozen=True)
class LayerIndexContract:
    """Documented mapping between literature 'layer 12' and instrumentation."""

    transformer_block_index: int = TRANSFORMER_BLOCK_INDEX_PHASE2
    hook_module_path: str = HOOK_MODULE_PATH_PHASE2

    def hidden_states_tuple_index_if_output_hidden_states(self) -> int:
        """If using model(..., output_hidden_states=True), embedding is index 0.

        Therefore the output of transformer block i is hidden_states[i + 1].
        Phase 2 primary path uses a forward hook on layers[i] instead.
        """
        return self.transformer_block_index + 1


def assert_layer_contract(
    transformer_block_index: int,
    hook_module_path: str,
    *,
    expected_index: int = TRANSFORMER_BLOCK_INDEX_PHASE2,
) -> None:
    if transformer_block_index != expected_index:
        raise ValueError(
            f"Expected transformer_block_index={expected_index}, got {transformer_block_index}"
        )
    expected_path = f"model.model.layers[{expected_index}]"
    if hook_module_path != expected_path:
        raise ValueError(f"Expected hook_module_path={expected_path!r}, got {hook_module_path!r}")


def resolve_block_module(model: torch.nn.Module, transformer_block_index: int) -> torch.nn.Module:
    """Return model.model.layers[transformer_block_index] for Mistral-like models."""
    try:
        layers = model.model.layers
    except AttributeError as exc:
        raise AttributeError("Model does not expose model.model.layers") from exc
    if transformer_block_index < 0 or transformer_block_index >= len(layers):
        raise IndexError(
            f"transformer_block_index {transformer_block_index} "
            f"out of range for {len(layers)} layers"
        )
    return layers[transformer_block_index]


def mean_pool_tokens(
    token_activations: torch.Tensor | np.ndarray,
    attention_mask: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Mean over non-padding tokens: sum(valid) / n_valid.

    token_activations: [batch, seq, hidden]
    attention_mask: [batch, seq] with 1 = real token, 0 = padding
    returns: [batch, hidden]
    """
    if isinstance(token_activations, np.ndarray):
        token_activations = torch.from_numpy(token_activations)
    if isinstance(attention_mask, np.ndarray):
        attention_mask = torch.from_numpy(attention_mask)

    if token_activations.ndim != 3:
        raise ValueError(f"Expected [B, T, H], got shape {tuple(token_activations.shape)}")
    if attention_mask.shape != token_activations.shape[:2]:
        raise ValueError(
            f"Mask shape {tuple(attention_mask.shape)} incompatible with "
            f"activations {tuple(token_activations.shape)}"
        )

    mask = attention_mask.to(dtype=token_activations.dtype)
    # Zero out padding positions explicitly (safe for both left and right padding).
    masked = token_activations * mask.unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
    return masked.sum(dim=1) / denom


def mean_pool_single(
    token_activations: torch.Tensor | np.ndarray,
    attention_mask: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Mean-pool a single example [T, H] with mask [T]."""
    if isinstance(token_activations, np.ndarray):
        token_activations = torch.from_numpy(token_activations)
    if isinstance(attention_mask, np.ndarray):
        attention_mask = torch.from_numpy(attention_mask)
    return mean_pool_tokens(token_activations.unsqueeze(0), attention_mask.unsqueeze(0))[0]
