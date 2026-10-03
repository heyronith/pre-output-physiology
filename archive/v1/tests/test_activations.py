"""Activation aggregation and layer-index contract tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pre_output_physiology.activations import (
    HOOK_MODULE_PATH_PHASE2,
    TRANSFORMER_BLOCK_INDEX_PHASE2,
    LayerIndexContract,
    assert_layer_contract,
    mean_pool_tokens,
)


def test_layer_contract_phase2() -> None:
    assert_layer_contract(TRANSFORMER_BLOCK_INDEX_PHASE2, HOOK_MODULE_PATH_PHASE2)
    contract = LayerIndexContract()
    assert contract.hidden_states_tuple_index_if_output_hidden_states() == 13


def test_layer_contract_rejects_mismatch() -> None:
    with pytest.raises(ValueError):
        assert_layer_contract(11, HOOK_MODULE_PATH_PHASE2)
    with pytest.raises(ValueError):
        assert_layer_contract(12, "model.model.layers[11]")


def test_mean_pool_excludes_padding_right() -> None:
    # Two tokens real, one pad on the right
    acts = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool_tokens(acts, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_mean_pool_excludes_padding_left() -> None:
    acts = torch.tensor(
        [
            [[100.0, 100.0], [1.0, 1.0], [3.0, 3.0]],
        ]
    )
    mask = torch.tensor([[0, 1, 1]])
    pooled = mean_pool_tokens(acts, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_mean_pool_numpy_input() -> None:
    acts = np.zeros((2, 3, 4), dtype=np.float32)
    acts[0, 0] = 1
    acts[0, 1] = 3
    mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.float32)
    pooled = mean_pool_tokens(acts, mask)
    assert pooled.shape == (2, 4)
    assert torch.allclose(pooled[0], torch.tensor([2.0, 2.0, 2.0, 2.0]))
