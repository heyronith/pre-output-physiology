"""Group-level train/validation split safeguards."""

from __future__ import annotations

from sklearn.model_selection import GroupShuffleSplit

from pre_output_physiology.provenance import sha256_text


def test_group_shuffle_split_no_prompt_overlap() -> None:
    prompts = [f"prompt-{i // 3}" for i in range(30)]
    hashes = [sha256_text(p) for p in prompts]
    y = [i % 2 for i in range(30)]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(prompts, y, hashes))
    train_g = {hashes[i] for i in train_idx}
    val_g = {hashes[i] for i in val_idx}
    assert train_g.isdisjoint(val_g)
    assert len(train_g) + len(val_g) == len(set(hashes))


def test_group_shuffle_split_seed42_is_deterministic() -> None:
    groups = [f"g{i // 2}" for i in range(40)]
    y = [0] * 20 + [1] * 20
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    a_tr, a_va = next(gss.split(range(40), y, groups))
    b_tr, b_va = next(gss.split(range(40), y, groups))
    assert list(a_tr) == list(b_tr)
    assert list(a_va) == list(b_va)
