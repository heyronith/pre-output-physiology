"""Phase-24I unit tests: synthetic data only (no GPU, no model, no TEST access)."""

from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pre_output_physiology import phase24h_internal_signal_breadth as H
from pre_output_physiology import phase24i_within_prompt_contrastive as I  # noqa: N812
from pre_output_physiology.phase24e_discovery import SealViolationError, safe_auroc
from pre_output_physiology.phase24g_diagnostics import assert_phase24f_immutable

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "run24i", REPO / "scripts/run_phase24i_within_prompt_contrastive.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run24i"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- synthetic data (32 layers, tiny dim) ----
def make_data(seed: int = 0, per: int = 8, layers: int = 32, dim: int = 6, vocab: int = 10):
    rng = np.random.default_rng(seed)
    tids, pids, labels, n_gen = [], [], [], []
    for p in range(I.N_DEV_PROMPTS):
        for j in range(per):
            tids.append(f"p{p:02d}__r{j}")
            pids.append(f"p{p:02d}")
            labels.append("deceptive" if j % 2 else "honest")
            n_gen.append(int(rng.integers(5, 14)))
    n = len(tids)
    y = np.array([1 if lab == "deceptive" else 0 for lab in labels])
    sig = rng.normal(size=(layers, dim))
    prompt_offset = {
        f"p{p:02d}": rng.normal(size=(layers, dim)) * 2.0 for p in range(I.N_DEV_PROMPTS)
    }
    off = np.stack([prompt_offset[p] for p in pids])
    h0 = (off + rng.normal(size=(n, layers, dim))).astype(np.float32)
    h1 = (0.5 * h0 + rng.normal(size=h0.shape) + 0.8 * y[:, None, None] * sig).astype(
        np.float32
    )
    lg = (rng.normal(size=(n, vocab)) + 0.3 * y[:, None]).astype(np.float32)
    texts = [
        f"prompt {pids[i]} " + ("sorry cannot " if y[i] else "sure here ") + "w1"
        for i in range(n)
    ]
    return H.DevData(
        tids=tids, prompt_ids=pids, labels=labels, n_generated=np.asarray(n_gen),
        texts={1: texts}, logits={1: lg}, h0=h0, h={1: h1},
    )


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def dev_prompts(data):
    return sorted(set(data.prompt_ids))


@pytest.fixture(scope="module")
def synth_folds(dev_prompts):
    # synthetic prompt names differ from real ones: same construction, 7 x 4 partition
    return {s: H.outer_folds(dev_prompts, s) for s in I.OUTER_SALTS}


@pytest.fixture(scope="module")
def fold0(data, dev_prompts, synth_folds):
    tr, ev = I.train_eval_split(synth_folds, dev_prompts, 0, 0)
    return tr, ev, I.run_outer_fold(data, tr, ev, salt=0, fold=0)


# ---- exact Phase-24H fold reuse + hash ----
def test_exact_phase24h_fold_reuse_and_hash() -> None:
    assert I.PHASE24H_FOLDS_SHA256 == (
        "dee54b0f39fc3a7c79afcbe5a90bccf9383e51da1f2ab21cf83b74c2f0d85eb6"
    )
    split = json.loads(
        (REPO / "artifacts/phase24c_design/split_manifest.json").read_text(encoding="utf-8")
    )
    dev = sorted(set(split["train_prompt_ids"]) | set(split["validation_prompt_ids"]))
    folds = I.load_phase24h_folds(REPO, dev)
    assert tuple(sorted(folds)) == (0, 1, 2) and I.OUTER_SALTS == (0, 1, 2)
    for salt in I.OUTER_SALTS:
        assert len(folds[salt]) == 7
        assert all(len(f) == 4 for f in folds[salt])
        # identical to what Phase 24H's own deterministic rule produces
        assert [sorted(f) for f in folds[salt]] == [
            sorted(f) for f in H.outer_folds(dev, salt)
        ]
        for f in range(7):
            tr, ev = I.train_eval_split(folds, dev, salt, f)
            assert len(tr) == 24 and len(ev) == 4 and not set(tr) & set(ev)
    from pre_output_physiology.phase24c_design import sha256_file

    assert sha256_file(str(REPO / I.PHASE24H_FOLDS_REL)) == I.PHASE24H_FOLDS_SHA256


def test_fold_hash_mismatch_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(I, "PHASE24H_FOLDS_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        I.load_phase24h_folds(REPO)


def test_no_new_fold_generation_code() -> None:
    src = inspect.getsource(I)
    assert "outer_fold_key" not in src and "def outer_folds" not in src
    assert "sha256(phase24h|outer" not in src


def test_input_hash_manifest_verification(tmp_path) -> None:
    manifest = json.loads((REPO / I.PHASE24H_INPUT_HASHES_REL).read_text(encoding="utf-8"))
    assert manifest["aggregate_sha256"] == I.PHASE24H_INPUT_AGGREGATE_SHA256
    ok = I.verify_against_phase24h_hashes(dict(manifest["files"]), REPO)
    assert ok["verified"] is True and ok["n_files_now"] == 448
    bad = dict(manifest["files"])
    k = sorted(bad)[0]
    bad[k] = "0" * 64
    res = I.verify_against_phase24h_hashes(bad, REPO)
    assert res["verified"] is False and res["differing_vs_manifest"] == [k]
    short = {k2: v for k2, v in manifest["files"].items() if k2 != k}
    assert I.verify_against_phase24h_hashes(short, REPO)["verified"] is False


# ---- guards ----
def test_test_path_rejection() -> None:
    bad = [
        REPO / "artifacts/phase24f_confirmation/activations_local/LOCKED_TEST/x.npz",
        REPO / "artifacts/phase24d_collection/by_split/LOCKED_TEST/labels_SEALED.json",
        REPO / "artifacts/phase24d_collection/by_split/LOCKED_TEST/x.npz",
        "/tmp/split=test/x.npz",
        REPO / "artifacts/phase24f_confirmation/activations_local/x.npz",
        REPO / "artifacts/phase24d_collection/new_trajectories_SEALED/x.npz",
    ]
    for p in bad:
        with pytest.raises(SealViolationError):
            I.assert_dev_input_path(p)
        with pytest.raises(SealViolationError):
            I.assert_dev_activation_path(p, REPO)
    ok = REPO / H.ACT_DEV_REL / "TRAIN" / "roleplay_003__r00.npz"
    I.assert_dev_activation_path(ok, REPO)
    with pytest.raises(SealViolationError):
        I.assert_dev_activation_path(REPO / H.ACT_DEV_REL / "OTHER" / "a.npz", REPO)
    with pytest.raises(SealViolationError):
        H.load_dev_npz_arrays(bad[0], REPO)


def test_388_honest_deceptive_cohort() -> None:
    mod = _load_script()
    split = json.loads(mod.SPLIT_PATH.read_text(encoding="utf-8"))
    lab_tr, amb_tr = mod._label_map(mod.LABELS_TRAIN, set(split["train_prompt_ids"]))
    lab_va, amb_va = mod._label_map(mod.LABELS_VAL, set(split["validation_prompt_ids"]))
    labels = {**lab_tr, **lab_va}
    assert len(labels) == I.N_EXPECTED_RECORDS == 388
    assert {v["label"] for v in labels.values()} == {"honest", "deceptive"}
    assert amb_tr + amb_va == 60  # ambiguous excluded, never in the cohort
    prompts = {v["prompt_id"] for v in labels.values()}
    assert len(prompts) == 28 and prompts == (
        set(split["train_prompt_ids"]) | set(split["validation_prompt_ids"])
    )
    assert not prompts & set(split["test_prompt_ids"])
    rec = json.loads(
        (REPO / "artifacts/phase24h_internal_signal_breadth/record_index.json").read_text()
    )
    assert sorted(labels) == rec["tids"] and len(rec["tids"]) == 388
    with pytest.raises(ValueError):
        H.DevData(
            tids=["a"], prompt_ids=["p"], labels=["ambiguous"], n_generated=np.array([5]),
            texts={}, logits={}, h0=np.zeros((1, 1, 1)), h={},
        )


# ---- Δh construction ----
def test_delta_is_h1_minus_h0(data) -> None:
    rows = np.arange(0, 40, 3)
    for layer in (0, 7, 31):
        d = I.delta_layer(data, rows, layer)
        np.testing.assert_allclose(d, data.h[1][rows, layer, :] - data.h0[rows, layer, :])
    full = I.delta_all_layers(data, rows)
    assert full.shape == (len(rows), 32, data.h0.shape[2])
    np.testing.assert_allclose(full, data.h[1][rows] - data.h0[rows])
    assert I.LEAD_TIME == 1 and I.LEAD_VARIANT == "DELTA" and I.N_LAYERS == 32
    assert I.LEAD_KEY == "C_MULTILAYER|DELTA|t1"


# ---- same-prompt pairs / orientations / weights ----
def _toy_pairs():
    pids = np.array(["a"] * 5 + ["b"] * 4 + ["c"] * 3)
    y = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0])  # prompt c is single-class
    X = np.arange(len(y) * 3, dtype=float).reshape(len(y), 3) ** 1.1
    return X, y, pids


def test_pairs_are_same_prompt_only() -> None:
    X, y, pids = _toy_pairs()
    pairs = I.build_pair_index(y, pids)
    assert pairs.n_pairs == 2 * 3 + 1 * 3  # a: 2H x 3D ; b: 1H x 3D ; c: none
    assert (pids[pairs.d_idx] == pids[pairs.h_idx]).all()
    assert (pids[pairs.d_idx] == pairs.prompts).all()
    assert (y[pairs.d_idx] == 1).all() and (y[pairs.h_idx] == 0).all()
    assert "c" not in set(pairs.prompts.tolist())
    assert pairs.per_prompt_counts() == {"a": 6, "b": 3}
    assert len({(int(d), int(h)) for d, h in zip(pairs.d_idx, pairs.h_idx, strict=True)}) == 9
    with pytest.raises(ValueError):
        I.pairwise_design(X[pids == "c"], y[pids == "c"], pids[pids == "c"])


def test_both_symmetric_orientations() -> None:
    X, y, pids = _toy_pairs()
    Xp, yp, wp, pairs = I.pairwise_design(X, y, pids)
    P = pairs.n_pairs
    assert Xp.shape == (2 * P, 3) and len(yp) == len(wp) == 2 * P
    np.testing.assert_allclose(Xp[:P], X[pairs.d_idx] - X[pairs.h_idx])
    np.testing.assert_allclose(Xp[P:], X[pairs.h_idx] - X[pairs.d_idx])
    np.testing.assert_allclose(Xp[:P], -Xp[P:])
    assert (yp[:P] == 1).all() and (yp[P:] == 0).all()
    np.testing.assert_allclose(wp[:P], wp[P:])


def test_prompt_equal_and_class_equal_weighting() -> None:
    X, y, pids = _toy_pairs()
    Xp, yp, wp, pairs = I.pairwise_design(X, y, pids)
    pp = np.concatenate([pairs.prompts, pairs.prompts])
    totals = {p: wp[pp == p].sum() for p in ("a", "b")}
    assert totals["a"] == pytest.approx(totals["b"]) == pytest.approx(1.0)
    assert wp[yp == 1].sum() == pytest.approx(wp[yp == 0].sum())
    for p in ("a", "b"):
        assert wp[(pp == p) & (yp == 1)].sum() == pytest.approx(
            wp[(pp == p) & (yp == 0)].sum()
        )
    # large-prompt does not dominate a small prompt despite 2x the pairs
    assert pairs.per_prompt_counts()["a"] == 2 * pairs.per_prompt_counts()["b"]


def test_no_between_prompt_prevalence_signal() -> None:
    # a feature that only encodes prompt prevalence carries NO within-prompt signal
    rng = np.random.default_rng(0)
    pids, y, X = [], [], []
    for p in range(10):
        prev = 0.2 if p % 2 == 0 else 0.8
        for j in range(10):
            pids.append(f"p{p}")
            y.append(int(j < prev * 10))
            X.append([prev + rng.normal(scale=1e-3), rng.normal()])
    X, y, pids = np.asarray(X), np.asarray(y), np.asarray(pids)
    m = I.fit_pairwise(X, y, pids, I.LAYER_C)
    assert abs(m.coef[0]) < 0.5 * max(abs(m.coef[1]), 1e-9) + 0.2
    # constant-within-prompt feature => pair difference exactly 0 => zero weight
    Xc = np.stack([np.repeat(np.arange(10.0), 10), rng.normal(size=100)], axis=1)
    m2 = I.fit_pairwise(Xc, y, pids, I.LAYER_C)
    assert m2.coef[0] == pytest.approx(0.0, abs=1e-9)


def test_pairwise_probe_has_no_intercept_and_recovers_signal() -> None:
    rng = np.random.default_rng(1)
    pids, y, X = [], [], []
    direction = np.array([1.0, -1.0, 0.0, 0.5])
    for p in range(12):
        base = rng.normal(size=4) * 5
        for j in range(8):
            lab = j % 2
            pids.append(f"p{p}")
            y.append(lab)
            X.append(base + lab * direction + rng.normal(scale=0.3, size=4))
    X, y, pids = np.asarray(X), np.asarray(y), np.asarray(pids)
    m = I.fit_pairwise(X, y, pids, I.LAYER_C)
    s = m.score(X)
    for p in set(pids.tolist()):
        idx = pids == p
        assert safe_auroc(y[idx], s[idx]) > 0.9
    assert m.n_prompts == 12


# ---- no eval prompt in any fit ----
def test_no_eval_prompt_in_fitting(data, dev_prompts, synth_folds, monkeypatch) -> None:
    tr, ev = I.train_eval_split(synth_folds, dev_prompts, 1, 2)
    seen: list[tuple[set[str], int, float]] = []
    real = I.fit_pairwise

    def spy(X, y, prompts, C, seed=I.ANALYSIS_SEED):
        seen.append((set(np.asarray(prompts).tolist()), np.asarray(X).shape[1], C))
        return real(X, y, prompts, C, seed)

    monkeypatch.setattr(I, "fit_pairwise", spy)
    out = I.run_outer_fold(data, tr, ev, salt=1, fold=2)
    assert seen
    for prompts, _dim, _c in seen:
        assert prompts <= set(tr) and not prompts & set(ev)
    assert set(np.asarray(data.prompt_ids)[out.eval_rows]) <= set(ev)
    with pytest.raises(AssertionError):
        I.run_outer_fold(data, tr, [*ev, tr[0]])
    monkeypatch.setattr(I, "fit_pairwise", real)
    # corrupting eval labels cannot change any prediction
    d2 = make_data()
    ev_set = set(ev)
    d2.labels = [
        ("honest" if lab == "deceptive" else "deceptive") if d2.prompt_ids[i] in ev_set else lab
        for i, lab in enumerate(d2.labels)
    ]
    out2 = I.run_outer_fold(d2, tr, ev, salt=1, fold=2)
    np.testing.assert_allclose(out.act, out2.act)
    np.testing.assert_allclose(out.surf, out2.surf)
    np.testing.assert_allclose(out.comb, out2.comb)


def test_no_per_eval_prompt_fitting_in_metrics(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("model fit called inside the metric code")

    monkeypatch.setattr(LogisticRegression, "fit", boom)
    preds = _fake_preds(np.random.default_rng(3))
    res = I.full_metrics(preds, n_boot=50, seed=1)
    assert res["primary"]["n_qualifying"] == 12
    for fn in (I.primary_analysis, I.per_prompt_salt_table, I.pooled_descriptive,
               I.salt_prompt_equal_delta, I.compare_with_phase24h):
        assert ".fit(" not in inspect.getsource(fn)


# ---- inner OOF is prompt-held-out ----
def test_inner_oof_prompt_held_out(monkeypatch) -> None:
    rng = np.random.default_rng(2)
    pids, y = [], []
    for p in range(20):
        for j in range(6):
            pids.append(f"p{p:02d}")
            y.append(j % 2)
    pids, y = np.asarray(pids), np.asarray(y)
    X = rng.normal(size=(len(y), 4))
    calls: list[set[str]] = []
    real = I.fit_pairwise

    def spy(Xt, yt, pt, C, seed=I.ANALYSIS_SEED):
        calls.append(set(np.asarray(pt).tolist()))
        return real(Xt, yt, pt, C, seed)

    monkeypatch.setattr(I, "fit_pairwise", spy)
    oof = I.pairwise_oof_scores(X, y, pids, 1.0)
    assert np.isfinite(oof).all()
    allp = set(pids.tolist())
    held = [allp - c for c in calls]
    assert len(held) == I.INNER_FOLDS == 5
    for i, a in enumerate(held):
        assert a
        for b in held[i + 1:]:
            assert not a & b
    assert set().union(*held) == allp
    # each row's score equals the score of the model that never saw its prompt
    monkeypatch.setattr(I, "fit_pairwise", real)
    from pre_output_physiology.phase24c_design import prompt_group_folds

    for hold in prompt_group_folds(sorted(allp), n_folds=5):
        te = np.isin(pids, hold)
        m = real(X[~te], y[~te], pids[~te], 1.0)
        np.testing.assert_allclose(oof[te], m.score(X[te]))


# ---- 32 layers, frozen Cs, no selection ----
def test_32_layers_no_selection_and_frozen_cs(data, dev_prompts, synth_folds, monkeypatch) -> None:
    assert (I.LAYER_C, I.META_C, I.COMBINED_C) == (0.1, 1.0, 10.0)
    tr, ev = I.train_eval_split(synth_folds, dev_prompts, 0, 3)
    calls: list[tuple[int, float]] = []
    real = I.fit_pairwise

    def spy(X, y, prompts, C, seed=I.ANALYSIS_SEED):
        calls.append((np.asarray(X).shape[1], C))
        return real(X, y, prompts, C, seed)

    monkeypatch.setattr(I, "fit_pairwise", spy)
    out = I.run_outer_fold(data, tr, ev, salt=0, fold=3)
    dim = data.h0.shape[2]
    layer_calls = [c for c in calls if c[0] == dim]
    assert len(layer_calls) == 32 * (1 + I.INNER_FOLDS)  # 32 layers x (full + 5 inner)
    assert {c for _, c in layer_calls} == {0.1}
    meta_calls = [c for c in calls if c[0] == 32]
    assert meta_calls and {c for _, c in meta_calls} == {1.0}
    comb_calls = [c for c in calls if c[0] == 2]
    assert len(comb_calls) == 1 and comb_calls[0][1] == 10.0
    assert len(out.info["meta_coef"]) == 32
    assert "none" in out.info["layers_selected"]
    assert out.info["n_layers"] == 32
    src = inspect.getsource(I)
    for banned in ("PCA", "SVC", "select_meta_hp", "argmax", "RAW", "t2", "t4"):
        assert banned not in src.replace("RAW_", ""), banned
    Z_oof, Z_ev = I.layer_scores(data, np.where(np.isin(data.prompt_ids, tr))[0],
                                 np.where(np.isin(data.prompt_ids, ev))[0])
    assert Z_oof.shape[1] == Z_ev.shape[1] == 32


# ---- SURFACE at t=1 reused ----
def test_surface_t1_reuse(data, dev_prompts, synth_folds, monkeypatch) -> None:
    tr, ev = I.train_eval_split(synth_folds, dev_prompts, 0, 1)
    calls = []
    real = H.fit_surface

    def spy(d, t, trr, evr, seed=H.ANALYSIS_SEED):
        calls.append(t)
        return real(d, t, trr, evr, seed)

    monkeypatch.setattr(H, "fit_surface", spy)
    out = I.run_outer_fold(data, tr, ev, salt=0, fold=1)
    assert calls == [1]
    assert (I.SURFACE_C_TEXT, I.SURFACE_C_LOGITS, I.SURFACE_C_META) == (0.01, 0.01, 0.1)
    np.testing.assert_allclose(out.surf, out.surface.ev)
    np.testing.assert_array_equal(out.eval_rows, out.surface.eval_rows)
    assert data.survivors(1)[out.eval_rows].all()
    # supplied cache is used (no refit) and is identical
    calls.clear()
    out2 = I.run_outer_fold(data, tr, ev, salt=0, fold=1, surface=out.surface)
    assert calls == []
    np.testing.assert_allclose(out2.surf, out.surf)
    # stale cache rows are rejected
    bad = H.SurfaceOut(t=1, train_rows=out.surface.train_rows[:-1],
                       eval_rows=out.surface.eval_rows, oof=out.surface.oof,
                       ev=out.surface.ev)
    with pytest.raises(AssertionError):
        I.run_outer_fold(data, tr, ev, salt=0, fold=1, surface=bad)


# ---- fold outputs ----
def test_fold_outputs_and_pair_counts(data, fold0) -> None:
    tr, ev, out = fold0
    n_ev = int(np.isin(data.prompt_ids, ev).sum())
    assert len(out.eval_rows) == len(out.surf) == len(out.act) == len(out.comb) == n_ev
    assert np.isfinite(out.surf).all() and np.isfinite(out.act).all()
    assert np.isfinite(out.comb).all()
    tp = out.info["train_pairs"]
    assert tp["n_pair_bearing_prompts"] == 24 and tp["n_rollouts"] == out.info["n_train_rollouts"]
    assert tp["n_hxd_pairs"] == sum(tp["pairs_per_prompt"].values()) == 24 * 16
    assert tp["n_training_samples_both_orientations"] == 2 * tp["n_hxd_pairs"]
    assert out.info["combined_coef_surface_activation"] and out.info["meta_n_pairs"] > 0


def test_signal_recovery_sanity(data, synth_folds, dev_prompts) -> None:
    # synthetic planted within-prompt signal must be found on held-out prompts
    results = []
    for f in range(I.N_OUTER_FOLDS):
        tr, ev = I.train_eval_split(synth_folds, dev_prompts, 0, f)
        results.append(I.run_outer_fold(data, tr, ev, salt=0, fold=f))
    preds = I.assemble_predictions(data, results)
    assert len(preds[0]["rows"]) == len(set(preds[0]["rows"].tolist())) == data.n
    m = I.full_metrics({0: preds[0]}, n_boot=50, seed=1)
    assert m["primary"]["mean_activation_auroc_descriptive_only"] > 0.8
    assert m["primary"]["n_qualifying"] == 28


# ---- primary estimand ----
def _fake_preds(rng, n_prompts=12, per=6):
    out = {}
    for s in I.OUTER_SALTS:
        pr, y, surf, comb, act = [], [], [], [], []
        for p in range(n_prompts):
            for j in range(per):
                pr.append(f"p{p}")
                y.append(j % 2)
                surf.append(rng.normal())
                comb.append(rng.normal() + 1.0 * (j % 2))
                act.append(rng.normal() + 1.0 * (j % 2))
        out[s] = {
            "rows": np.arange(len(y)), "y": np.array(y), "prompts": np.array(pr),
            "surf": np.array(surf), "act": np.array(act), "comb": np.array(comb),
        }
    return out


def test_primary_is_prompt_equal_within_prompt_delta_not_pooled() -> None:
    # Prompts differ in score LEVEL: pooled AUROC is driven by between-prompt offsets and
    # is very different from the prompt-equal within-prompt mean.
    rng = np.random.default_rng(7)
    per_salt = {}
    for s in I.OUTER_SALTS:
        pr, y, surf, comb, act = [], [], [], [], []
        for p in range(10):
            level = 10.0 * (p % 2)  # deceptive-heavy prompts get higher scores overall
            for j in range(8):
                lab = int(j < (6 if p % 2 else 2))
                pr.append(f"p{p}")
                y.append(lab)
                surf.append(level + rng.normal(scale=0.1))  # pooled looks great, within=chance
                comb.append(level + rng.normal(scale=0.1))
                act.append(rng.normal())
        per_salt[s] = {"rows": np.arange(len(y)), "y": np.array(y), "prompts": np.array(pr),
                       "surf": np.array(surf), "act": np.array(act), "comb": np.array(comb)}
    res = I.primary_analysis(per_salt, n_boot=100, seed=3)
    # manual prompt-equal, salt-averaged within-prompt delta
    manual = []
    for p in range(10):
        ds = []
        for s in I.OUTER_SALTS:
            d = per_salt[s]
            idx = d["prompts"] == f"p{p}"
            ds.append(safe_auroc(d["y"][idx], d["comb"][idx])
                      - safe_auroc(d["y"][idx], d["surf"][idx]))
        manual.append(np.mean(ds))
    assert res["mean_delta"] == pytest.approx(float(np.mean(manual)))
    assert res["median_delta"] == pytest.approx(float(np.median(manual)))
    assert res["fraction_gt_0"] == pytest.approx(float(np.mean(np.asarray(manual) > 0)))
    pooled = I.pooled_descriptive(per_salt)
    assert pooled["0"]["surface_auroc"] > 0.65  # pooled AUROC is level-driven
    assert abs(res["mean_surface_auroc"] - 0.5) < 0.25  # within-prompt is near chance
    assert pooled["0"]["surface_auroc"] - res["mean_surface_auroc"] > 0.1
    # per-prompt equal weight: duplicating a prompt's rollouts does not change its weight
    assert len(res["rows"]) == 10


def test_salt_averaging_one_value_per_prompt() -> None:
    rng = np.random.default_rng(11)
    preds = _fake_preds(rng, n_prompts=5, per=6)
    res = I.primary_analysis(preds, n_boot=20, seed=1)
    for r in res["rows"]:
        assert r["n_salts"] == 3 and set(r["delta_by_salt"]) == {"0", "1", "2"}
        assert r["delta_auroc"] == pytest.approx(np.mean(list(r["delta_by_salt"].values())))
    assert res["mean_delta"] == pytest.approx(np.mean([r["delta_auroc"] for r in res["rows"]]))
    for s in I.OUTER_SALTS:
        assert res["per_salt_mean_delta"][str(s)] == pytest.approx(
            np.mean([r["delta_by_salt"][str(s)] for r in res["rows"]])
        )
        assert I.salt_prompt_equal_delta(preds, s) == pytest.approx(
            res["per_salt_mean_delta"][str(s)]
        )


def test_qualifying_prompt_rule_two_honest_two_deceptive() -> None:
    rng = np.random.default_rng(4)
    preds = _fake_preds(rng, n_prompts=4, per=4)  # 2H + 2D each -> all qualify
    assert I.primary_analysis(preds, n_boot=10, seed=1)["n_qualifying"] == 4
    assert I.qualifies(np.array([0, 0, 1, 1])) and not I.qualifies(np.array([0, 1, 1, 1]))
    assert not I.qualifies(np.array([0, 0, 0, 1])) and not I.qualifies(np.array([0, 0, 0, 0]))
    for s in preds:  # drop one deceptive from p0 -> 2H + 1D -> disqualified
        m = (preds[s]["prompts"] == "p0") & (preds[s]["y"] == 1)
        keep = np.ones(len(m), bool)
        keep[np.where(m)[0][1:]] = False
        preds[s] = {k: v[keep] for k, v in preds[s].items()}
    res = I.primary_analysis(preds, n_boot=10, seed=1)
    assert res["n_qualifying"] == 3
    assert {r["prompt_id"] for r in res["rows"]} == {"p1", "p2", "p3"}
    assert I.MIN_PER_CLASS_WITHIN_PROMPT == 2


def test_bootstrap_is_10000_and_deterministic() -> None:
    assert I.N_BOOTSTRAP == 10000
    preds = _fake_preds(np.random.default_rng(5))
    a = I.primary_analysis(preds, n_boot=I.N_BOOTSTRAP, seed=7)
    b = I.primary_analysis(preds, n_boot=I.N_BOOTSTRAP, seed=7)
    assert a["bootstrap"]["n_reps"] == 10000
    assert a["bootstrap"] == b["bootstrap"]
    lo, hi = a["bootstrap"]["mean_ci95"]
    assert lo < a["mean_delta"] < hi
    c = I.primary_analysis(preds, n_boot=200, seed=8)
    assert c["bootstrap"]["mean_ci95"] != a["bootstrap"]["mean_ci95"]
    assert "I.N_BOOTSTRAP" in inspect.getsource(_load_script().run_full)


# ---- promotion ----
def test_five_part_promotion_rule() -> None:
    base = dict(mean_delta=0.05, ci_low=0.01, per_salt_delta=[0.04, 0.05, 0.06],
                median_delta=0.02, fraction_positive=0.60)
    d = I.promotion_decision(**base)
    assert d["label"] == I.PROMOTED == "promising_for_independent_replication"
    assert d["promoted"] is True and len(d["criteria"]) == 5
    variants = {
        "mean": dict(base, mean_delta=0.0),
        "ci": dict(base, ci_low=0.0),
        "salt": dict(base, per_salt_delta=[0.04, -0.01, 0.06]),
        "salt_count": dict(base, per_salt_delta=[0.04, 0.05]),
        "median": dict(base, median_delta=0.0),
        "frac": dict(base, fraction_positive=0.59),
        "nan": dict(base, mean_delta=float("nan")),
        "nanfrac": dict(base, fraction_positive=float("nan")),
    }
    for name, kw in variants.items():
        r = I.promotion_decision(**kw)
        assert r["promoted"] is False, name
        assert r["label"] == I.NOT_PROMOTED
    strong = _fake_preds(np.random.default_rng(9))
    m = I.full_metrics(strong, n_boot=200, seed=2)
    assert set(m["promotion"]["criteria"]) == {
        "prompt_equal_mean_delta_gt_0", "bootstrap_lower_gt_0", "mean_delta_gt_0_in_all_salts",
        "median_prompt_delta_gt_0", "fraction_positive_prompts_ge_0.60",
    }


def test_shuffle_only_if_promoted_in_runner() -> None:
    src = inspect.getsource(_load_script().run_full)
    assert 'if metrics["promotion"]["promoted"]' in src
    assert I.N_SHUFFLES == 100 and I.SHUFFLE_SALTS == (0,)
    assert I.shuffle_control_verdict(0.2, [0.0] * 50 + [0.05] * 50)["revoked"] is False
    assert I.shuffle_control_verdict(0.05, list(np.linspace(-0.1, 0.1, 100)))["revoked"] is True
    assert I.shuffle_control_verdict(float("nan"), [0.0])["revoked"] is True
    # one-sided p <= 0.05 AND real > q95 are both required
    v = I.shuffle_control_verdict(0.5, [0.0] * 99 + [0.6])
    assert v["n_shuffle_ge_real"] == 1 and v["revoked"] is False
    assert I.shuffle_control_verdict(0.5, [0.6] * 10 + [0.0] * 90)["revoked"] is True


# ---- shuffle control ----
def test_shuffle_preserves_prompt_and_h0_h1_bundles(data) -> None:
    lvl = data.survivors(1).astype(int)
    m1 = I.shuffle_permutation(data.prompt_ids, lvl, 0)
    m2 = I.shuffle_permutation(data.prompt_ids, lvl, 0)
    m3 = I.shuffle_permutation(data.prompt_ids, lvl, 1)
    np.testing.assert_array_equal(m1, m2)
    assert not np.array_equal(m1, m3)
    assert sorted(m1.tolist()) == list(range(data.n))
    pid = np.asarray(data.prompt_ids)
    np.testing.assert_array_equal(pid[m1], pid)  # source rollout is from the same prompt
    assert (m1 != np.arange(data.n)).any()
    rows = np.arange(data.n)
    for layer in (0, 15, 31):
        shuffled = I.delta_layer(data, rows, layer, m1)
        # h0 and h1 move TOGETHER: shuffled Δ is exactly the donor rollout's own Δ
        np.testing.assert_allclose(shuffled, data.h[1][m1, layer] - data.h0[m1, layer])
        np.testing.assert_allclose(shuffled, I.delta_layer(data, m1, layer))
    # labels / TEXT / LOGITS / prompt untouched by the shuffle
    assert data.labels == make_data().labels and data.texts[1] == make_data().texts[1]
    np.testing.assert_array_equal(data.logits[1], make_data().logits[1])


def test_shuffle_run_keeps_surface_and_changes_activation(
    data, dev_prompts, synth_folds, fold0
) -> None:
    tr, ev, real = fold0
    amap = I.shuffle_permutation(data.prompt_ids, data.survivors(1).astype(int), 3)
    shuf = I.run_outer_fold(data, tr, ev, salt=0, fold=0, act_map=amap, surface=real.surface)
    np.testing.assert_allclose(shuf.surf, real.surf)
    assert not np.allclose(shuf.act, real.act)
    ident = I.run_outer_fold(data, tr, ev, salt=0, fold=0, act_map=np.arange(data.n),
                             surface=real.surface)
    np.testing.assert_allclose(ident.comb, real.comb)
    np.testing.assert_allclose(ident.act, real.act)


# ---- determinism / immutability / compute ----
def test_deterministic_seeds(data, dev_prompts, synth_folds, fold0) -> None:
    assert I.derive_seed("shuffle", 1, "p") == I.derive_seed("shuffle", 1, "p")
    assert I.derive_seed("shuffle", 1, "p") != I.derive_seed("shuffle", 2, "p")
    assert I.derive_seed("shuffle", 1, "p") != H.derive_seed("shuffle", 1, "p")
    assert I.ANALYSIS_SEED == H.ANALYSIS_SEED == 2408
    tr, ev, out = fold0
    out2 = I.run_outer_fold(data, tr, ev, salt=0, fold=0)
    np.testing.assert_array_equal(out.comb, out2.comb)
    np.testing.assert_array_equal(out.act, out2.act)


def test_phase24f_immutable() -> None:
    a = assert_phase24f_immutable(REPO)
    assert a["immutable"] is True and a["primary_pass"] is False
    assert I.phase24f_tree_hash(REPO) == I.phase24f_tree_hash(REPO)
    src = inspect.getsource(I)
    assert "write_text" not in src and "write_json" not in src  # library never writes
    assert "phase24f_confirmation" not in inspect.getsource(_load_script().build_records)


def test_no_gpu_or_generation_codepath() -> None:
    forbidden = ("import torch", "cuda", ".generate(", "AutoModel", "import modal", "vllm", '"mps"')
    for p in (
        REPO / "src/pre_output_physiology/phase24i_within_prompt_contrastive.py",
        REPO / "scripts/run_phase24i_within_prompt_contrastive.py",
    ):
        txt = p.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in txt, f"{tok} in {p.name}"
    code = (
        "import sys; import pre_output_physiology.phase24i_within_prompt_contrastive;"
        "assert 'torch' not in sys.modules and 'modal' not in sys.modules"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env={"PYTHONPATH": str(REPO / "src")},
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    txt = (REPO / "scripts/run_phase24i_within_prompt_contrastive.py").read_text()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert k in txt.split("import argparse")[0]  # set BEFORE numpy/sklearn import


def test_config_matches_library_and_status() -> None:
    mod = _load_script()
    cfg = mod.load_config()
    assert mod.config_matches_library(cfg) == []
    assert I.STATUS == "phase24i_within_prompt_contrastive_complete_awaiting_independent_code_audit"
    assert cfg["status_on_completion"] == I.STATUS
    assert cfg["phase24f"]["primary_pass"] is False
    assert cfg["authorizations"]["locked_test_access"] is False
    assert cfg["authorizations"]["gpu_or_model_generation"] is False
    assert cfg["authorizations"]["rerun_phase24h"] is False
    assert cfg["lead"]["phase24h_row"] == "C_MULTILAYER|DELTA|t1"
    assert cfg["compute"]["threads_env"]["OMP_NUM_THREADS"] == 1
    assert cfg["compute"]["stop_if_estimated_runtime_hours_gt"] == 4
    assert I.STARTING_SHA == "baec005ac60208adb13f43e17f9e1b8c915e9af3"


def test_comparison_with_phase24h_is_read_only_and_descriptive() -> None:
    from pre_output_physiology.phase24c_design import sha256_file

    path = REPO / I.PHASE24H_ROW_METRICS_REL
    before = sha256_file(str(path))
    preds = _fake_preds(np.random.default_rng(13), n_prompts=5)
    m = I.full_metrics(preds, n_boot=20, seed=1)
    cmp_ = I.compare_with_phase24h(REPO, m)
    assert cmp_["descriptive_only"] is True
    h = cmp_["phase24h"]
    for k in ("global_pooled_delta_auroc", "within_prompt_mean_delta",
              "within_prompt_median_delta", "within_prompt_fraction_gt_0"):
        assert np.isfinite(h[k]) and k in cmp_["phase24i"]
    assert sha256_file(str(path)) == before
