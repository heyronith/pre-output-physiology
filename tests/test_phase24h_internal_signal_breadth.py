"""Phase-24H unit tests: synthetic data only (no GPU, no model, no TEST access)."""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from pre_output_physiology import phase24h_internal_signal_breadth as H
from pre_output_physiology.phase24e_discovery import SealViolationError
from pre_output_physiology.phase24g_diagnostics import assert_phase24f_immutable

REPO = Path(__file__).resolve().parents[1]


# ---- synthetic data ----
def make_data(seed: int = 0, per: int = 6, layers: int = 6, dim: int = 8, vocab: int = 12):
    rng = np.random.default_rng(seed)
    tids, pids, labels, n_gen = [], [], [], []
    for p in range(H.N_DEV_PROMPTS):
        for j in range(per):
            tids.append(f"p{p:02d}__r{j}")
            pids.append(f"p{p:02d}")
            labels.append("deceptive" if (j + p) % 2 else "honest")
            n_gen.append(int(rng.integers(5, 14)))
    n = len(tids)
    y = np.array([1 if lab == "deceptive" else 0 for lab in labels])
    n_gen_a = np.asarray(n_gen)
    sig = rng.normal(size=(layers, dim))
    h0 = rng.normal(size=(n, layers, dim)).astype(np.float32)
    h, logits, texts = {}, {}, {}
    for t in H.EARLY_TIMES:
        surv = n_gen_a > t
        ht = (0.5 * h0 + rng.normal(size=h0.shape) + 0.8 * y[:, None, None] * sig).astype(
            np.float32
        )
        ht[~surv] = 0.0
        h[t] = ht
        lg = (rng.normal(size=(n, vocab)) + 0.3 * y[:, None]).astype(np.float32)
        lg[~surv] = 0.0
        logits[t] = lg
        texts[t] = [
            f"prompt {pids[i]} " + ("sorry cannot " if y[i] else "sure here ") + f"w{t}"
            for i in range(n)
        ]
    return H.DevData(
        tids=tids, prompt_ids=pids, labels=labels, n_generated=n_gen_a, texts=texts,
        logits=logits, h0=h0, h=h,
    )


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def dev_prompts(data):
    return sorted(set(data.prompt_ids))


@pytest.fixture(scope="module")
def fold0(data, dev_prompts):
    tr, ev = H.train_eval_split(dev_prompts, 0, 0)
    return tr, ev, H.run_outer_fold(data, tr, ev)


@pytest.fixture(scope="module")
def salt0_folds(data, dev_prompts):
    outs = []
    for f in range(H.N_OUTER_FOLDS):
        tr, ev = H.train_eval_split(dev_prompts, 0, f)
        outs.append((0, f, H.run_outer_fold(data, tr, ev)))
    return outs


# ---- guards ----
def test_test_path_rejection() -> None:
    bad = [
        REPO / "artifacts/phase24f_confirmation/activations_local/LOCKED_TEST/x.npz",
        REPO / "artifacts/phase24d_collection/by_split/LOCKED_TEST/labels_SEALED.json",
        REPO / "artifacts/phase24d_collection/by_split/LOCKED_TEST/x.npz",
        "/tmp/split=test/x.npz",
        REPO / "artifacts/phase24f_confirmation/activations_local/x.npz",
    ]
    for p in bad:
        with pytest.raises(SealViolationError):
            H.assert_dev_input_path(p)
        with pytest.raises(SealViolationError):
            H.assert_dev_activation_path(p, REPO)
        with pytest.raises(SealViolationError):
            H.load_dev_npz_arrays(p, REPO)  # guard fires before any file access
    ok = REPO / H.ACT_DEV_REL / "TRAIN" / "roleplay_003__r00.npz"
    H.assert_dev_activation_path(ok, REPO)
    with pytest.raises(SealViolationError):
        H.assert_dev_activation_path(REPO / H.ACT_DEV_REL / "OTHER" / "a.npz", REPO)
    with pytest.raises(SealViolationError):
        H.assert_split_allowed("test")
    H.assert_split_allowed("train")


def test_exactly_28_dev_prompts() -> None:
    import json

    split = json.loads(
        (REPO / "artifacts/phase24c_design/split_manifest.json").read_text(encoding="utf-8")
    )
    dev = sorted(set(split["train_prompt_ids"]) | set(split["validation_prompt_ids"]))
    assert len(dev) == H.N_DEV_PROMPTS == 28
    H.assert_dev_prompts(dev, split)
    with pytest.raises(SealViolationError):
        H.assert_dev_prompts(dev[:27], split)
    with pytest.raises(SealViolationError):
        H.assert_dev_prompts([*dev[:27], split["test_prompt_ids"][0]], split)


# ---- folds ----
def test_outer_fold_determinism_and_structure(dev_prompts) -> None:
    import hashlib

    for salt in H.OUTER_SALTS:
        f1 = H.outer_folds(dev_prompts, salt)
        f2 = H.outer_folds(list(reversed(dev_prompts)), salt)
        assert f1 == f2
        assert len(f1) == 7 and all(len(f) == 4 for f in f1)
        flat = [p for f in f1 for p in f]
        assert sorted(flat) == dev_prompts
        ordered = sorted(
            dev_prompts,
            key=lambda p: (hashlib.sha256(f"phase24h|outer|{salt}|{p}".encode()).hexdigest(), p),
        )
        for i, p in enumerate(ordered):
            assert p in f1[i % 7]
    assert H.outer_folds(dev_prompts, 0) != H.outer_folds(dev_prompts, 1)
    with pytest.raises(ValueError):
        H.outer_folds(dev_prompts[:27], 0)


def test_every_trajectory_three_heldout_predictions(dev_prompts) -> None:
    for salt in H.OUTER_SALTS:
        seen: list[str] = []
        for f in range(7):
            tr, ev = H.train_eval_split(dev_prompts, salt, f)
            assert len(tr) == 24 and len(ev) == 4
            assert not set(tr) & set(ev)
            seen.extend(ev)
        assert sorted(seen) == dev_prompts  # each prompt held out exactly once per salt


def test_prompt_isolation(data, dev_prompts, fold0) -> None:
    tr, ev, out = fold0
    pid = np.asarray(data.prompt_ids)
    for rp in out.rows.values():
        assert set(pid[rp.eval_rows]) <= set(ev)
        assert not set(pid[rp.eval_rows]) & set(tr)
    with pytest.raises(AssertionError):
        H.run_outer_fold(data, tr, [*ev, tr[0]])
    # eval labels are never seen during fitting: corrupting them cannot change predictions
    d2 = make_data()
    ev_set = set(ev)
    d2.labels = [
        ("honest" if lab == "deceptive" else "deceptive") if d2.prompt_ids[i] in ev_set else lab
        for i, lab in enumerate(d2.labels)
    ]
    out2 = H.run_outer_fold(d2, tr, ev)
    for k, rp in out.rows.items():
        np.testing.assert_allclose(rp.comb, out2.rows[k].comb)
        np.testing.assert_allclose(rp.surf, out2.rows[k].surf)


# ---- raw vs delta ----
def test_raw_vs_delta_construction(data) -> None:
    rows = np.where(data.survivors(4))[0][:5]
    raw = H.get_activations(data, 4, rows, "RAW")
    dlt = H.get_activations(data, 4, rows, "DELTA")
    np.testing.assert_array_equal(raw, data.h[4][rows])
    np.testing.assert_allclose(dlt, data.h[4][rows] - data.h0[rows])
    assert not np.allclose(raw, dlt)
    with pytest.raises(ValueError):
        H.get_activations(data, 4, rows, "OTHER")


# ---- same-time surface comparisons ----
def test_same_time_surface_comparison(data, fold0) -> None:
    _, _, out = fold0
    for key, rp in out.rows.items():
        info = H.parse_row_key(key)
        if info["scope"].startswith("t"):
            t = int(info["scope"][1:])
            assert rp.surface_time == t
        sf = out.surface[rp.surface_time]
        np.testing.assert_array_equal(rp.eval_rows, sf.eval_rows)
        np.testing.assert_allclose(rp.surf, sf.ev)
        assert data.survivors(rp.surface_time)[rp.eval_rows].all()
    # search rows: baseline is SURFACE at the *selected* time
    for v in H.VARIANTS:
        rp = out.rows[H.single_key(v, "search")]
        assert rp.surface_time == rp.meta["selected_time"]


def test_multitime_uses_surface_t8(data, fold0) -> None:
    _, _, out = fold0
    for v in H.VARIANTS:
        rp = out.rows[f"{H.FAM_D}|{v}"]
        assert rp.surface_time == 8
        assert data.survivors(8)[rp.eval_rows].all()
        np.testing.assert_allclose(rp.surf, out.surface[8].ev)
        np.testing.assert_array_equal(rp.eval_rows, out.surface[8].eval_rows)
        assert rp.meta["baseline"] == "SURFACE@t8"
    assert H.required_scope(f"{H.FAM_D}|RAW") == ("RAW", H.EARLY_TIMES)


# ---- train-only scaling / PCA ----
def test_train_only_scaling_and_pca() -> None:
    rng = np.random.default_rng(1)
    Ztr = rng.normal(size=(60, 10))
    y = (rng.random(60) > 0.5).astype(int)
    labs = ["deceptive" if v else "honest" for v in y]
    pids = [f"p{i // 3}" for i in range(60)]
    fm = H.fit_meta("pca_logit", 4, Ztr, y, labs, pids)
    np.testing.assert_allclose(fm.scaler.mean_, Ztr.mean(axis=0))
    assert fm.pca is not None and fm.pca.n_components == 4
    # eval data (very different) cannot change fitted parameters
    Zev = rng.normal(loc=50.0, size=(5, 10))
    mean_before = fm.scaler.mean_.copy()
    comps_before = fm.pca.components_.copy()
    fm.score(Zev)
    np.testing.assert_array_equal(fm.scaler.mean_, mean_before)
    np.testing.assert_array_equal(fm.pca.components_, comps_before)
    fm2 = H.fit_meta("pca_logit", 4, Ztr, y, labs, pids)
    np.testing.assert_allclose(fm.score(Zev), fm2.score(Zev))
    # non-finite training rows are excluded from scaler stats
    Zbad = Ztr.copy()
    Zbad[0, 0] = np.nan
    fm3 = H.fit_meta("logit", 1.0, Zbad, y, labs, pids)
    np.testing.assert_allclose(fm3.scaler.mean_, Ztr[1:].mean(axis=0))
    svm = H.fit_meta("rbf_svm", 1.0, Ztr, y, labs, pids)
    assert svm.clf.gamma == "scale"


# ---- OOF stacking without prompt leakage ----
def test_oof_stacking_no_prompt_leakage(monkeypatch) -> None:
    rng = np.random.default_rng(2)
    n = 80
    pids = [f"p{i // 4:02d}" for i in range(n)]
    y = np.array([i % 2 for i in range(n)])
    labs = ["deceptive" if v else "honest" for v in y]
    Z = rng.normal(size=(n, 5))
    calls: list[set[str]] = []
    real = H.fit_meta

    def spy(kind, hp, Zt, yt, lt, pt, seed=H.ANALYSIS_SEED):
        calls.append(set(pt))
        return real(kind, hp, Zt, yt, lt, pt, seed)

    monkeypatch.setattr(H, "fit_meta", spy)
    oof = H.meta_oof_scores("logit", 1.0, Z, y, labs, pids)
    assert np.isfinite(oof).all()
    allp = set(pids)
    held = [allp - c for c in calls]
    assert len(held) == H.INNER_FOLDS
    for i, a in enumerate(held):
        assert a, "empty held-out fold"
        for b in held[i + 1 :]:
            assert not a & b  # prompts held out exactly once
    assert set().union(*held) == allp
    # a row's OOF score comes from a model that never saw its prompt
    for c in calls:
        assert len(c) < len(allp)


# ---- same-prompt analysis ----
def _fake_preds(rng, n_prompts=12, per=6):
    out = {}
    for s in H.OUTER_SALTS:
        pr, y, surf, comb = [], [], [], []
        for p in range(n_prompts):
            for j in range(per):
                pr.append(f"p{p}")
                y.append(j % 2)
                surf.append(rng.normal())
                comb.append(rng.normal() + 1.0 * (j % 2))
        out[s] = {
            "rows": np.arange(len(y)), "y": np.array(y), "prompts": np.array(pr),
            "surf": np.array(surf), "act": np.array(comb), "comb": np.array(comb),
        }
    return out


def test_no_fitting_per_prompt_in_same_prompt_analysis(monkeypatch) -> None:
    def boom(*a, **k):
        raise AssertionError("model fit called inside same-prompt analysis")

    monkeypatch.setattr(LogisticRegression, "fit", boom)
    monkeypatch.setattr(SVC, "fit", boom)
    preds = _fake_preds(np.random.default_rng(3))
    res = H.within_prompt_analysis(preds, n_boot=50, seed=1)
    assert res["n_qualifying"] == 12
    assert res["median_delta"] > 0
    assert ".fit(" not in inspect.getsource(H.within_prompt_analysis)
    full = H.row_metrics(preds, n_boot=50, seed=1)
    assert full["same_prompt"]["n_qualifying"] == 12


def test_same_prompt_qualification_rule() -> None:
    rng = np.random.default_rng(4)
    preds = _fake_preds(rng, n_prompts=3, per=4)
    # prompt p0: only 1 deceptive -> disqualified
    for s in preds:
        m = (preds[s]["prompts"] == "p0") & (preds[s]["y"] == 1)
        keep = np.ones(len(m), bool)
        keep[np.where(m)[0][1:]] = False
        preds[s] = {k: v[keep] for k, v in preds[s].items()}
    res = H.within_prompt_analysis(preds, n_boot=10, seed=1)
    assert res["n_qualifying"] == 2
    assert {r["prompt_id"] for r in res["rows"]} == {"p1", "p2"}


# ---- promotion ----
def test_promotion_rule() -> None:
    base = dict(
        pooled_delta=0.05, ci_low=0.01, per_salt_delta=[0.04, 0.05, 0.06],
        within_median=0.02, fraction_positive=0.60,
    )
    assert H.promotion_decision(**base)["label"] == H.PROMOTED
    assert H.promotion_decision(**base)["label"] == "promising_for_independent_replication"
    variants = {
        "pooled": dict(base, pooled_delta=0.0),
        "ci": dict(base, ci_low=0.0),
        "salt": dict(base, per_salt_delta=[0.04, -0.01, 0.06]),
        "salt_count": dict(base, per_salt_delta=[0.04, 0.05]),
        "median": dict(base, within_median=0.0),
        "frac": dict(base, fraction_positive=0.59),
        "nan": dict(base, pooled_delta=float("nan")),
        "nanfrac": dict(base, fraction_positive=float("nan")),
    }
    for name, kw in variants.items():
        d = H.promotion_decision(**kw)
        assert d["promoted"] is False, name
        assert d["label"] == H.NOT_PROMOTED


# ---- activation shuffle control ----
def test_shuffle_permutation_properties(data) -> None:
    lvl = data.survival_level()
    m1 = H.shuffle_permutation(data.prompt_ids, lvl, 0)
    m2 = H.shuffle_permutation(data.prompt_ids, lvl, 0)
    m3 = H.shuffle_permutation(data.prompt_ids, lvl, 1)
    np.testing.assert_array_equal(m1, m2)
    assert not np.array_equal(m1, m3)
    assert sorted(m1.tolist()) == list(range(data.n))
    pid = np.asarray(data.prompt_ids)
    np.testing.assert_array_equal(pid[m1], pid)  # prompt-preserving
    np.testing.assert_array_equal(lvl[m1], lvl)  # survival stratum preserved
    assert (m1 != np.arange(data.n)).any()


def test_shuffle_control_runs_and_verdict(data, dev_prompts, fold0) -> None:
    tr, ev, real = fold0
    amap = H.shuffle_permutation(data.prompt_ids, data.survival_level(), 3)
    shuf = H.run_outer_fold(
        data, tr, ev, act_map=amap, variants=("RAW",), times=(1,),
        surface_cache=dict(real.surface),
    )
    key = f"{H.FAM_C}|RAW|t1"
    np.testing.assert_allclose(shuf.rows[key].surf, real.rows[key].surf)  # surface unchanged
    assert not np.allclose(shuf.rows[key].act, real.rows[key].act)
    ident = H.run_outer_fold(
        data, tr, ev, act_map=np.arange(data.n), variants=("RAW",), times=(1,),
        surface_cache=dict(real.surface),
    )
    np.testing.assert_allclose(ident.rows[key].comb, real.rows[key].comb)
    # verdicts
    assert H.shuffle_control_verdict(0.2, [0.0] * 50 + [0.05] * 50)["revoked"] is False
    assert H.shuffle_control_verdict(0.05, list(np.linspace(-0.1, 0.1, 100)))["revoked"] is True
    assert H.shuffle_control_verdict(float("nan"), [0.0])["revoked"] is True
    assert H.N_SHUFFLES == 100


# ---- seeds / determinism ----
def test_deterministic_seeds(data, dev_prompts, fold0) -> None:
    assert H.derive_seed("shuffle", 1, "p") == H.derive_seed("shuffle", 1, "p")
    assert H.derive_seed("shuffle", 1, "p") != H.derive_seed("shuffle", 2, "p")
    tr, ev, out = fold0
    out2 = H.run_outer_fold(data, tr, ev)
    for k, rp in out.rows.items():
        np.testing.assert_array_equal(rp.comb, out2.rows[k].comb)
    preds = _fake_preds(np.random.default_rng(5))
    a = H.prompt_bootstrap_delta(preds, 30, 7)
    b = H.prompt_bootstrap_delta(preds, 30, 7)
    assert a == b


def test_full_salt_assembly_all_rows(data, salt0_folds) -> None:
    preds = H.assemble_predictions(data, salt0_folds)
    assert set(preds) == set(H.row_keys_all())
    assert len(H.row_keys_all()) == 36
    for t in H.EARLY_TIMES:
        key = f"{H.FAM_C}|RAW|t{t}"
        rows = preds[key][0]["rows"]
        assert len(rows) == len(set(rows.tolist())) == int(data.survivors(t).sum())
    rows8 = preds[f"{H.FAM_D}|DELTA"][0]["rows"]
    assert set(rows8.tolist()) == set(np.where(data.survivors(8))[0].tolist())
    m = H.row_metrics({0: preds[f"{H.FAM_C}|RAW|t1"][0]}, n_boot=20, seed=1)
    assert np.isfinite(m["pooled_delta_auroc"])
    assert "activation_auroc_descriptive_only" in m["per_salt"]["0"]


# ---- immutability / no GPU ----
def test_phase24f_immutable() -> None:
    a = assert_phase24f_immutable(REPO)
    assert a["immutable"] is True and a["primary_pass"] is False
    assert H.phase24f_tree_hash(REPO) == H.phase24f_tree_hash(REPO)
    src = inspect.getsource(H)
    assert "write_text" not in src and "write_json" not in src  # library never writes


def test_no_gpu_or_generation_codepath() -> None:
    forbidden = ("import torch", "cuda", ".generate(", "AutoModel", "import modal", "vllm", '"mps"')
    for p in (
        REPO / "src/pre_output_physiology/phase24h_internal_signal_breadth.py",
        REPO / "scripts/run_phase24h_internal_signal_breadth.py",
    ):
        txt = p.read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in txt, f"{tok} in {p.name}"
    code = (
        "import sys; import pre_output_physiology.phase24h_internal_signal_breadth;"
        "assert 'torch' not in sys.modules and 'modal' not in sys.modules"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env={"PYTHONPATH": str(REPO / "src")},
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


def test_config_matches_library_and_status() -> None:
    spec = importlib.util.spec_from_file_location(
        "run24h", REPO / "scripts/run_phase24h_internal_signal_breadth.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run24h"] = mod
    spec.loader.exec_module(mod)
    cfg = mod.load_config()
    assert mod.config_matches_library(cfg) == []
    assert H.STATUS == "phase24h_internal_signal_breadth_complete_awaiting_independent_code_audit"
    assert cfg["phase24f"]["primary_pass"] is False
    assert cfg["authorizations"]["locked_test_access"] is False
    assert cfg["authorizations"]["gpu_or_model_generation"] is False
    assert H.EARLY_TIMES == (1, 2, 4, 8)
