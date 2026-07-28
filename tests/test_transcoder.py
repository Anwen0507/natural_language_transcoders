"""Unit tests for the transcoder-specific code paths — no GPU / model / network.

Covers the genuinely-new logic added for layer-N → layer-M transcoders:
  * stage_pair_target — the keyed (doc_id, n_raw_tokens) join, every input-
    validation assertion, chunked joins with duplicate target keys, the
    full-coverage assertion, empty inputs, and the N→M sidecar provenance.
  * schema.transcoder_gold / transcoder_delta_mode — gold selection + env toggle.
  * schema.load_predict_mean_baselines — the FVE denominator on the GOLD column,
    the delta branch (v_M − v_N), and the autoencoder auto-detect fallback.
  * sidecar target_layer_index — round-trip + old-sidecar back-compat.
  * train_actor's baseline call contract (AST-level — the module itself needs
    miles + GPUs; see tests/test_transcoder_training_paths.py for the rest of
    the training-path wiring).

Runs standalone (`python tests/test_transcoder.py`) or under pytest. Builds its
own tiny parquets in a tmpdir; needs only numpy / pyarrow / torch + nla.
"""
import ast
import contextlib
import io
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nla.datagen.sidecar import (  # noqa: E402
    NLADatasetMeta, NLAExtractionMeta, deserialize_sidecar, read_sidecar,
    serialize_sidecar, write_sidecar,
)
from nla.schema import (  # noqa: E402
    ACTIVATION_COLUMN, TARGET_ACTIVATION_COLUMN, NLATokenMeta,
    compute_predict_mean_baselines, load_predict_mean_baselines,
    transcoder_delta_mode, transcoder_gold,
)
from nla.storage import LocalStorage  # noqa: E402

D = 8
N, M = 4, 8
ST = LocalStorage()
SRC_KEYS = [("d0", 60), ("d0", 70), ("d1", 80), ("d2", 90)]
KEY_IDX = {k: i for i, k in enumerate(SRC_KEYS)}


# Deterministic-but-random (per-key) vectors: distinct AND not parallel, so
# delta (v_M − v_N) differs from absolute (v_M) under L2-normalization.
def v_src(idx):
    return np.random.default_rng(idx).standard_normal(D).astype(np.float32)


def v_tgt(idx):
    return np.random.default_rng(1000 + idx).standard_normal(D).astype(np.float32)


def _fsl(arr):
    return pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1), type=pa.float32()), D)


def _write(path, keys, vecfn, stage, layer, *, norm="none", base_model="test/model",
           d_model_meta=None, rows=None):
    if rows is None:
        rows = (np.stack([vecfn(KEY_IDX.get(k, 900 + i)) for i, k in enumerate(keys)])
                if keys else np.zeros((0, D), np.float32))
    tbl = pa.table({
        "doc_id": pa.array([k[0] for k in keys], type=pa.string()),
        "n_raw_tokens": pa.array([k[1] for k in keys], type=pa.int64()),
        ACTIVATION_COLUMN: _fsl(rows),
    })
    ST.ensure_parent(str(path))
    pq.write_table(tbl, ST.open_write(str(path)))
    ex = NLAExtractionMeta(
        base_model=base_model, d_model=d_model_meta or D, layer_index=layer, norm=norm,
        corpus="x", corpus_slice={"start": 0, "length": len(keys)}, positions_per_doc=3)
    tok = NLATokenMeta(injection_char="<", injection_token_id=1,
                       injection_left_neighbor_id=2, injection_right_neighbor_id=3,
                       critic_suffix_ids=[9])
    write_sidecar(ST, str(path), NLADatasetMeta(
        dataset_id=f"ds_{stage}_L{layer}", stage=stage, row_count=len(keys),
        extraction=ex, tokens=tok,
        prompt_templates={"actor": "a<INJECT>", "critic": "c {explanation}"}))


def _run_pair(src, tgt, out):
    return subprocess.run(
        [sys.executable, "-m", "nla.datagen.stage_pair_target",
         "--input", str(src), "--target-base", str(tgt),
         "--target-layer", str(M), "--output", str(out)],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})


def _pair_inproc(src, tgt, out, target_layer=M):
    """Run stage_pair_target.main() in-process (assertions surface directly,
    and module state like _CHUNK_SIZE can be patched)."""
    import nla.datagen.stage_pair_target as spt
    old_argv = sys.argv
    sys.argv = ["stage_pair_target", "--input", str(src), "--target-base", str(tgt),
                "--target-layer", str(target_layer), "--output", str(out)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spt.main()
    finally:
        sys.argv = old_argv


def _expect_assert(fn, needle):
    try:
        fn()
    except AssertionError as e:
        assert needle in str(e), f"tripped the WRONG assert: wanted {needle!r}, got: {e}"
        return
    raise AssertionError(f"expected AssertionError containing {needle!r}, but call succeeded")


@contextlib.contextmanager
def _delta_env(value):
    """Set/unset NLA_TRANSCODER_DELTA for the duration; always restore."""
    old = os.environ.get("NLA_TRANSCODER_DELTA")
    try:
        if value is None:
            os.environ.pop("NLA_TRANSCODER_DELTA", None)
        else:
            os.environ["NLA_TRANSCODER_DELTA"] = value
        yield
    finally:
        if old is None:
            os.environ.pop("NLA_TRANSCODER_DELTA", None)
        else:
            os.environ["NLA_TRANSCODER_DELTA"] = old


def test_stage_pair_target_join():
    """Keyed join: right target per (doc_id, n_raw_tokens), source untouched, N→M provenance."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src, tgt, out = d / "ar_sft_N.parquet", d / "base_M.parquet", d / "paired.parquet"
        _write(src, SRC_KEYS, v_src, "ar_sft", N)
        # Shuffled order + extra unused keys ⇒ proves keyed (not positional) join.
        _write(tgt, [("d2", 90), ("d1", 80), ("xX", 11), ("d0", 60), ("d0", 70), ("yY", 12)],
               v_tgt, "base", M)
        r = _run_pair(src, tgt, out)
        assert r.returncode == 0, r.stderr
        t = pq.read_table(out)
        assert TARGET_ACTIVATION_COLUMN in t.column_names
        got_src = np.array(t[ACTIVATION_COLUMN].to_pylist(), np.float32)
        got_tgt = np.array(t[TARGET_ACTIVATION_COLUMN].to_pylist(), np.float32)
        keys = list(zip(t["doc_id"].to_pylist(), t["n_raw_tokens"].to_pylist()))
        assert keys == SRC_KEYS, f"row identity/order changed: {keys}"
        for i, k in enumerate(keys):
            assert np.allclose(got_src[i], v_src(KEY_IDX[k])), f"source mutated at {k}"
            assert np.allclose(got_tgt[i], v_tgt(KEY_IDX[k])), f"wrong target paired at {k}"
            assert not np.allclose(got_tgt[i], got_src[i])
        sc = read_sidecar(ST, str(out))
        assert sc.extraction.layer_index == N
        assert sc.extraction.target_layer_index == M


def test_coverage_gap_fails_loud():
    """A missing target position must abort, not silently mis-pair."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src, tgt, out = d / "ar.parquet", d / "base_missing.parquet", d / "x.parquet"
        _write(src, SRC_KEYS, v_src, "ar_sft", N)
        _write(tgt, [k for k in SRC_KEYS if k != ("d2", 90)], v_tgt, "base", M)
        r = _run_pair(src, tgt, out)
        assert r.returncode != 0
        assert "no target-layer vector" in r.stderr, r.stderr[-400:]


def test_transcoder_gold():
    """None→source, absolute→v_M, delta→v_M−v_N; numpy and torch."""
    s, t = v_src(3), v_tgt(3)
    assert np.allclose(transcoder_gold(s, None, False), s)
    assert np.allclose(transcoder_gold(s, None, True), s)
    assert np.allclose(transcoder_gold(s, t, False), t)
    assert np.allclose(transcoder_gold(s, t, True), t - s)
    st, tt = torch.tensor(s), torch.tensor(t)
    assert torch.allclose(transcoder_gold(st, tt, True), tt - st)
    assert torch.allclose(transcoder_gold(st, None, False), st)


def test_predict_mean_baselines_gold_column():
    """FVE denominator uses the gold column: var(v_M) absolute, var(v_M−v_N) delta."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "paired.parquet"
        src = np.stack([v_src(i) for i in range(len(SRC_KEYS))])
        tgt = np.stack([v_tgt(i) for i in range(len(SRC_KEYS))])
        pq.write_table(pa.table({ACTIVATION_COLUMN: _fsl(src),
                                 TARGET_ACTIVATION_COLUMN: _fsl(tgt)}), str(out))
        for scale in (None, float(np.sqrt(D))):
            abs_ref = compute_predict_mean_baselines(torch.tensor(tgt), scale)
            del_ref = compute_predict_mean_baselines(torch.tensor(tgt - src), scale)
            src_ref = compute_predict_mean_baselines(torch.tensor(src), scale)
            assert np.allclose(load_predict_mean_baselines(
                str(out), scale, target_column=TARGET_ACTIVATION_COLUMN, delta=False), abs_ref)
            assert np.allclose(load_predict_mean_baselines(
                str(out), scale, target_column=TARGET_ACTIVATION_COLUMN, delta=True), del_ref)
            assert np.allclose(load_predict_mean_baselines(
                str(out), scale, target_column=None), src_ref)


def test_transcoder_delta_mode_env():
    """Env toggle contract: ONLY the literal string "1" enables delta mode.
    Anything else ("true", "", unset) stays absolute — pinned so a sloppy
    `export NLA_TRANSCODER_DELTA=true` fails loud in review, not silently."""
    for val, expect in [(None, False), ("0", False), ("1", True),
                        ("true", False), ("", False), ("2", False)]:
        with _delta_env(val):
            assert transcoder_delta_mode() is expect, (val, expect)


def test_sidecar_target_layer_roundtrip_and_backcompat():
    """target_layer_index: None and int round-trip; OLD sidecars written before
    the field existed (key absent entirely) still deserialize to None."""
    ex = NLAExtractionMeta(base_model="m", d_model=4, layer_index=2, norm="none",
                           corpus="c", corpus_slice={"start": 0, "length": 1},
                           positions_per_doc=1)
    meta = NLADatasetMeta(dataset_id="x", stage="base", row_count=1, extraction=ex)
    text = serialize_sidecar(meta)
    assert deserialize_sidecar(text).extraction.target_layer_index is None

    meta8 = NLADatasetMeta(dataset_id="x8", stage="base", row_count=1,
                           extraction=replace(ex, target_layer_index=8))
    assert deserialize_sidecar(serialize_sidecar(meta8)).extraction.target_layer_index == 8

    d = yaml.safe_load(text)
    d["extraction"].pop("target_layer_index")  # simulate a pre-transcoder sidecar
    old = deserialize_sidecar(yaml.safe_dump(d))
    assert old.extraction.target_layer_index is None


def test_pair_target_validation_asserts():
    """Every stage_pair_target input-validation assert fires — and fires with
    ITS message (proves each check is reachable and ordered before the join)."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        out = d / "o.parquet"
        ok_src = d / "src.parquet"
        ok_tgt = d / "tgt.parquet"
        _write(ok_src, SRC_KEYS, v_src, "ar_sft", N)
        _write(ok_tgt, SRC_KEYS, v_tgt, "base", M)

        av = d / "av.parquet"
        _write(av, SRC_KEYS, v_src, "av_sft", N)
        _expect_assert(lambda: _pair_inproc(av, ok_tgt, out),
                       "must be an ar_sft or rl parquet")

        normed = d / "normed.parquet"
        _write(normed, SRC_KEYS, v_tgt, "base", M, norm="unit")
        _expect_assert(lambda: _pair_inproc(ok_src, normed, out),
                       "must hold RAW vectors")

        other = d / "other.parquet"
        _write(other, SRC_KEYS, v_tgt, "base", M, base_model="other/model")
        _expect_assert(lambda: _pair_inproc(ok_src, other, out),
                       "base_model mismatch")

        wided = d / "wided.parquet"
        _write(wided, SRC_KEYS, v_tgt, "base", M, d_model_meta=D + 1)
        _expect_assert(lambda: _pair_inproc(ok_src, wided, out),
                       "d_model mismatch")

        _expect_assert(lambda: _pair_inproc(ok_src, ok_tgt, out, target_layer=M + 1),
                       "!= --target-base extraction layer")

        same = d / "sameL.parquet"
        _write(same, SRC_KEYS, v_tgt, "base", N)
        _expect_assert(lambda: _pair_inproc(ok_src, same, out, target_layer=N),
                       "that's an autoencoder")

        gap = d / "gap.parquet"
        _write(gap, [k for k in SRC_KEYS if k != ("d2", 90)], v_tgt, "base", M)
        _expect_assert(lambda: _pair_inproc(ok_src, gap, out),
                       "no target-layer vector")


def test_pair_chunked_join_first_dup_wins():
    """Keyed join across MANY small batches (_CHUNK_SIZE=2): row identity/order
    preserved, a DUPLICATE target key resolves to its FIRST occurrence, unused
    target rows ignored, and provenance (dataset_id / parents / row_count) set."""
    import nla.datagen.stage_pair_target as spt
    keys5 = [("d0", 60), ("d0", 70), ("d1", 80), ("d2", 90), ("d3", 95)]
    idx5 = {k: i for i, k in enumerate(keys5)}
    src_rows = np.stack([v_src(50 + i) for i in range(5)])
    dup_first, dup_second = v_tgt(70), v_tgt(71)
    assert not np.allclose(dup_first, dup_second)
    tgt_keys = [("zz", 1), ("d1", 80), ("d0", 60), ("d3", 95),
                ("d0", 60), ("d2", 90), ("d4", 99), ("d0", 70)]
    tgt_rows = np.stack([
        v_tgt(60), v_tgt(idx5[("d1", 80)]), dup_first, v_tgt(idx5[("d3", 95)]),
        dup_second, v_tgt(idx5[("d2", 90)]), v_tgt(64), v_tgt(idx5[("d0", 70)]),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        src, tgt, out = d / "rl.parquet", d / "b.parquet", d / "o.parquet"
        _write(src, keys5, v_src, "rl", N, rows=src_rows)
        _write(tgt, tgt_keys, v_tgt, "base", M, rows=tgt_rows)
        old_chunk = spt._CHUNK_SIZE
        spt._CHUNK_SIZE = 2
        try:
            _pair_inproc(src, tgt, out)
        finally:
            spt._CHUNK_SIZE = old_chunk
        t = pq.read_table(out)
        keys = list(zip(t["doc_id"].to_pylist(), t["n_raw_tokens"].to_pylist()))
        assert keys == keys5, f"row identity/order changed: {keys}"
        got_src = np.array(t[ACTIVATION_COLUMN].to_pylist(), np.float32)
        got_tgt = np.array(t[TARGET_ACTIVATION_COLUMN].to_pylist(), np.float32)
        assert np.allclose(got_src, src_rows), "source vectors mutated"
        expect = {k: v_tgt(idx5[k]) for k in keys5}
        expect[("d0", 60)] = dup_first
        for i, k in enumerate(keys):
            assert np.allclose(got_tgt[i], expect[k]), f"wrong target paired at {k}"
        sc = read_sidecar(ST, str(out))
        assert sc.row_count == 5
        assert sc.extraction.layer_index == N and sc.extraction.target_layer_index == M
        assert sc.dataset_id == f"ds_rl_L{N}__xcoder_L{N}_to_L{M}"
        assert sc.parent_datasets == [f"ds_rl_L{N}", f"ds_base_L{M}"]
        assert sc.created_by == "nla.datagen.stage_pair_target"


def test_pair_empty_input_ok():
    """0-row input pairs to a valid 0-row output (schema + sidecar intact) —
    exercises the chunks==[] → zeros((0,d)) lookup path."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        src, tgt, out = d / "e.parquet", d / "b.parquet", d / "o.parquet"
        _write(src, [], v_src, "rl", N)
        _write(tgt, SRC_KEYS, v_tgt, "base", M)
        _pair_inproc(src, tgt, out)
        t = pq.read_table(out)
        assert t.num_rows == 0
        assert TARGET_ACTIVATION_COLUMN in t.column_names
        assert read_sidecar(ST, str(out)).row_count == 0


def test_baselines_autodetect_absent_target():
    """train_actor passes target_column/delta UNCONDITIONALLY; on an autoencoder
    parquet (no target column) the loader must fall back to activation_vector
    and ignore delta — bit-identical to the classic call."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "plain.parquet"
        src = np.stack([v_src(i) for i in range(len(SRC_KEYS))])
        pq.write_table(pa.table({ACTIVATION_COLUMN: _fsl(src)}), str(out))
        ref = compute_predict_mean_baselines(torch.tensor(src), None)
        for delta in (False, True):
            got = load_predict_mean_baselines(
                str(out), None, target_column=TARGET_ACTIVATION_COLUMN, delta=delta)
            assert np.allclose(got, ref), f"delta={delta} changed the fallback"


def test_train_actor_baseline_call_wiring():
    """train_actor.py needs miles + GPUs to import, so pin its CALL CONTRACT at
    the AST level: the single load_predict_mean_baselines call site must pass
    target_column=TARGET_ACTIVATION_COLUMN and delta=transcoder_delta_mode().
    The behavior of that argument combination is fully tested above."""
    tree = ast.parse((REPO / "nla" / "train_actor.py").read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and (getattr(n.func, "id", None) == "load_predict_mean_baselines"
                  or getattr(n.func, "attr", None) == "load_predict_mean_baselines")]
    assert len(calls) == 1, f"expected exactly 1 call site, found {len(calls)}"
    kw = {k.arg: k.value for k in calls[0].keywords}
    tc = kw.get("target_column")
    assert isinstance(tc, ast.Name) and tc.id == "TARGET_ACTIVATION_COLUMN", (
        "baseline call must pass target_column=TARGET_ACTIVATION_COLUMN")
    dl = kw.get("delta")
    assert isinstance(dl, ast.Call) and getattr(dl.func, "id", None) == "transcoder_delta_mode", (
        "baseline call must pass delta=transcoder_delta_mode()")


def test_injection_token_cache_wellformed():
    """The committed token cache is auto-rewritten by find_injection_token();
    guard its shape — a malformed entry means silent wrong-position injection."""
    cache = yaml.safe_load(
        (REPO / "nla" / "datagen" / "injection_token_cache.yaml").read_text())
    assert isinstance(cache, dict) and cache, "cache empty or not a mapping"
    for name, entry in cache.items():
        assert isinstance(name, str) and name, f"bad tokenizer key {name!r}"
        assert set(entry) == {"char", "token_id"}, f"{name}: fields {set(entry)}"
        assert isinstance(entry["char"], str) and len(entry["char"]) == 1, (
            f"{name}: char must be a single character, got {entry['char']!r}")
        assert isinstance(entry["token_id"], int) and entry["token_id"] > 0, (
            f"{name}: token_id must be a positive int, got {entry['token_id']!r}")


if __name__ == "__main__":
    tests = [test_stage_pair_target_join, test_coverage_gap_fails_loud,
             test_transcoder_gold, test_predict_mean_baselines_gold_column,
             test_transcoder_delta_mode_env,
             test_sidecar_target_layer_roundtrip_and_backcompat,
             test_pair_target_validation_asserts,
             test_pair_chunked_join_first_dup_wins,
             test_pair_empty_input_ok,
             test_baselines_autodetect_absent_target,
             test_train_actor_baseline_call_wiring,
             test_injection_token_cache_wellformed]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nall {len(tests)} transcoder unit tests passed")
