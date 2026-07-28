"""Unit tests for the transcoder TRAINING-PATH wiring — no GPU / model / network.

The rollout/training modules import `miles` (and reward.py imports `ray`) —
the GPU training stack, not installable on a CPU CI box. Minimal shims are
installed into sys.modules BEFORE the nla imports — deliberately even if the
real packages exist, so these stay hermetic unit tests of OUR logic:

  * nla.data_source.NLADataSource — paired parquets: target_activation_vector
    read via the numpy fast path (never to_pylist) and stored per-sample in
    metadata; absent for single-layer parquets; <INJECT> substitution intact.
  * nla.rollout.sft_critic.generate_rollout — critic gold = transcoder_gold
    (autoencoder / absolute / delta) from sample metadata; delta re-read from
    the env per rollout.
  * nla.reward._prep_batch — reward gold identical to the training gold in all
    three modes; non-COMPLETED / extraction-miss samples still skipped.
    _lazy_init caches NLA_TRANSCODER_DELTA once.
  * nla.rollout.nla_generate.generate — the actor ALWAYS injects the scaled
    SOURCE v_N; only the stashed critic gold switches per mode; gold stays RAW;
    critic tokens still attached; extraction-miss FAILs after the gold stash.

Together with tests/test_transcoder.py this pins the core transcoder invariant:
every gold-producing site (rollout train-input, reward, FVE baseline) computes
the SAME gold for the same row and mode.

Runs standalone (`python tests/test_transcoder_training_paths.py`) or under
pytest. Needs only numpy / pyarrow / torch / transformers + nla.
"""
import asyncio
import contextlib
import os
import sys
import tempfile
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# --------------------------------------------------------------------------
# miles / ray shims — installed before any nla training-path import.
# --------------------------------------------------------------------------


class _Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class Sample:
    """Field subset of miles.utils.types.Sample that nla code touches."""
    prompt: object = None
    metadata: dict = field(default_factory=dict)
    response: str = ""
    status: _Status = _Status.PENDING
    tokens: list = field(default_factory=list)
    response_length: int = 0
    reward: float = 0.0
    loss_mask: list | None = None
    multimodal_train_inputs: dict | None = None
    group_index: int | None = None
    index: int | None = None

    Status = _Status  # no annotation → not a dataclass field


class RolloutDataSource:
    def __init__(self, args):
        pass

    def load(self, rollout_id=None):
        pass

    def save(self, rollout_id):
        pass


def _unpatched(name):
    def f(*a, **k):
        raise AssertionError(f"{name} must be monkeypatched by the test")
    return f


def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


_miles = _mod("miles")
_miles.rollout = _mod("miles.rollout")
_miles.rollout.data_source = _mod("miles.rollout.data_source",
                                  RolloutDataSource=RolloutDataSource)
_miles.rollout.generate_utils = _mod("miles.rollout.generate_utils")
_miles.rollout.generate_utils.generate_endpoint_utils = _mod(
    "miles.rollout.generate_utils.generate_endpoint_utils",
    compute_request_payload=_unpatched("compute_request_payload"),
    update_sample_from_response=_unpatched("update_sample_from_response"))
_miles.rollout.inference_rollout = _mod("miles.rollout.inference_rollout")
_miles.rollout.inference_rollout.inference_rollout_train = _mod(
    "miles.rollout.inference_rollout.inference_rollout_train",
    get_worker_urls=_unpatched("get_worker_urls"))
_miles.utils = _mod("miles.utils")
_miles.utils.processing_utils = _mod("miles.utils.processing_utils",
                                     load_tokenizer=_unpatched("load_tokenizer"))
_miles.utils.http_utils = _mod("miles.utils.http_utils", post=_unpatched("post"))
_miles.utils.types = _mod("miles.utils.types", Sample=Sample)
_mod("ray")

import nla.data_source as nds  # noqa: E402
import nla.reward as rw  # noqa: E402
import nla.rollout.nla_generate as ng  # noqa: E402
import nla.rollout.sft_critic as sc  # noqa: E402
from nla.schema import (  # noqa: E402
    ACTIVATION_COLUMN, MM_ACTIVATION_KEY, MM_CRITIC_GOLD_KEY,
    MM_CRITIC_TOKENS_KEY, TARGET_ACTIVATION_COLUMN, normalize_activation,
)

# Force the deterministic transport/routing paths regardless of the outer env
# (these are read from os.environ at nla_generate import time).
ng._BF16_B64_EMBEDS = False
ng._BYPASS_ROUTER = False
os.environ.pop("NLA_ROLLOUT_TEXT_DUMP", None)

D = 8


def vec(seed):
    return np.random.default_rng(seed).standard_normal(D).astype(np.float32)


SRC = [vec(i) for i in range(4)]
TGT = [vec(100 + i) for i in range(4)]


def _fsl(arr):
    return pa.FixedSizeListArray.from_arrays(
        pa.array(arr.reshape(-1), type=pa.float32()), D)


@contextlib.contextmanager
def _delta_env(value):
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


# --------------------------------------------------------------------------
# NLADataSource — paired-parquet loading
# --------------------------------------------------------------------------

nds.load_tokenizer = lambda *a, **k: object()
nds.resolve_sidecar_source = lambda **k: "unused-sidecar-source"
nds.load_nla_config = lambda src, tok: SimpleNamespace(injection_char="㊗")


def _ds_args(path):
    return SimpleNamespace(prompt_data=str(path), hf_checkpoint="fake",
                           input_key="prompt", rollout_seed=0,
                           rollout_shuffle=False, n_samples_per_prompt=1)


def test_data_source_paired_parquet():
    """Paired parquet: BOTH wide columns land in metadata as fp32 numpy rows
    (numpy fast path — a to_pylist fallthrough would hand back Python lists),
    row-aligned, with <INJECT> swapped in list prompts and small cols kept."""
    src, tgt = np.stack(SRC[:3]), np.stack(TGT[:3])
    tbl = pa.table({
        "prompt": pa.array([[{"role": "user", "content": f"explain <INJECT> #{i}"}]
                            for i in range(3)]),
        "response": pa.array([f"<explanation>e{i}</explanation>" for i in range(3)]),
        "doc_id": pa.array([f"d{i}" for i in range(3)]),
        "n_raw_tokens": pa.array([60 + i for i in range(3)], type=pa.int64()),
        ACTIVATION_COLUMN: _fsl(src),
        TARGET_ACTIVATION_COLUMN: _fsl(tgt),
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "paired.parquet"
        pq.write_table(tbl, str(path))
        ds = nds.NLADataSource(_ds_args(path))
    assert len(ds.dataset) == 3
    for i, s in enumerate(ds.dataset.samples):
        av = s.metadata["activation_vector"]
        tv = s.metadata[TARGET_ACTIVATION_COLUMN]
        assert isinstance(av, np.ndarray) and av.dtype == np.float32
        assert isinstance(tv, np.ndarray) and tv.dtype == np.float32, (
            "target must come through the numpy path, not to_pylist")
        assert np.allclose(av, SRC[i]) and np.allclose(tv, TGT[i]), f"row {i} misaligned"
        assert s.prompt[0]["content"] == f"explain ㊗ #{i}", "INJECT not substituted"
        assert s.metadata["response"] == f"<explanation>e{i}</explanation>"
        assert s.metadata["doc_id"] == f"d{i}"
        assert s.metadata["n_raw_tokens"] == 60 + i


def test_data_source_plain_parquet_no_target_key():
    """Single-layer parquet (string prompts): metadata must NOT grow a target
    key — its absence is what routes every downstream site to autoencoder gold."""
    src = np.stack(SRC[:2])
    tbl = pa.table({
        "prompt": pa.array([f"critic prompt {i} <summary>" for i in range(2)]),
        ACTIVATION_COLUMN: _fsl(src),
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "plain.parquet"
        pq.write_table(tbl, str(path))
        ds = nds.NLADataSource(_ds_args(path))
    assert len(ds.dataset) == 2
    for i, s in enumerate(ds.dataset.samples):
        assert TARGET_ACTIVATION_COLUMN not in s.metadata
        assert np.allclose(s.metadata["activation_vector"], SRC[i])
        assert s.prompt == f"critic prompt {i} <summary>"


# --------------------------------------------------------------------------
# sft_critic — AR-SFT critic gold
# --------------------------------------------------------------------------

class _CriticTok:
    def __call__(self, prompt, add_special_tokens=None):
        assert add_special_tokens is True  # must match the stage0 extractor
        return {"input_ids": [11, 12, 13]}


class _Buf:
    def __init__(self, groups):
        self.groups = groups

    def get_samples(self, n):
        return self.groups[:n]


def _run_sft_critic(env_val, paired):
    sc._TOKENIZER = _CriticTok()
    sc._SUFFIX_IDS = None
    sc._SUFFIX_CHECKED = False
    meta = {"activation_vector": SRC[0].copy()}
    if paired:
        meta[TARGET_ACTIVATION_COLUMN] = TGT[0].copy()
    s = Sample(prompt="p <summary>", metadata=meta)
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1,
                           hf_checkpoint="unused")
    with _delta_env(env_val):
        out = sc.generate_rollout(args, 0, _Buf([[s]]))
    (got,) = out[0]
    return got


def test_sft_critic_gold_modes():
    """Gold per (env, data) combination; delta env is re-read every rollout.
    Unpaired data under delta=1 must collapse to the plain autoencoder gold."""
    cases = [
        (None, True, TGT[0]),           # absolute (default env)
        ("0", True, TGT[0]),            # absolute (explicit)
        ("1", True, TGT[0] - SRC[0]),   # delta
        (None, False, SRC[0]),          # autoencoder
        ("1", False, SRC[0]),           # delta env, single-layer data → no-op
    ]
    for env_val, paired, expected in cases:
        got = _run_sft_critic(env_val, paired)
        gold = got.multimodal_train_inputs[MM_ACTIVATION_KEY]
        assert gold.shape == (1, D) and gold.dtype == torch.float32
        assert torch.allclose(gold, torch.from_numpy(expected).view(1, -1)), (
            f"wrong gold for env={env_val!r} paired={paired}")
        assert got.tokens == [11, 12, 13]
        assert got.response_length == 0 and got.reward == 0.0 and got.loss_mask == []


# --------------------------------------------------------------------------
# reward — _prep_batch gold + _lazy_init delta caching
# --------------------------------------------------------------------------

class _BatchTok:
    def __call__(self, prompts, add_special_tokens=None, padding=None,
                 return_tensors=None):
        assert add_special_tokens is True and padding is True and return_tensors == "pt"
        n = len(prompts)
        return {"input_ids": torch.arange(n * 3).view(n, 3),
                "attention_mask": torch.ones(n, 3, dtype=torch.long)}


def _reward_samples():
    return [
        Sample(metadata={"activation_vector": SRC[0],
                         TARGET_ACTIVATION_COLUMN: TGT[0]},
               response="<explanation>good</explanation>",
               status=Sample.Status.COMPLETED),
        Sample(metadata={"activation_vector": SRC[1],
                         TARGET_ACTIVATION_COLUMN: TGT[1]},
               response="no tags here",                       # extraction miss
               status=Sample.Status.COMPLETED),
        Sample(metadata={"activation_vector": SRC[2],
                         TARGET_ACTIVATION_COLUMN: TGT[2]},
               response="<explanation>valid but FAILED</explanation>",
               status=Sample.Status.FAILED),                  # trunc-promoted
        Sample(metadata={"activation_vector": SRC[3]},        # unpaired row
               response="<explanation>plain</explanation>",
               status=Sample.Status.COMPLETED),
    ]


def test_reward_prep_batch_gold_modes():
    """Reward gold == training gold for the same rows: v_M absolute, v_M − v_N
    delta, source for unpaired; miss/FAILED rows keep their penalty slots."""
    rw._TOKENIZER = _BatchTok()
    rw._CFG = SimpleNamespace(
        critic_prompt_template="<text>{explanation}</text> <summary>",
        mse_scale=None)
    for delta in (False, True):
        rw._DELTA = delta
        payload, orig_idx = rw._prep_batch(_reward_samples())
        assert orig_idx == [0, 3], "COMPLETED-with-tags filter changed"
        ids, mask, gold = payload
        assert ids.shape == (2, 3) and mask.shape == (2, 3)
        assert gold.dtype == torch.float32 and gold.shape == (2, D)
        exp0 = TGT[0] - SRC[0] if delta else TGT[0]
        assert torch.allclose(gold[0], torch.from_numpy(exp0)), f"delta={delta}"
        assert torch.allclose(gold[1], torch.from_numpy(SRC[3])), (
            "unpaired row must fall back to autoencoder gold")

    empty_payload, empty_idx = rw._prep_batch(
        [Sample(metadata={"activation_vector": SRC[0]}, response="x",
                status=Sample.Status.FAILED)])
    assert empty_payload is None and empty_idx == []


def test_reward_lazy_init_caches_delta_env():
    """_lazy_init reads NLA_TRANSCODER_DELTA once; later env flips don't move it
    (the mode must be frozen per process — reward and rollout read the same
    exported var at startup, which is what keeps them in agreement)."""
    saved = (rw._TOKENIZER, rw._CFG, rw._DELTA, rw.load_tokenizer, rw.load_nla_config)
    try:
        rw._TOKENIZER, rw._CFG, rw._DELTA = None, None, False
        tok = SimpleNamespace(padding_side="left")
        rw.load_tokenizer = lambda *a, **k: tok
        rw.load_nla_config = lambda src, t: SimpleNamespace(
            critic_prompt_template="t {explanation}", mse_scale=None)
        args = SimpleNamespace(nla_critic_sidecar_source="sidecar-dir",
                               critic_load=None)
        with _delta_env("1"):
            rw._lazy_init(args)
        assert rw._DELTA is True
        assert tok.padding_side == "right", "reward tokenizer must right-pad"
        with _delta_env("0"):
            rw._lazy_init(args)   # cached — no re-read
        assert rw._DELTA is True
    finally:
        rw._TOKENIZER, rw._CFG, rw._DELTA, rw.load_tokenizer, rw.load_nla_config = saved


# --------------------------------------------------------------------------
# nla_generate — RL rollout: source injected, gold stashed
# --------------------------------------------------------------------------

VOCAB, LEFT_ID, INJ_ID, RIGHT_ID = 16, 3, 4, 5
INJ_SCALE = 4.0


class _GenTok:
    def apply_chat_template(self, messages, tokenize=None, add_generation_prompt=None):
        assert tokenize is False and add_generation_prompt is True
        return "PROMPT"

    def encode(self, s, add_special_tokens=None):
        assert add_special_tokens is False  # chat template already has BOS
        return [LEFT_ID, INJ_ID, RIGHT_ID, 7]

    def __call__(self, prompt, add_special_tokens=None):
        assert add_special_tokens is True  # raw critic string needs BOS
        return {"input_ids": [21, 22]}


def _run_generate(delta, paired, response_text):
    emb = torch.nn.Embedding(VOCAB, D)
    with torch.no_grad():
        emb.weight.copy_(torch.arange(VOCAB * D, dtype=torch.float32).view(VOCAB, D) / 7.0)
    ng._TOKENIZER = _GenTok()
    ng._CFG = SimpleNamespace(
        injection_scale=INJ_SCALE, injection_token_id=INJ_ID,
        injection_left_neighbor_id=LEFT_ID, injection_right_neighbor_id=RIGHT_ID,
        critic_prompt_template="<text>{explanation}</text>", mse_scale=None)
    ng._EMBED, ng._EMBED_SCALE, ng._DELTA = emb, 1.0, delta

    captured = {}

    def fake_payload(args, input_ids, sampling_params):
        return {"input_ids": list(input_ids),
                "sampling_params": dict(sampling_params)}, None

    async def fake_post(url, payload):
        captured["url"] = url
        captured["payload"] = dict(payload)
        return {"meta_info": {"output_token_logprobs": [[0.0, 1]] * 2}}

    async def fake_update(args, sample, payload, output):
        sample.response = response_text
        sample.status = Sample.Status.COMPLETED

    ng.compute_request_payload = fake_payload
    ng.post = fake_post
    ng.update_sample_from_response = fake_update

    meta = {"activation_vector": SRC[0].copy()}
    if paired:
        meta[TARGET_ACTIVATION_COLUMN] = TGT[0].copy()
    sample = Sample(prompt=[{"role": "user", "content": "x"}], metadata=meta, index=0)
    args = SimpleNamespace(
        sglang_router_ip="h", sglang_router_port=1234,
        sglang_disable_radix_cache=False, rollout_max_context_len=512,
        save=None, _nla_embed_weight=emb.weight.data)  # identity → no reload
    out = asyncio.run(ng.generate(args, sample, {"max_new_tokens": 8}))
    return out, captured, emb


def test_generate_injects_source_stashes_gold():
    """THE transcoder invariant: the actor is always fed the scaled SOURCE v_N;
    only the stashed critic gold switches with mode — and it stays RAW (no
    mse_scale/injection_scale applied)."""
    resp = "<explanation>hi there</explanation>"
    cases = [
        (False, False, SRC[0]),           # autoencoder
        (True, False, SRC[0]),            # delta env on, unpaired → collapse
        (False, True, TGT[0]),            # transcoder absolute
        (True, True, TGT[0] - SRC[0]),    # transcoder delta
    ]
    v_scaled = normalize_activation(
        torch.from_numpy(SRC[0]).view(1, -1), INJ_SCALE)[0].numpy()
    for delta, paired, expected in cases:
        out, cap, emb = _run_generate(delta, paired, resp)
        label = f"delta={delta} paired={paired}"

        # MM_ACTIVATION_KEY feeds the actor's TRAIN-forward injection — it must
        # be the raw SOURCE in every mode, or train-recomputed logprobs diverge
        # from the rollout's. The gold rides in its own slot.
        src = out.multimodal_train_inputs[MM_ACTIVATION_KEY]
        assert src.shape == (1, D) and src.dtype == torch.float32, label
        assert torch.allclose(src, torch.from_numpy(SRC[0]).view(1, -1)), (
            f"train-side injection stash must be the raw SOURCE for {label}")
        gold = out.multimodal_train_inputs[MM_CRITIC_GOLD_KEY]
        assert gold.shape == (1, D) and gold.dtype == torch.float32, label
        assert torch.allclose(gold, torch.from_numpy(expected).view(1, -1)), (
            f"wrong critic gold for {label}")

        embeds = cap["payload"]["input_embeds"]   # np [T, d]
        assert embeds.shape == (4, D), label
        assert np.allclose(embeds[1], v_scaled, atol=1e-6), (
            f"{label}: injected slot must hold the SCALED SOURCE v_N")
        assert not np.allclose(embeds[1], SRC[0]), (
            "injection_scale was not applied to the injected vector")
        for pos, tok_id in [(0, LEFT_ID), (2, RIGHT_ID), (3, 7)]:
            assert np.allclose(embeds[pos], emb.weight[tok_id].detach().numpy()), (
                f"{label}: non-injected position {pos} altered")

        assert "input_ids" not in cap["payload"], "input_ids must be dropped"
        assert cap["url"] == "http://h:1234/generate"
        assert out.status == Sample.Status.COMPLETED
        assert out.multimodal_train_inputs[MM_CRITIC_TOKENS_KEY].tolist() == [21, 22]


def test_generate_extraction_miss_fails_after_gold():
    """No <explanation> tags → sample FAILs, no critic tokens; the gold stash
    still happens first (harmless — the critic-token filter drops the sample)."""
    out, cap, _ = _run_generate(True, True, "no tags at all")
    assert out.status == Sample.Status.FAILED
    gold = out.multimodal_train_inputs[MM_CRITIC_GOLD_KEY]
    assert torch.allclose(gold, torch.from_numpy(TGT[0] - SRC[0]).view(1, -1))
    assert MM_CRITIC_TOKENS_KEY not in out.multimodal_train_inputs


def test_generate_lazy_init_reads_delta_env():
    """nla_generate._lazy_init picks NLA_TRANSCODER_DELTA up from the env (the
    RolloutManager process) and announces delta mode on stdout — the loud
    breadcrumb that tells a run log which gold this rollout produced."""
    import io as _io
    saved = (ng._TOKENIZER, ng._CFG, ng._EMBED, ng._EMBED_SCALE, ng._DELTA,
             ng.load_tokenizer, ng.load_nla_config_from_args,
             ng.load_embedding_only, ng.AutoConfig, ng.resolve_embed_scale)
    try:
        ng._TOKENIZER = None  # force full init
        ng.load_tokenizer = lambda *a, **k: _GenTok()
        cfg = SimpleNamespace(critic_prompt_template="t {explanation}",
                              injection_scale=None, mse_scale=None)
        ng.load_nla_config_from_args = lambda args, tok: (cfg, "sidecar-src")
        ng.load_embedding_only = lambda ckpt, dtype=None: torch.nn.Embedding(4, D)
        ng.AutoConfig = SimpleNamespace(
            from_pretrained=lambda *a, **k: SimpleNamespace(model_type="qwen2"))
        ng.resolve_embed_scale = lambda hf_config: 1.0
        args = SimpleNamespace(hf_checkpoint="fake", partial_rollout=False)
        buf = _io.StringIO()
        with _delta_env("1"), contextlib.redirect_stdout(buf):
            ng._lazy_init(args)
        assert ng._DELTA is True
        assert "transcoder DELTA mode ON" in buf.getvalue()

        ng._TOKENIZER = None
        with _delta_env(None), contextlib.redirect_stdout(_io.StringIO()):
            ng._lazy_init(args)
        assert ng._DELTA is False, "unset env must resolve to absolute mode"
    finally:
        (ng._TOKENIZER, ng._CFG, ng._EMBED, ng._EMBED_SCALE, ng._DELTA,
         ng.load_tokenizer, ng.load_nla_config_from_args,
         ng.load_embedding_only, ng.AutoConfig, ng.resolve_embed_scale) = saved


if __name__ == "__main__":
    tests = [test_data_source_paired_parquet,
             test_data_source_plain_parquet_no_target_key,
             test_sft_critic_gold_modes,
             test_reward_prep_batch_gold_modes,
             test_reward_lazy_init_caches_delta_env,
             test_generate_injects_source_stashes_gold,
             test_generate_extraction_miss_fails_after_gold,
             test_generate_lazy_init_reads_delta_env]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nall {len(tests)} training-path unit tests passed")
