import tempfile
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from delta_nla.data import DeltaStatistics, atomic_write_table, fixed_list_array, fixed_list_numpy
from delta_nla.models import inject_vectors
from delta_nla.prompts import parse_explanation
from delta_nla import teacher


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embed


def test_standardization_is_exactly_invertible():
    stats = DeltaStatistics(
        mean=torch.tensor([1.0, -2.0, 0.5]),
        scale=3.25,
        prediction_kl_baseline=1.0,
        count=10,
        d_model=3,
    )
    delta = torch.randn(7, 3)
    restored = stats.unstandardize(stats.standardize(delta))
    assert torch.allclose(delta, restored, atol=1e-6, rtol=1e-6)


def test_injection_is_fixed_identity_coordinates():
    model = TinyModel()
    ids = torch.tensor([[1, 7, 2], [3, 7, 4]])
    vectors = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 0.25, 8.0]])
    embedded = inject_vectors(model, ids, vectors, injection_token_id=7, alpha=0.5)
    assert torch.equal(embedded[:, 1], vectors * 0.5)
    assert torch.equal(embedded[:, 0], model.embed(ids)[:, 0])


def test_injection_rejects_missing_or_duplicate_marker():
    model = TinyModel()
    vectors = torch.randn(2, 4)
    for ids in (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[7, 7], [3, 7]])):
        try:
            inject_vectors(model, ids, vectors, injection_token_id=7, alpha=1.0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid injection marker count was accepted")


def test_fixed_width_parquet_roundtrip():
    value = np.random.default_rng(1).standard_normal((5, 8)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.parquet"
        atomic_write_table(pa.table({"delta": fixed_list_array(value)}), path)
        restored = fixed_list_numpy(pq.read_table(path)["delta"])
    np.testing.assert_array_equal(value, restored)


def test_explanation_contract():
    text, valid = parse_explanation(
        "noise <explanation>\n- strengthens a noun candidate\n- resolves syntax\n</explanation> tail"
    )
    assert valid
    assert text == "- strengthens a noun candidate\n- resolves syntax"
    _, valid = parse_explanation("just a plausible continuation")
    assert not valid


def test_teacher_batch_retries_only_invalid_outputs(monkeypatch):
    calls = []

    def fake_generate(_model, _processor, prompts, _cfg, retry):
        calls.append((list(prompts), retry))
        if retry == 0:
            return [
                "<explanation>\n- first valid point\n- second valid point\n</explanation>",
                "invalid",
            ]
        return [
            "<explanation>\n- corrected first point\n- corrected second point\n</explanation>"
        ]

    monkeypatch.setattr(teacher, "_generate_resilient", fake_generate)
    cfg = {"teacher": {"max_retries": 2}}
    raw, explanations, valid, attempts, sanitized = teacher._label_prompts(
        object(), object(), ["prompt-a", "prompt-b"], cfg
    )

    assert calls == [(["prompt-a", "prompt-b"], 0), (["prompt-b"], 1)]
    assert valid == [True, True]
    assert attempts == [1, 2]
    assert sanitized == [False, False]
    assert explanations[0] == "- first valid point\n- second valid point"
    assert explanations[1] == "- corrected first point\n- corrected second point"
    assert raw[1].startswith("<explanation>")


def test_remote_teacher_runtime_requires_endpoint(monkeypatch):
    monkeypatch.setenv("DELTA_NLA_TEACHER_BACKEND", "openai_compat")
    monkeypatch.delenv("DELTA_NLA_TEACHER_BASE_URL", raising=False)
    cfg = {
        "models": {"teacher": "local", "teacher_revision": "main"},
    }
    try:
        teacher._teacher_runtime(cfg)
    except ValueError as exc:
        assert "BASE_URL" in str(exc)
    else:
        raise AssertionError("remote teacher without an endpoint was accepted")


def test_remote_prompt_forbids_visible_reasoning():
    prompt = teacher._remote_prompt("diagnostics", retry=0)
    assert "diagnostics" in prompt
    assert "Do not reveal reasoning" in prompt
    assert "Begin immediately with <explanation>" in prompt


def test_remote_logit_bias_validation(monkeypatch):
    monkeypatch.setenv(
        "DELTA_NLA_TEACHER_LOGIT_BIAS_JSON", '{"47502": -100, "18927": -75}'
    )
    assert teacher._remote_logit_bias() == {"47502": -100.0, "18927": -75.0}

    monkeypatch.setenv("DELTA_NLA_TEACHER_LOGIT_BIAS_JSON", "[]")
    try:
        teacher._remote_logit_bias()
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("non-object logit bias was accepted")


def test_remote_teacher_retries_forbidden_terms(monkeypatch):
    calls = []

    def fake_generate(_model, _processor, prompts, _cfg, retry):
        calls.append((list(prompts), retry))
        if retry == 0:
            return [
                "<explanation>\n- strengthens the probability of a noun ending next"
                "\n- suppresses an unrelated punctuation continuation now\n</explanation>"
            ]
        return [
            "<explanation>\n- strengthens a likely noun ending as the next token"
            "\n- suppresses an unrelated punctuation continuation now\n</explanation>"
        ]

    monkeypatch.setenv("DELTA_NLA_TEACHER_BACKEND", "openai_compat")
    monkeypatch.setattr(teacher, "_generate_resilient", fake_generate)
    cfg = {"teacher": {"max_retries": 2}}
    _, explanations, valid, attempts, sanitized = teacher._label_prompts(
        object(), object(), ["prompt"], cfg
    )

    assert calls == [(["prompt"], 0), (["prompt"], 1)]
    assert valid == [True]
    assert attempts == [2]
    assert sanitized == [False]
    assert not teacher._has_forbidden_explanation_terms(explanations[0])


def test_remote_teacher_sanitizes_exhausted_content_violation(monkeypatch):
    output = (
        "<explanation>\n"
        "- lowers graduate candidates by decreasing their logit scores substantially now\n"
        "- strengthens of and from as more suitable continuations in context\n"
        "</explanation>"
    )

    def fake_generate(_model, _processor, prompts, _cfg, _retry):
        return [output for _ in prompts]

    monkeypatch.setenv("DELTA_NLA_TEACHER_BACKEND", "openai_compat")
    monkeypatch.setattr(teacher, "_generate_resilient", fake_generate)
    cfg = {"teacher": {"max_retries": 2}}
    raw, explanations, valid, attempts, sanitized = teacher._label_prompts(
        object(), object(), ["prompt"], cfg
    )

    assert "logit scores" in raw[0]
    assert "candidate support" in explanations[0]
    assert valid == [True]
    assert attempts == [2]
    assert sanitized == [True]
    assert not teacher._has_forbidden_explanation_terms(explanations[0])


def test_remote_max_in_flight_uses_rolling_limit(monkeypatch):
    monkeypatch.setenv("DELTA_NLA_TEACHER_MAX_IN_FLIGHT", "48")
    assert teacher._remote_max_in_flight(256) == 48
    assert teacher._remote_max_in_flight(32) == 32


def test_guided_explanation_regex_enforces_bullet_word_count():
    valid = (
        "<explanation>\n"
        "- one two three four five six seven eight nine ten\n"
        "- one two three four five six seven eight nine ten eleven\n"
        "</explanation>"
    )
    too_short = valid.replace(
        "one two three four five six seven eight nine ten\n",
        "one two three four five six seven eight nine\n",
        1,
    )
    assert re.fullmatch(teacher.GUIDED_EXPLANATION_REGEX, valid)
    assert not re.fullmatch(teacher.GUIDED_EXPLANATION_REGEX, too_short)
