import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from delta_nla.data import DeltaStatistics, atomic_write_table, fixed_list_array, fixed_list_numpy
from delta_nla.models import inject_vectors
from delta_nla.prompts import parse_explanation


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
