"""Parquet data contracts and zero-Python-object vector loading."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file


SPLITS = ("av_sft", "ar_sft", "rl", "eval")


def extraction_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "data" / "extracted"


def label_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "data" / "teacher_labels"


def extraction_shards(run_dir: str | Path) -> list[Path]:
    return sorted(extraction_dir(run_dir).glob("shard_*.parquet"))


def label_shards(run_dir: str | Path) -> list[Path]:
    return sorted(label_dir(run_dir).glob("shard_*.parquet"))


def fixed_list_array(array: np.ndarray) -> pa.FixedSizeListArray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected rank-2 vectors, got {array.shape}")
    return pa.FixedSizeListArray.from_arrays(
        pa.array(array.reshape(-1), type=pa.float32()), array.shape[1]
    )


def fixed_list_numpy(column: pa.ChunkedArray | pa.Array) -> np.ndarray:
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    flat = column.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    width = column.type.list_size
    # Arrow often exposes a read-only NumPy view. Returning an owned array
    # avoids undefined-behavior warnings when torch.from_numpy wraps it before
    # a device transfer, and makes the mutability contract explicit.
    return flat.reshape(len(column), width).copy()


def atomic_write_table(table: pa.Table, path: str | Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, **kwargs)
    os.replace(tmp, path)


def extracted_dataset(run_dir: str | Path) -> pads.Dataset:
    shards = extraction_shards(run_dir)
    if not shards:
        raise FileNotFoundError(f"no extraction shards under {extraction_dir(run_dir)}")
    return pads.dataset([str(p) for p in shards], format="parquet")


def iter_split_batches(
    run_dir: str | Path,
    split: str,
    columns: list[str],
    batch_size: int = 4096,
) -> Iterator[pa.RecordBatch]:
    if split not in SPLITS:
        raise ValueError(split)
    scanner = extracted_dataset(run_dir).scanner(
        columns=columns,
        filter=pads.field("split") == split,
        batch_size=batch_size,
    )
    yield from scanner.to_batches()


def split_row_count(run_dir: str | Path, split: str) -> int:
    return extracted_dataset(run_dir).count_rows(filter=pads.field("split") == split)


def load_split_vectors(
    run_dir: str | Path,
    split: str,
    *,
    include_context: bool = False,
    limit: int | None = None,
) -> dict[str, object]:
    columns = ["row_id", "r0", "delta"]
    if include_context:
        columns.extend(["context", "diagnostics", "next_token_id"])
    table = extracted_dataset(run_dir).to_table(
        columns=columns, filter=pads.field("split") == split
    )
    if limit is not None:
        table = table.slice(0, limit)
    result: dict[str, object] = {
        "row_id": np.asarray(table["row_id"].to_pylist(), dtype=object),
        "r0": fixed_list_numpy(table["r0"]),
        "delta": fixed_list_numpy(table["delta"]),
    }
    if include_context:
        result["context"] = table["context"].to_pylist()
        result["diagnostics"] = table["diagnostics"].to_pylist()
        result["next_token_id"] = np.asarray(table["next_token_id"].to_numpy())
    return result


def load_teacher_labels(
    run_dir: str | Path, split: str | None = None, *, valid_only: bool = True
) -> dict[str, str]:
    shards = label_shards(run_dir)
    if not shards:
        raise FileNotFoundError(f"no teacher label shards under {label_dir(run_dir)}")
    dataset = pads.dataset([str(p) for p in shards], format="parquet")
    filt = pads.field("split") == split if split else None
    columns = ["row_id", "explanation"]
    if "valid_format" in dataset.schema.names:
        columns.append("valid_format")
        if valid_only:
            valid_filter = pads.field("valid_format") == True  # noqa: E712
            filt = valid_filter if filt is None else filt & valid_filter
    table = dataset.to_table(columns=columns, filter=filt)
    ids = table["row_id"].to_pylist()
    explanations = table["explanation"].to_pylist()
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate row_id in teacher labels")
    return dict(zip(ids, explanations, strict=True))


@dataclass(frozen=True)
class DeltaStatistics:
    mean: torch.Tensor
    scale: float
    prediction_kl_baseline: float
    count: int
    d_model: int

    @classmethod
    def load(cls, run_dir: str | Path, device: str | torch.device = "cpu") -> "DeltaStatistics":
        root = Path(run_dir) / "artifacts"
        metadata = json.loads((root / "delta_stats.json").read_text())
        mean = load_file(str(root / "delta_mean.safetensors"))["mean"].to(device)
        if mean.numel() != metadata["d_model"]:
            raise RuntimeError("delta mean width does not match metadata")
        return cls(
            mean=mean,
            scale=float(metadata["scale"]),
            prediction_kl_baseline=float(metadata["prediction_kl_baseline"]),
            count=int(metadata["count"]),
            d_model=int(metadata["d_model"]),
        )

    def standardize(self, delta: torch.Tensor) -> torch.Tensor:
        return (delta - self.mean.to(delta.device, delta.dtype)) / self.scale

    def unstandardize(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean.to(x.device, x.dtype) + self.scale * x
