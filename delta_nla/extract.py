"""Extract pre-block-0 to post-final-block deltas and readout diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from delta_nla.config import atomic_json, load_config, run_dir, seed_everything, write_run_manifest
from delta_nla.data import SPLITS, atomic_write_table, extraction_dir, extraction_shards, fixed_list_array


def _hash_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _choose_split(seed: int, doc_index: int, remaining: dict[str, int]) -> str:
    live = [(name, count) for name, count in remaining.items() if count > 0]
    total = sum(count for _, count in live)
    if total <= 0:
        raise StopIteration
    rng = random.Random(_hash_seed(f"split|{seed}|{doc_index}"))
    needle = rng.randrange(total)
    offset = 0
    for name, count in live:
        offset += count
        if needle < offset:
            return name
    raise AssertionError("weighted split selection fell through")


def _sample_positions(
    token_ids: list[int], *, doc_id: str, seed: int, minimum: int, count: int,
    special_ids: set[int]
) -> list[int]:
    # Require one following token so held-out next-token agreement can also be
    # reported against the observed corpus continuation.
    candidates = [
        p for p in range(minimum - 1, max(minimum - 1, len(token_ids) - 1))
        if token_ids[p] not in special_ids
    ]
    if not candidates:
        return []
    rng = random.Random(_hash_seed(f"position|{seed}|{doc_id}"))
    return sorted(rng.sample(candidates, min(count, len(candidates))))


def _token_record(tokenizer, token_id: int, value_name: str, value: float) -> dict[str, Any]:
    return {
        "token_id": int(token_id),
        "text": tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False),
        "token": tokenizer.convert_ids_to_tokens(int(token_id)),
        value_name: round(float(value), 7),
    }


@torch.no_grad()
def _diagnostics(tokenizer, logits_in: torch.Tensor, logits_out: torch.Tensor, k: int) -> list[str]:
    logits_in = logits_in.float()
    logits_out = logits_out.float()
    logp_in = F.log_softmax(logits_in, dim=-1)
    logp_out = F.log_softmax(logits_out, dim=-1)
    p_in = logp_in.exp()
    p_out = logp_out.exp()
    pin, iin = p_in.topk(k, dim=-1)
    pout, iout = p_out.topk(k, dim=-1)
    change = logits_out - logits_in
    inc, iinc = change.topk(k, dim=-1)
    dec_neg, idec = (-change).topk(k, dim=-1)
    entropy_in = -(p_in * logp_in).sum(dim=-1)
    entropy_out = -(p_out * logp_out).sum(dim=-1)
    margin_in = logits_in.topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)
    margin_out = logits_out.topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)
    records: list[str] = []
    for row in range(logits_in.shape[0]):
        record = {
            "input_probe": {
                "top_candidates": [
                    _token_record(tokenizer, iin[row, j], "probability", pin[row, j])
                    for j in range(k)
                ],
                "entropy": round(float(entropy_in[row]), 6),
                "top1_top2_logit_margin": round(float(margin_in[row]), 6),
            },
            "output_prediction": {
                "top_candidates": [
                    _token_record(tokenizer, iout[row, j], "probability", pout[row, j])
                    for j in range(k)
                ],
                "entropy": round(float(entropy_out[row]), 6),
                "top1_top2_logit_margin": round(float(margin_out[row]), 6),
            },
            "largest_logit_increases": [
                _token_record(tokenizer, iinc[row, j], "change", inc[row, j])
                for j in range(k)
            ],
            "largest_logit_decreases": [
                _token_record(tokenizer, idec[row, j], "change", -dec_neg[row, j])
                for j in range(k)
            ],
            "entropy_change_output_minus_input": round(
                float(entropy_out[row] - entropy_in[row]), 6
            ),
            "margin_change_output_minus_input": round(
                float(margin_out[row] - margin_in[row]), 6
            ),
        }
        records.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    return records


def _schema(width: int) -> pa.Schema:
    vec = pa.list_(pa.float32(), width)
    return pa.schema([
        ("row_id", pa.string()),
        ("split", pa.string()),
        ("doc_id", pa.string()),
        ("doc_index", pa.int64()),
        ("token_position", pa.int64()),
        ("context_tokens", pa.int64()),
        ("context", pa.string()),
        ("next_token_id", pa.int64()),
        ("r0", vec),
        ("delta", vec),
        ("diagnostics", pa.string()),
    ])


def _parquet_files(dataset: str, dataset_config: str | None) -> list[str]:
    files = [
        name for name in list_repo_files(dataset, repo_type="dataset")
        if name.endswith(".parquet")
    ]
    if dataset_config:
        # FineWeb's HF config `sample-10BT` maps to repository path
        # `sample/10BT/`. Preserve a general exact-substring fallback for
        # other parquet-backed corpora while failing loud if ambiguous.
        candidates = {
            dataset_config.strip("/"),
            dataset_config.replace("-", "/", 1).strip("/"),
        }
        matched = [
            name for name in files
            if any(f"/{candidate}/" in f"/{name}" for candidate in candidates)
        ]
        if matched:
            files = matched
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"no parquet data files found for dataset={dataset!r}, config={dataset_config!r}"
        )
    return files


def _document_iterator(
    dataset: str,
    dataset_config: str | None,
    text_column: str,
    start_doc: int,
):
    """Yield indexed documents through synchronous, finite shard downloads.

    `datasets(..., streaming=True)` leaves an HTTP prefetch thread alive when
    we stop at a finite quota and aborts CPython during shutdown. Direct Hub
    downloads complete before Parquet iteration begins and have no dangling
    iterator/thread state. Global row numbering across sorted files makes
    resume deterministic.
    """
    global_index = 0
    for filename in _parquet_files(dataset, dataset_config):
        local = hf_hub_download(dataset, filename, repo_type="dataset")
        parquet = pq.ParquetFile(local)
        file_rows = parquet.metadata.num_rows
        if global_index + file_rows <= start_doc:
            global_index += file_rows
            continue
        if text_column not in parquet.schema_arrow.names:
            raise KeyError(f"{filename} has no text column {text_column!r}")
        for batch in parquet.iter_batches(batch_size=1024, columns=[text_column]):
            texts = batch.column(text_column).to_pylist()
            for text in texts:
                index = global_index
                global_index += 1
                if index < start_doc:
                    continue
                yield index, text


def _existing_progress(root: Path) -> tuple[Counter, int, int]:
    counts: Counter = Counter()
    next_doc = 0
    shards = sorted(root.glob("shard_*.parquet"))
    for shard in shards:
        table = pq.read_table(shard, columns=["split", "doc_index", "row_id"])
        counts.update(table["split"].to_pylist())
        if table.num_rows:
            next_doc = max(next_doc, int(pa.compute.max(table["doc_index"]).as_py()) + 1)
    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        next_doc = max(next_doc, int(state.get("next_doc_index", 0)))
    return counts, next_doc, len(shards)


def _save_projection(model, cfg: dict[str, Any], width: int) -> None:
    root = run_dir(cfg) / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    tensor_path = root / "target_projection.safetensors"
    metadata_path = root / "target_projection.json"
    if tensor_path.exists() and metadata_path.exists():
        return
    norm = model.model.norm
    lm_head = model.get_output_embeddings()
    tensors = {
        "norm_weight": norm.weight.detach().float().cpu().contiguous(),
        "unembedding_weight": lm_head.weight.detach().to(torch.bfloat16).cpu().contiguous(),
    }
    save_file(tensors, str(tensor_path))
    atomic_json(metadata_path, {
        "target_model": cfg["models"]["target"],
        "model_revision": getattr(model.config, "_commit_hash", None),
        "d_model": width,
        "vocab_size": int(lm_head.weight.shape[0]),
        "rms_norm_eps": float(model.config.rms_norm_eps),
        "endpoint_input": "input to decoder block 0",
        "endpoint_output": "output of final decoder block before final RMSNorm",
        "subtraction_dtype": "float32",
    })


def extract(config_path: str) -> None:
    cfg = load_config(config_path)
    write_run_manifest(cfg)
    seed_everything(int(cfg["seed"]))
    root = extraction_dir(cfg["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    counts, next_doc, shard_index = _existing_progress(root)
    quotas = {k: int(v) for k, v in cfg["data"]["quotas"].items()}
    for split in SPLITS:
        if counts[split] > quotas[split]:
            raise RuntimeError(f"existing {split} count exceeds configured quota")
    remaining = {split: quotas[split] - counts[split] for split in SPLITS}
    if not any(remaining.values()):
        print("Extraction already complete.")
        return

    device = torch.device(cfg["runtime"]["device"])
    model_name = cfg["models"]["target"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=cfg["models"]["qwen_revision"]
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=cfg["models"]["qwen_revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device).eval()
    layers = model.model.layers
    width = int(model.config.hidden_size)
    if len(layers) != int(model.config.num_hidden_layers):
        raise RuntimeError("decoder layer count differs from model config")
    _save_projection(model, cfg, width)

    captured: dict[str, torch.Tensor] = {}

    def first_pre(_module, args):
        captured["r0"] = args[0].detach().clone()

    def last_post(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["rL"] = hidden.detach().clone()

    h0 = layers[0].register_forward_pre_hook(first_pre)
    hL = layers[-1].register_forward_hook(last_post)

    dcfg = cfg["data"]
    if dcfg["dataset_split"] != "train":
        raise ValueError("direct FineWeb parquet extraction currently supports split=train")
    iterator = iter(_document_iterator(
        dcfg["dataset"], dcfg.get("dataset_config"), dcfg["text_column"], next_doc
    ))
    special_ids = set(tokenizer.all_special_ids)
    rows: list[dict[str, Any]] = []
    last_processed_doc = next_doc - 1
    pbar = tqdm(total=sum(remaining.values()), initial=0, desc="extracted rows")

    def flush() -> None:
        nonlocal rows, shard_index
        if not rows:
            return
        table = pa.table({
            "row_id": pa.array([r["row_id"] for r in rows], type=pa.string()),
            "split": pa.array([r["split"] for r in rows], type=pa.string()),
            "doc_id": pa.array([r["doc_id"] for r in rows], type=pa.string()),
            "doc_index": pa.array([r["doc_index"] for r in rows], type=pa.int64()),
            "token_position": pa.array([r["token_position"] for r in rows], type=pa.int64()),
            "context_tokens": pa.array([r["context_tokens"] for r in rows], type=pa.int64()),
            "context": pa.array([r["context"] for r in rows], type=pa.string()),
            "next_token_id": pa.array([r["next_token_id"] for r in rows], type=pa.int64()),
            "r0": fixed_list_array(np.stack([r["r0"] for r in rows])),
            "delta": fixed_list_array(np.stack([r["delta"] for r in rows])),
            "diagnostics": pa.array([r["diagnostics"] for r in rows], type=pa.string()),
        }, schema=_schema(width))
        path = root / f"shard_{shard_index:05d}.parquet"
        atomic_write_table(table, path, compression="zstd", compression_level=3)
        shard_index += 1
        rows = []
        atomic_json(root / "state.json", {
            "next_doc_index": last_processed_doc + 1,
            "counts": dict(counts),
            "shards": shard_index,
        })

    try:
        while any(value > 0 for value in remaining.values()):
            documents: list[tuple[int, str, str, str]] = []
            while len(documents) < int(dcfg["extraction_batch_documents"]):
                doc_index, text = next(iterator)
                last_processed_doc = doc_index
                if not isinstance(text, str) or not text.strip():
                    continue
                split = _choose_split(int(cfg["seed"]), doc_index, remaining)
                doc_id = f"{dcfg['dataset']}:{dcfg['dataset_split']}:{doc_index}"
                documents.append((doc_index, doc_id, split, text))

            encoded = tokenizer(
                [d[3] for d in documents],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(dcfg["max_context_tokens"]),
                add_special_tokens=True,
            )
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            positions: list[list[int]] = []
            for i, (_idx, doc_id, split, _text) in enumerate(documents):
                ids = encoded["input_ids"][i, : int(lengths[i])].tolist()
                positions.append(_sample_positions(
                    ids,
                    doc_id=doc_id,
                    seed=int(cfg["seed"]),
                    minimum=int(dcfg["min_context_tokens"]),
                    count=min(int(dcfg["positions_per_document"]), remaining[split]),
                    special_ids=special_ids,
                ))

            if not any(positions):
                continue
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            captured.clear()
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            if set(captured) != {"r0", "rL"}:
                raise RuntimeError("endpoint hooks did not both fire")

            selectors: list[tuple[int, int]] = []
            row_meta: list[tuple[int, str, str, int]] = []
            for batch_index, pos_list in enumerate(positions):
                doc_index, doc_id, split, _text = documents[batch_index]
                allowed = min(len(pos_list), remaining[split])
                for pos in pos_list[:allowed]:
                    selectors.append((batch_index, pos))
                    row_meta.append((doc_index, doc_id, split, pos))
                    remaining[split] -= 1
                    counts[split] += 1
            bidx = torch.tensor([x[0] for x in selectors], device=device)
            pidx = torch.tensor([x[1] for x in selectors], device=device)
            r0 = captured["r0"][bidx, pidx].float()
            rL = captured["rL"][bidx, pidx].float()
            delta = rL - r0  # deliberately FP32
            if not torch.isfinite(delta).all():
                raise RuntimeError("non-finite full-transformer delta")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits_in = model.lm_head(model.model.norm(r0.to(torch.bfloat16)))
                logits_out = model.lm_head(model.model.norm(rL.to(torch.bfloat16)))
            diag = _diagnostics(
                tokenizer, logits_in, logits_out, int(dcfg["top_k_diagnostics"])
            )
            r0_np = r0.cpu().numpy()
            delta_np = delta.cpu().numpy()
            ids_cpu = encoded["input_ids"]
            for j, (doc_index, doc_id, split, pos) in enumerate(row_meta):
                batch_index = selectors[j][0]
                context_ids = ids_cpu[batch_index, : pos + 1].tolist()
                rows.append({
                    "row_id": f"{doc_index}:{pos}",
                    "split": split,
                    "doc_id": doc_id,
                    "doc_index": doc_index,
                    "token_position": pos,
                    "context_tokens": pos + 1,
                    "context": tokenizer.decode(
                        context_ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                    "next_token_id": int(ids_cpu[batch_index, pos + 1]),
                    "r0": r0_np[j],
                    "delta": delta_np[j],
                    "diagnostics": diag[j],
                })
            pbar.update(len(row_meta))
            if len(rows) >= int(dcfg["extraction_shard_rows"]):
                flush()
        flush()
    finally:
        h0.remove()
        hL.remove()
        pbar.close()
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
        del iterator

    atomic_json(root / "complete.json", {
        "counts": dict(counts),
        "quotas": quotas,
        "d_model": width,
        "num_transformer_blocks": len(layers),
        "endpoint": "pre-block-0 to post-final-block before final RMSNorm",
        "subtraction_dtype": "float32",
    })
    if dict(counts) != quotas:
        raise RuntimeError(f"final counts {dict(counts)} do not equal quotas {quotas}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    extract(args.config)


if __name__ == "__main__":
    main()
