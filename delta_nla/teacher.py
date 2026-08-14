"""Generate diagnostic-grounded SFT explanations with local Gemma 4 31B IT."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.dataset as pads
import torch
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:  # pragma: no cover - compatibility alias in some releases
    from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM

from delta_nla.config import atomic_json, load_config, run_dir, sha256_text
from delta_nla.data import atomic_write_table, extracted_dataset, label_dir, label_shards
from delta_nla.prompts import (
    TEACHER_SYSTEM_PROMPT,
    parse_explanation,
    teacher_prompt,
)


LABEL_SPLITS = ("av_sft", "ar_sft")


def _existing_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for shard in sorted(root.glob("shard_*.parquet")):
        current = set(pads.dataset(str(shard), format="parquet").to_table(columns=["row_id"])["row_id"].to_pylist())
        overlap = ids.intersection(current)
        if overlap:
            raise RuntimeError(f"duplicate teacher row IDs, sample={next(iter(overlap))}")
        ids.update(current)
    return ids


def _rows(run: str | Path, completed: set[str]) -> Iterator[dict[str, Any]]:
    filt = (pads.field("split") == "av_sft") | (pads.field("split") == "ar_sft")
    scanner = extracted_dataset(run).scanner(
        columns=["row_id", "split", "context", "diagnostics"],
        filter=filt,
        batch_size=256,
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            if row["row_id"] not in completed:
                yield row


def _move_inputs(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {k: _move_inputs(v, device) for k, v in value.items()}
    return value


def _encode(processor, messages: list[dict[str, str]], enable_thinking: bool):
    kwargs = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    try:
        return processor.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


@torch.inference_mode()
def _generate_one(model, processor, prompt: str, cfg: dict[str, Any], retry: int) -> str:
    if retry:
        prompt += (
            "\n\nFORMAT CORRECTION: Your prior answer was unusable. Output only "
            "<explanation> followed by 2-3 hyphen bullets and </explanation>."
        )
    messages = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = _encode(processor, messages, bool(cfg["teacher"]["enable_thinking"]))
    # Gemma's multimodal wrapper can expose an early CPU-owned parameter even
    # when the language model is placed on CUDA.  Input IDs must follow the
    # actual token embedding table, not that arbitrary first parameter.
    device = model.get_input_embeddings().weight.device
    inputs = _move_inputs(inputs, device)
    input_ids = inputs["input_ids"]
    generation = {
        "max_new_tokens": int(cfg["teacher"]["max_new_tokens"]),
        "do_sample": bool(cfg["teacher"]["do_sample"]) or retry > 0,
        "use_cache": True,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if generation["do_sample"]:
        generation["temperature"] = max(float(cfg["teacher"]["temperature"]), 0.2)
        generation["top_p"] = 0.95
    output = model.generate(**inputs, **generation)
    new_tokens = output[0, input_ids.shape[-1] :]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def generate_labels(config_path: str, limit: int | None = None) -> None:
    cfg = load_config(config_path)
    root = label_dir(cfg["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    completed = _existing_ids(root)
    expected = int(cfg["data"]["quotas"]["av_sft"]) + int(cfg["data"]["quotas"]["ar_sft"])
    if len(completed) >= expected:
        print("Teacher labeling already complete.")
        return

    model_name = cfg["models"]["teacher"]
    processor = AutoProcessor.from_pretrained(
        model_name, revision=cfg["models"]["teacher_revision"]
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        revision=cfg["models"]["teacher_revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
    ).eval()
    if torch.cuda.max_memory_allocated() > 79 * 1024**3:
        raise RuntimeError("teacher leaves no safe H100 headroom")

    rows_buffer: list[dict[str, Any]] = []
    shard_index = len(label_shards(cfg["run_dir"]))
    attempted = 0
    valid_count = 0
    start = time.monotonic()

    def flush() -> None:
        nonlocal rows_buffer, shard_index
        if not rows_buffer:
            return
        table = pa.Table.from_pylist(rows_buffer, schema=pa.schema([
            ("row_id", pa.string()),
            ("split", pa.string()),
            ("explanation", pa.string()),
            ("raw_output", pa.string()),
            ("valid_format", pa.bool_()),
            ("attempts", pa.int64()),
        ]))
        atomic_write_table(
            table,
            root / f"shard_{shard_index:05d}.parquet",
            compression="zstd",
            compression_level=5,
        )
        shard_index += 1
        rows_buffer = []

    remaining_total = expected - len(completed)
    if limit is not None:
        remaining_total = min(remaining_total, limit)
    pbar = tqdm(total=remaining_total, desc="Gemma teacher labels")
    try:
        for row in _rows(cfg["run_dir"], completed):
            if limit is not None and attempted >= limit:
                break
            diagnostics = json.dumps(
                json.loads(row["diagnostics"]), indent=2, ensure_ascii=False
            )
            prompt = teacher_prompt(row["context"], diagnostics)
            raw = ""
            explanation = ""
            valid = False
            attempts = 0
            for retry in range(int(cfg["teacher"]["max_retries"])):
                attempts += 1
                raw = _generate_one(model, processor, prompt, cfg, retry)
                explanation, valid = parse_explanation(raw)
                if valid:
                    break
            if not valid:
                raise RuntimeError(
                    f"teacher failed the strict explanation format after {attempts} attempts "
                    f"for row_id={row['row_id']}; last output={raw[:500]!r}"
                )
            attempted += 1
            valid_count += int(valid)
            rows_buffer.append({
                "row_id": row["row_id"],
                "split": row["split"],
                "explanation": explanation,
                "raw_output": raw,
                "valid_format": valid,
                "attempts": attempts,
            })
            pbar.update(1)
            if len(rows_buffer) >= int(cfg["teacher"]["shard_rows"]):
                flush()
        flush()
    finally:
        pbar.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.monotonic() - start
    all_ids = _existing_ids(root)
    summary = {
        "completed_rows": len(all_ids),
        "expected_rows": expected,
        "rows_attempted_this_process": attempted,
        "valid_rows_this_process": valid_count,
        "format_valid_rate_this_process": valid_count / max(attempted, 1),
        "elapsed_seconds_this_process": elapsed,
        "rows_per_hour_this_process": attempted / max(elapsed, 1e-6) * 3600,
        "teacher_model": model_name,
        "teacher_prompt_sha256": sha256_text(TEACHER_SYSTEM_PROMPT + teacher_prompt("", "")),
        "enable_thinking": bool(cfg["teacher"]["enable_thinking"]),
    }
    atomic_json(root / "progress.json", summary)
    if len(all_ids) == expected:
        atomic_json(root / "complete.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    generate_labels(args.config, args.limit)


if __name__ == "__main__":
    main()
