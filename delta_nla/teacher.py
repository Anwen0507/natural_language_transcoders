"""Generate diagnostic-grounded SFT explanations with a local teacher."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import itertools
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterator

import httpx
import pyarrow as pa
import pyarrow.dataset as pads
import torch
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:  # pragma: no cover - compatibility alias in some releases
    from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM

from delta_nla.config import atomic_json, git_revision, load_config, sha256_text
from delta_nla.data import atomic_write_table, extracted_dataset, label_dir, label_shards
from delta_nla.prompts import (
    TEACHER_SYSTEM_PROMPT,
    parse_explanation,
    teacher_prompt,
)


LABEL_SPLITS = ("av_sft", "ar_sft")
_GUIDED_BULLET_REGEX = r"- (?:[^<>\s]+[ \t]+){9,24}[^<>\s]+"
GUIDED_EXPLANATION_REGEX = (
    r"<explanation>\n"
    + _GUIDED_BULLET_REGEX
    + "\n"
    + _GUIDED_BULLET_REGEX
    + "(?:\n"
    + _GUIDED_BULLET_REGEX
    + ")?"
    r"\n</explanation>"
)
_FORBIDDEN_EXPLANATION_TERMS = re.compile(
    r"\b(?:logits?|probabilit(?:y|ies)|entropy|vectors?|probes?|diagnostics?)\b",
    flags=re.IGNORECASE,
)


def _teacher_runtime(cfg: dict[str, Any]) -> dict[str, str]:
    """Resolve optional serving overrides without changing the locked run config."""
    backend = os.environ.get("DELTA_NLA_TEACHER_BACKEND", "transformers")
    model = os.environ.get("DELTA_NLA_TEACHER_MODEL", cfg["models"]["teacher"])
    revision = os.environ.get(
        "DELTA_NLA_TEACHER_REVISION", cfg["models"]["teacher_revision"]
    )
    base_url = os.environ.get("DELTA_NLA_TEACHER_BASE_URL", "").rstrip("/")
    if backend == "openai_compat" and not base_url:
        raise ValueError(
            "DELTA_NLA_TEACHER_BASE_URL is required for openai_compat"
        )
    if backend not in {"transformers", "openai_compat"}:
        raise ValueError(f"unsupported teacher backend {backend!r}")
    return {
        "backend": backend,
        "model": model,
        "revision": revision,
        "base_url": base_url,
    }


def _remote_prompt(prompt: str, retry: int) -> str:
    correction = (
        "\n\nFORMAT OR CONTENT CORRECTION: Your prior answer was unusable. "
        "Output only <explanation> followed by 2-3 hyphen bullets and "
        "</explanation>, and paraphrase any forbidden technical terms."
    )
    return (
        f"{TEACHER_SYSTEM_PROMPT}\n\n{prompt}"
        f"{correction if retry else ''}\n\n"
        "This is a direct, short labeling task. Do not reveal reasoning or emit "
        "<think> tags. Begin immediately with <explanation>. Each bullet must "
        "contain 10-25 whitespace-delimited words. Never use the terms logits, "
        "probability, probabilities, entropy, vectors, probes, or diagnostics. "
        "Prioritize concrete strengthened and suppressed token candidates; avoid "
        "generic claims about diversity when concrete candidates are available."
    )


def _has_forbidden_explanation_terms(explanation: str) -> bool:
    """Return whether a label violates the teacher's plain-language contract."""
    return _FORBIDDEN_EXPLANATION_TERMS.search(explanation) is not None


def _remote_logit_bias() -> dict[str, float]:
    """Parse an optional model-specific token ban for the serving backend."""
    raw = os.environ.get("DELTA_NLA_TEACHER_LOGIT_BIAS_JSON", "").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, Mapping):
        raise ValueError("DELTA_NLA_TEACHER_LOGIT_BIAS_JSON must be a JSON object")
    return {str(int(token_id)): float(bias) for token_id, bias in parsed.items()}


def _generate_openai_batch(
    prompts: Sequence[str], cfg: dict[str, Any], retry: int
) -> list[str]:
    """Generate concurrent requests so an OpenAI-compatible server can batch."""
    runtime = _teacher_runtime(cfg)
    timeout_seconds = float(
        os.environ.get("DELTA_NLA_TEACHER_REQUEST_TIMEOUT", "600")
    )
    guided = os.environ.get("DELTA_NLA_TEACHER_GUIDED_REGEX", "0") == "1"
    do_sample = bool(cfg["teacher"]["do_sample"]) or retry > 0
    logit_bias = _remote_logit_bias()

    def payload(prompt: str) -> dict[str, Any]:
        value: dict[str, Any] = {
            "model": runtime["model"],
            "messages": [{"role": "user", "content": _remote_prompt(prompt, retry)}],
            "max_tokens": int(cfg["teacher"]["max_new_tokens"]),
            "temperature": (
                max(float(cfg["teacher"]["temperature"]), 0.2)
                if do_sample
                else 0.0
            ),
            "seed": int(cfg["seed"]),
        }
        if do_sample:
            value["top_p"] = 0.95
        if guided:
            # vLLM <=0.11 accepts this top-level OpenAI extension.
            value["guided_regex"] = GUIDED_EXPLANATION_REGEX
        if logit_bias:
            value["logit_bias"] = logit_bias
        return value

    limits = httpx.Limits(
        max_connections=max(len(prompts), 1),
        max_keepalive_connections=max(len(prompts), 1),
    )
    timeout = httpx.Timeout(timeout_seconds, connect=30.0)
    with httpx.Client(timeout=timeout, limits=limits) as client:

        def generate_one(prompt: str) -> str:
            response = client.post(
                f"{runtime['base_url']}/chat/completions",
                headers={"Authorization": "Bearer EMPTY"},
                json=payload(prompt),
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"teacher server returned {response.status_code}: "
                    f"{response.text[:1000]}"
                ) from exc
            body = response.json()
            return str(body["choices"][0]["message"]["content"] or "").strip()

        with ThreadPoolExecutor(max_workers=max(len(prompts), 1)) as executor:
            return list(executor.map(generate_one, prompts))


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


def _encode(
    processor,
    conversations: list[list[dict[str, str]]],
    enable_thinking: bool,
    *,
    padding: bool,
):
    kwargs = dict(
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": padding},
    )
    try:
        return processor.apply_chat_template(
            conversations, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        return processor.apply_chat_template(conversations, **kwargs)


@torch.inference_mode()
def _generate_batch(
    model,
    processor,
    prompts: Sequence[str],
    cfg: dict[str, Any],
    retry: int,
) -> list[str]:
    """Generate one greedy/sampled response per prompt in a padded batch."""
    if not prompts:
        return []
    if _teacher_runtime(cfg)["backend"] == "openai_compat":
        return _generate_openai_batch(prompts, cfg, retry)
    correction = (
        "\n\nFORMAT CORRECTION: Your prior answer was unusable. Output only "
        "<explanation> followed by 2-3 hyphen bullets and </explanation>."
    )
    conversations = [
        [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt + (correction if retry else "")},
        ]
        for prompt in prompts
    ]
    inputs = _encode(
        processor,
        conversations,
        bool(cfg["teacher"]["enable_thinking"]),
        padding=len(prompts) > 1,
    )
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
    new_tokens = output[:, input_ids.shape[-1] :]
    return [
        text.strip()
        for text in processor.batch_decode(new_tokens, skip_special_tokens=True)
    ]


def _generate_resilient(
    model,
    processor,
    prompts: Sequence[str],
    cfg: dict[str, Any],
    retry: int,
) -> list[str]:
    """Split an oversized batch on CUDA OOM instead of losing the run."""
    try:
        return _generate_batch(model, processor, prompts, cfg, retry)
    except torch.OutOfMemoryError:
        if len(prompts) == 1:
            raise
    # Leave the exception handler before retrying so its traceback no longer
    # retains tensors from the failed generation call.
    gc.collect()
    torch.cuda.empty_cache()
    midpoint = len(prompts) // 2
    print(
        f"teacher batch of {len(prompts)} hit CUDA OOM; retrying as "
        f"{midpoint}+{len(prompts) - midpoint}",
        file=sys.stderr,
        flush=True,
    )
    return _generate_resilient(
        model, processor, prompts[:midpoint], cfg, retry
    ) + _generate_resilient(model, processor, prompts[midpoint:], cfg, retry)


def _label_prompts(
    model,
    processor,
    prompts: Sequence[str],
    cfg: dict[str, Any],
) -> tuple[list[str], list[str], list[bool], list[int]]:
    """Generate a batch, retrying only responses that violate the contract."""
    raw = [""] * len(prompts)
    explanations = [""] * len(prompts)
    valid = [False] * len(prompts)
    attempts = [0] * len(prompts)
    for retry in range(int(cfg["teacher"]["max_retries"])):
        pending = [index for index, is_valid in enumerate(valid) if not is_valid]
        if not pending:
            break
        outputs = _generate_resilient(
            model,
            processor,
            [prompts[index] for index in pending],
            cfg,
            retry,
        )
        if len(outputs) != len(pending):
            raise RuntimeError(
                f"teacher returned {len(outputs)} outputs for {len(pending)} prompts"
            )
        for index, output in zip(pending, outputs, strict=True):
            attempts[index] += 1
            raw[index] = output
            explanations[index], valid[index] = parse_explanation(output)
            if (
                valid[index]
                and os.environ.get("DELTA_NLA_TEACHER_BACKEND", "transformers")
                == "openai_compat"
                and _has_forbidden_explanation_terms(explanations[index])
            ):
                valid[index] = False
    return raw, explanations, valid, attempts


def _load_teacher(cfg: dict[str, Any]):
    runtime = _teacher_runtime(cfg)
    if runtime["backend"] == "openai_compat":
        return None, None
    model_name = runtime["model"]
    processor = AutoProcessor.from_pretrained(
        model_name, revision=runtime["revision"]
    )
    # Decoder-only generation must use left padding so the final real token is
    # aligned across examples and receives the next-token prediction.
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        revision=runtime["revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map={"": 0},
    ).eval()
    if torch.cuda.max_memory_allocated() > 79 * 1024**3:
        raise RuntimeError("teacher leaves no safe H100 headroom")
    return processor, model


def generate_labels(config_path: str, limit: int | None = None) -> None:
    cfg = load_config(config_path)
    root = label_dir(cfg["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    completed = _existing_ids(root)
    expected = int(cfg["data"]["quotas"]["av_sft"]) + int(cfg["data"]["quotas"]["ar_sft"])
    if len(completed) >= expected:
        print("Teacher labeling already complete.")
        return

    runtime = _teacher_runtime(cfg)
    model_name = runtime["model"]
    code_revision = git_revision(Path(__file__).resolve().parents[1])
    processor, model = _load_teacher(cfg)
    configured_batch_size = int(cfg["teacher"]["batch_size"])
    batch_size = int(
        os.environ.get("DELTA_NLA_TEACHER_BATCH_SIZE", configured_batch_size)
    )
    if batch_size <= 0:
        raise ValueError("effective teacher batch size must be positive")
    print(json.dumps({
        "teacher_runtime": {
            "backend": runtime["backend"],
            "model": model_name,
            "revision": runtime["revision"],
            "configured_batch_size": configured_batch_size,
            "effective_batch_size": batch_size,
            "length_sort_window": batch_size * 8,
            "guided_regex": os.environ.get(
                "DELTA_NLA_TEACHER_GUIDED_REGEX", "0"
            )
            == "1",
        }
    }, sort_keys=True), flush=True)

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
            ("teacher_model", pa.string()),
            ("teacher_revision", pa.string()),
            ("teacher_backend", pa.string()),
            ("teacher_prompt_sha256", pa.string()),
            ("teacher_code_revision", pa.string()),
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
    pbar = tqdm(total=remaining_total, desc=f"{model_name} teacher labels")
    prompt_material = (
        _remote_prompt(teacher_prompt("", ""), 0)
        if runtime["backend"] == "openai_compat"
        else TEACHER_SYSTEM_PROMPT + teacher_prompt("", "")
    )
    if os.environ.get("DELTA_NLA_TEACHER_GUIDED_REGEX", "0") == "1":
        prompt_material += GUIDED_EXPLANATION_REGEX
    prompt_material += json.dumps(_remote_logit_bias(), sort_keys=True)
    prompt_sha256 = sha256_text(prompt_material)
    try:
        source = _rows(cfg["run_dir"], completed)
        # Sorting a small window by prompt length limits padding/KV-cache waste
        # while retaining bounded memory and resumability by row ID.
        sort_window = batch_size * 8
        while attempted < remaining_total:
            window = list(
                itertools.islice(
                    source,
                    min(sort_window, remaining_total - attempted),
                )
            )
            if not window:
                break
            prepared: list[tuple[dict[str, Any], str]] = []
            for row in window:
                diagnostics = json.dumps(
                    json.loads(row["diagnostics"]), indent=2, ensure_ascii=False
                )
                prepared.append(
                    (row, teacher_prompt(row["context"], diagnostics))
                )
            prepared.sort(key=lambda item: len(item[1]))
            for offset in range(0, len(prepared), batch_size):
                current = prepared[offset : offset + batch_size]
                prompts = [prompt for _, prompt in current]
                raw, explanations, valid, attempts = _label_prompts(
                    model, processor, prompts, cfg
                )
                if not all(valid):
                    failed = valid.index(False)
                    row = current[failed][0]
                    raise RuntimeError(
                        "teacher failed the strict explanation format after "
                        f"{attempts[failed]} attempts for row_id={row['row_id']}; "
                        f"last output={raw[failed][:500]!r}"
                    )
                for index, (row, _) in enumerate(current):
                    rows_buffer.append({
                        "row_id": row["row_id"],
                        "split": row["split"],
                        "explanation": explanations[index],
                        "raw_output": raw[index],
                        "valid_format": valid[index],
                        "attempts": attempts[index],
                        "teacher_model": model_name,
                        "teacher_revision": runtime["revision"],
                        "teacher_backend": runtime["backend"],
                        "teacher_prompt_sha256": prompt_sha256,
                        "teacher_code_revision": code_revision,
                    })
                attempted += len(current)
                valid_count += sum(valid)
                pbar.update(len(current))
                if len(rows_buffer) >= int(cfg["teacher"]["shard_rows"]):
                    flush()
        flush()
    finally:
        pbar.close()
        if model is not None:
            del model
        gc.collect()
        if runtime["backend"] == "transformers":
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
        "teacher_revision": runtime["revision"],
        "teacher_backend": runtime["backend"],
        "teacher_prompt_sha256": prompt_sha256,
        "teacher_code_revision": code_revision,
        "enable_thinking": bool(cfg["teacher"]["enable_thinking"]),
        "configured_batch_size": configured_batch_size,
        "effective_batch_size": batch_size,
        "length_sort_window": sort_window,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated()
            if runtime["backend"] == "transformers"
            else None
        ),
    }
    atomic_json(root / "progress.json", summary)
    if len(all_ids) == expected:
        atomic_json(root / "complete.json", summary)
    print(json.dumps(summary, indent=2))


def benchmark_batch_sizes(
    config_path: str,
    batch_sizes: Sequence[int],
    examples: int,
    output_path: str | None = None,
) -> None:
    """Compare deterministic outputs and throughput without writing labels."""
    cfg = load_config(config_path)
    if examples <= 0 or any(size <= 0 for size in batch_sizes):
        raise ValueError("benchmark examples and batch sizes must be positive")
    rows = list(itertools.islice(_rows(cfg["run_dir"], set()), examples))
    if len(rows) != examples:
        raise RuntimeError(f"requested {examples} benchmark rows, found {len(rows)}")
    prompts = [
        teacher_prompt(
            row["context"],
            json.dumps(json.loads(row["diagnostics"]), indent=2, ensure_ascii=False),
        )
        for row in rows
    ]
    processor, model = _load_teacher(cfg)
    runtime = _teacher_runtime(cfg)
    results: dict[str, Any] = {}
    output_by_size: dict[str, list[str]] = {}
    baseline: list[str] | None = None
    try:
        for batch_size in batch_sizes:
            if runtime["backend"] == "transformers":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            outputs: list[str] = []
            explanations: list[str] = []
            valid: list[bool] = []
            attempts: list[int] = []
            for offset in range(0, len(prompts), batch_size):
                batch_raw, batch_explanations, batch_valid, batch_attempts = (
                    _label_prompts(
                        model,
                        processor,
                        prompts[offset : offset + batch_size],
                        cfg,
                    )
                )
                outputs.extend(batch_raw)
                explanations.extend(batch_explanations)
                valid.extend(batch_valid)
                attempts.extend(batch_attempts)
            if runtime["backend"] == "transformers":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            if baseline is None:
                baseline = outputs
            output_by_size[str(batch_size)] = outputs
            baseline_canonical = [parse_explanation(output)[0] for output in baseline]
            format_valid = [parse_explanation(output)[1] for output in outputs]
            results[str(batch_size)] = {
                "elapsed_seconds": elapsed,
                "rows_per_second": len(outputs) / elapsed,
                "rows_per_hour": len(outputs) / elapsed * 3600,
                "valid_labels_per_hour": sum(valid) / elapsed * 3600,
                "contract_valid_rate": sum(valid) / len(valid),
                "format_valid_rate": sum(format_valid) / len(format_valid),
                "forbidden_term_rate": sum(
                    _has_forbidden_explanation_terms(explanation)
                    for explanation in explanations
                )
                / len(explanations),
                "total_generation_attempts": sum(attempts),
                "retry_rate": sum(attempt > 1 for attempt in attempts) / len(attempts),
                "exact_match_to_first_size": sum(
                    output == reference
                    for output, reference in zip(outputs, baseline, strict=True)
                ) / len(outputs),
                "canonical_match_to_first_size": sum(
                    explanation == reference
                    for explanation, reference in zip(
                        explanations, baseline_canonical, strict=True
                    )
                ) / len(outputs),
                "peak_cuda_memory_bytes": (
                    torch.cuda.max_memory_allocated()
                    if runtime["backend"] == "transformers"
                    else None
                ),
            }
        first_rate = results[str(batch_sizes[0])]["rows_per_second"]
        for result in results.values():
            result["speedup_to_first_size"] = result["rows_per_second"] / first_rate
    finally:
        if model is not None:
            del model
        gc.collect()
        if runtime["backend"] == "transformers":
            torch.cuda.empty_cache()
    report = {
        "examples": examples,
        "teacher_runtime": runtime,
        "batch_sizes_in_order": list(batch_sizes),
        "results": results,
        "rows": [
            {
                "row_id": row["row_id"],
                "split": row["split"],
                "outputs": {
                    size: outputs[index]
                    for size, outputs in output_by_size.items()
                },
            }
            for index, row in enumerate(rows)
        ],
    }
    if output_path:
        atomic_json(output_path, report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--benchmark-batch-sizes",
        help="comma-separated sizes; benchmark only and do not write labels",
    )
    parser.add_argument("--benchmark-examples", type=int, default=32)
    parser.add_argument("--benchmark-output")
    args = parser.parse_args()
    if args.benchmark_batch_sizes:
        if args.limit is not None:
            parser.error("--limit cannot be combined with --benchmark-batch-sizes")
        sizes = [int(value) for value in args.benchmark_batch_sizes.split(",")]
        benchmark_batch_sizes(
            args.config,
            sizes,
            args.benchmark_examples,
            output_path=args.benchmark_output,
        )
    else:
        generate_labels(args.config, args.limit)


if __name__ == "__main__":
    main()
