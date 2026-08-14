"""Fail-fast semantic, memory, and generation checks before paid production stages."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from delta_nla.config import atomic_json, load_config, run_dir, seed_everything, write_run_manifest
from delta_nla.data import atomic_write_table, fixed_list_array, fixed_list_numpy
from delta_nla.models import DeltaReconstructor, inject_vectors
from delta_nla.policy import generate_actor, response_log_probs
from delta_nla.prompts import INJECTION_CHAR, actor_prompt, parse_explanation
from delta_nla.sft import TokenBuilder


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(int(cfg["seed"]))
    write_run_manifest(cfg)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 support is required")
    if torch.cuda.get_device_properties(0).total_memory < 79 * 1024**3:
        raise RuntimeError("this launch configuration requires an 80GB H100")

    # Fixed-size-list vector roundtrip protects the 896-wide parquet contract.
    sample = np.random.default_rng(1).standard_normal((3, 896)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vectors.parquet"
        atomic_write_table(pa.table({"delta": fixed_list_array(sample)}), path)
        restored = fixed_list_numpy(pq.read_table(path)["delta"])
        np.testing.assert_array_equal(sample, restored)

    parsed, valid = parse_explanation(
        "<explanation>\n- Strengthens a likely noun continuation.\n"
        "- Resolves the current syntactic attachment.\n</explanation>"
    )
    if not valid or parsed.count("\n") != 1:
        raise RuntimeError("explanation parser contract failed")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["models"]["av_init"], revision=cfg["models"]["qwen_revision"]
    )
    builder = TokenBuilder(tokenizer)
    if tokenizer.encode(INJECTION_CHAR, add_special_tokens=False) != [builder.injection_token_id]:
        raise RuntimeError("injection marker tokenizer drift")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["models"]["av_init"], revision=cfg["models"]["qwen_revision"],
        dtype=torch.float32, attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).cuda().eval()
    x = torch.randn(2, model.config.hidden_size, device="cuda")
    generated = generate_actor(
        model, tokenizer, builder.actor_prefix, x, builder.injection_token_id,
        float(cfg["delta"]["injection_alpha"]), max_new_tokens=4, do_sample=False,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logp = response_log_probs(
            model, generated.full_ids, generated.full_attention_mask,
            generated.response_ids, generated.response_mask, x,
            builder.injection_token_id, float(cfg["delta"]["injection_alpha"]),
            generated.prefix_length,
        )
    if logp.shape != generated.response_ids.shape or not torch.isfinite(logp).all():
        raise RuntimeError("injected generation/log-prob recomputation failed")
    del model
    torch.cuda.empty_cache()

    ar = DeltaReconstructor.from_base(
        cfg["models"]["ar_init"], revision=cfg["models"]["qwen_revision"],
        dtype=torch.float32, device="cuda",
    ).eval()
    if len(ar.backbone.layers) != int(ar.backbone.config.num_hidden_layers):
        raise RuntimeError("AR is not full depth")
    if not isinstance(ar.backbone.norm, torch.nn.Identity):
        raise RuntimeError("AR final norm was not bypassed")
    ids, mask = builder.ar_batch([parsed, parsed])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred, hidden = ar(ids.cuda(), mask.cuda())
    if pred.shape != (2, 896) or not torch.isfinite(pred).all():
        raise RuntimeError("AR vector output contract failed")
    del ar
    torch.cuda.empty_cache()

    result = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers_imported": True,
        "injection_token_id": builder.injection_token_id,
        "actor_prefix_tokens": len(builder.actor_prefix),
        "generation_shape": list(generated.response_ids.shape),
        "ar_output_shape": list(pred.shape),
    }
    atomic_json(run_dir(cfg) / "preflight.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
