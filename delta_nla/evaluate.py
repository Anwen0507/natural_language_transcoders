"""Held-out round-trip evaluation and shortcut controls for SFT and RL checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from delta_nla.config import atomic_json, load_config, run_dir, seed_everything
from delta_nla.data import DeltaStatistics, load_split_vectors
from delta_nla.losses import reconstruction_losses
from delta_nla.models import DeltaReconstructor, TargetProjection, load_actor
from delta_nla.policy import generate_actor
from delta_nla.sft import TokenBuilder


@torch.no_grad()
def _ar_predict(ar, builder, explanations: list[str], micro: int) -> torch.Tensor:
    parts = []
    ar.eval()
    for start in range(0, len(explanations), micro):
        ids, mask = builder.ar_batch(explanations[start : start + micro])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _ = ar(ids.cuda(), mask.cuda())
        parts.append(pred.float())
    return torch.cat(parts)


@torch.no_grad()
def _behavior_metrics(
    r0: torch.Tensor,
    true_delta: torch.Tensor,
    pred_delta: torch.Tensor,
    projection: TargetProjection,
    micro: int = 32,
) -> dict[str, float]:
    kl_sum = top1 = top5 = 0.0
    n = r0.shape[0]
    for start in range(0, n, micro):
        end = min(n, start + micro)
        true_logits = projection(r0[start:end] + true_delta[start:end])
        pred_logits = projection(r0[start:end] + pred_delta[start:end])
        true_logp = F.log_softmax(true_logits, dim=-1)
        true_p = true_logp.exp()
        pred_logp = F.log_softmax(pred_logits, dim=-1)
        kl_sum += float((true_p * (true_logp - pred_logp)).sum())
        true_top = true_logits.topk(5, dim=-1).indices
        pred_top = pred_logits.topk(5, dim=-1).indices
        top1 += float((true_top[:, 0] == pred_top[:, 0]).sum())
        top5 += float(
            (true_top[:, :, None] == pred_top[:, None, :]).any(dim=-1).float().sum()
        )
    return {
        "prediction_kl": kl_sum / n,
        "top1_agreement": top1 / n,
        "top5_overlap_fraction": top5 / (5 * n),
    }


@torch.no_grad()
def _vector_metrics(
    r0: torch.Tensor,
    true_delta: torch.Tensor,
    pred_x: torch.Tensor,
    stats: DeltaStatistics,
    projection: TargetProjection,
) -> dict[str, float]:
    pred_delta = stats.unstandardize(pred_x.float())
    error = pred_delta - true_delta.float()
    centered = true_delta.float() - stats.mean
    mse = error.square().mean()
    baseline = centered.square().mean()
    true_norm = true_delta.float().norm(dim=-1).clamp_min(1e-8)
    pred_norm = pred_delta.norm(dim=-1)
    result = {
        "raw_delta_mse": float(mse),
        "raw_delta_mean_predictor_mse": float(baseline),
        "raw_delta_fve": 1.0 - float(mse / baseline),
        "delta_cosine": float(F.cosine_similarity(pred_delta, true_delta.float(), dim=-1).mean()),
        "delta_norm_ratio": float((pred_norm / true_norm).mean()),
        "delta_relative_norm_error": float(((pred_norm - true_norm).abs() / true_norm).mean()),
        "endpoint_mse": float(error.square().mean()),
    }
    result.update(_behavior_metrics(r0, true_delta, pred_delta, projection))
    return result


@torch.no_grad()
def evaluate_pair(
    name: str,
    av_path: Path,
    ar_path: Path,
    data: dict[str, object],
    cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(av_path)
    builder = TokenBuilder(tokenizer)
    av = load_actor(av_path, dtype=torch.float32, device="cuda").eval()
    ar = DeltaReconstructor.from_checkpoint(ar_path, dtype=torch.float32, device="cuda").eval()
    stats = DeltaStatistics.load(cfg["run_dir"], device="cuda")
    projection = TargetProjection.load(cfg["run_dir"], device="cuda")
    n = len(data["row_id"])
    batch_size = int(cfg["rl"]["prompts_per_step"])
    all_explanations: list[str] = []
    all_raw: list[str] = []
    all_valid: list[torch.Tensor] = []
    all_cap: list[torch.Tensor] = []
    x_all = stats.standardize(torch.from_numpy(data["delta"]).cuda())
    r0_all = torch.from_numpy(data["r0"]).cuda()
    for start in range(0, n, batch_size):
        generated = generate_actor(
            av, tokenizer, builder.actor_prefix, x_all[start : start + batch_size],
            builder.injection_token_id, float(cfg["delta"]["injection_alpha"]),
            max_new_tokens=int(cfg["rl"]["response_max_new_tokens"]),
            do_sample=False,
        )
        all_explanations.extend(generated.explanations)
        all_raw.extend(generated.raw_text)
        all_valid.append(generated.valid_format.cpu())
        all_cap.append(generated.cap_hit.cpu())
    pred_x = _ar_predict(
        ar, builder, all_explanations, int(cfg["rl"]["ar_micro_batch_size"])
    )
    metrics: dict[str, Any] = {
        "checkpoint": name,
        "examples": n,
        "valid_format_rate": float(torch.cat(all_valid).float().mean()),
        "cap_hit_rate": float(torch.cat(all_cap).float().mean()),
        **_vector_metrics(r0_all, torch.from_numpy(data["delta"]).cuda(), pred_x, stats, projection),
    }

    # AR shortcut control: correct deltas paired with other explanations.
    permutation = np.random.default_rng(int(cfg["seed"]) + 991).permutation(n)
    shuffled_explanations = [all_explanations[i] for i in permutation]
    shuffled_pred = _ar_predict(
        ar, builder, shuffled_explanations, int(cfg["rl"]["ar_micro_batch_size"])
    )
    shuffled_mse = (shuffled_pred - x_all).square().mean()
    metrics["ar_shuffled_explanation_standardized_mse"] = float(shuffled_mse)

    # AV shortcut controls: shuffle or erase injected deltas, then score the
    # resulting language against each row's original target delta.
    for control_name, control_x in (
        ("shuffled_injection", x_all[torch.from_numpy(permutation).cuda()]),
        ("zero_injection", torch.zeros_like(x_all)),
    ):
        control_explanations: list[str] = []
        control_valid: list[torch.Tensor] = []
        for start in range(0, n, batch_size):
            generated = generate_actor(
                av, tokenizer, builder.actor_prefix, control_x[start : start + batch_size],
                builder.injection_token_id, float(cfg["delta"]["injection_alpha"]),
                max_new_tokens=int(cfg["rl"]["response_max_new_tokens"]),
                do_sample=False,
            )
            control_explanations.extend(generated.explanations)
            control_valid.append(generated.valid_format.cpu())
        control_pred = _ar_predict(
            ar, builder, control_explanations, int(cfg["rl"]["ar_micro_batch_size"])
        )
        metrics[f"{control_name}_standardized_mse_against_original"] = float(
            (control_pred - x_all).square().mean()
        )
        metrics[f"{control_name}_valid_format_rate"] = float(
            torch.cat(control_valid).float().mean()
        )
        metrics[f"{control_name}_exact_text_match_to_real"] = float(
            np.mean([a == b for a, b in zip(control_explanations, all_explanations, strict=True)])
        )

    # Fixed mean-delta baseline (x_hat=0) in every reported metric space.
    baseline = _vector_metrics(
        r0_all,
        torch.from_numpy(data["delta"]).cuda(),
        torch.zeros_like(x_all),
        stats,
        projection,
    )
    metrics["mean_delta_baseline"] = baseline
    samples = [
        {
            "row_id": str(data["row_id"][i]),
            "context": data["context"][i],
            "explanation": all_explanations[i],
            "raw_generation": all_raw[i],
        }
        for i in range(min(100, n))
    ]
    atomic_json(output_dir / f"{name}_metrics.json", metrics)
    atomic_json(output_dir / f"{name}_samples.json", samples)
    del av, ar, projection
    torch.cuda.empty_cache()
    return metrics


def evaluate(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(int(cfg["seed"]) + 900)
    output = run_dir(cfg) / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    data = load_split_vectors(cfg["run_dir"], "eval", include_context=True)
    pairs = {
        "sft": (
            run_dir(cfg) / "sft" / "av" / "final",
            run_dir(cfg) / "sft" / "ar" / "final",
        ),
        "rl": (
            run_dir(cfg) / "rl" / "final" / "av",
            run_dir(cfg) / "rl" / "final" / "ar",
        ),
    }
    results = {}
    for name, (av, ar) in pairs.items():
        if not av.exists() or not ar.exists():
            raise FileNotFoundError(f"missing {name} checkpoint pair: {av}, {ar}")
        results[name] = evaluate_pair(name, av, ar, data, cfg, output)
    atomic_json(output / "comparison.json", results)
    atomic_json(output / "complete.json", {"evaluated": list(results), "examples": len(data["row_id"])})
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
