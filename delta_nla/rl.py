"""Synchronous single-H100 GRPO with a concurrently trained delta AR."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer

from delta_nla.config import atomic_json, load_config, run_dir, seed_everything
from delta_nla.data import DeltaStatistics, load_split_vectors
from delta_nla.losses import reconstruction_losses
from delta_nla.models import DeltaReconstructor, TargetProjection, load_actor
from delta_nla.policy import GeneratedBatch, generate_actor, response_log_probs
from delta_nla.sft import TokenBuilder, _save_actor, _save_ar


def _append_metric(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _adam(parameters, learning_rate: float, weight_decay: float):
    kwargs = dict(
        lr=float(learning_rate), weight_decay=float(weight_decay),
        betas=(0.9, 0.95), eps=1e-8,
    )
    try:
        return torch.optim.AdamW(parameters, fused=True, **kwargs)
    except TypeError:
        return torch.optim.AdamW(parameters, **kwargs)


def _rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def _score_ar(
    ar,
    builder: TokenBuilder,
    explanations: list[str],
    x: torch.Tensor,
    r0: torch.Tensor,
    stats: DeltaStatistics,
    projection: TargetProjection,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ar.eval()
    micro = int(cfg["rl"]["ar_micro_batch_size"])
    total_parts, mse_parts, kl_parts, pred_parts = [], [], [], []
    for start in range(0, len(explanations), micro):
        end = min(len(explanations), start + micro)
        ids, mask = builder.ar_batch(explanations[start:end])
        ids, mask = ids.cuda(), mask.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _ = ar(ids, mask)
        losses = reconstruction_losses(
            pred, x[start:end], r0[start:end], stats, projection,
            float(cfg["delta"]["prediction_loss_weight"]),
        )
        total_parts.append(losses.total.detach())
        mse_parts.append(losses.vector_mse.detach())
        kl_parts.append(losses.prediction_kl.detach())
        pred_parts.append(pred.detach())
    return (
        torch.cat(total_parts), torch.cat(mse_parts),
        torch.cat(kl_parts), torch.cat(pred_parts),
    )


def _update_av(
    av,
    reference,
    optimizer,
    generated: GeneratedBatch,
    x: torch.Tensor,
    advantages: torch.Tensor,
    builder: TokenBuilder,
    cfg: dict[str, Any],
) -> tuple[float, float, float, bool]:
    av.train()
    reference.eval()
    optimizer.zero_grad(set_to_none=True)
    micro = int(cfg["rl"]["av_micro_batch_size"])
    n = x.shape[0]
    total_loss = total_policy = total_kl = 0.0
    for start in range(0, n, micro):
        end = min(n, start + micro)
        weight = (end - start) / n
        ids = generated.full_ids[start:end]
        full_mask = generated.full_attention_mask[start:end]
        response_ids = generated.response_ids[start:end]
        response_mask = generated.response_mask[start:end]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            current = response_log_probs(
                av, ids, full_mask, response_ids, response_mask, x[start:end],
                builder.injection_token_id, float(cfg["delta"]["injection_alpha"]),
                generated.prefix_length,
            )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reference_logp = response_log_probs(
                reference, ids, full_mask, response_ids, response_mask, x[start:end],
                builder.injection_token_id, float(cfg["delta"]["injection_alpha"]),
                generated.prefix_length,
            )
        # Exactly one policy epoch per fresh rollout: current.detach() is the
        # behavior-policy log probability and is bit-identical by construction.
        old = current.detach()
        ratio = torch.exp((current - old).clamp(-20, 20))
        adv = advantages[start:end, None]
        unclipped = ratio * adv
        clipped = ratio.clamp(
            1.0 - float(cfg["rl"]["policy_clip"]),
            1.0 + float(cfg["rl"]["policy_clip"]),
        ) * adv
        token_mask = response_mask.float()
        denom = token_mask.sum(dim=1).clamp_min(1)
        policy_per_sequence = -(
            torch.minimum(unclipped, clipped) * token_mask
        ).sum(dim=1) / denom
        # k2 estimator requested in the design: 1/2(log p - log p_ref)^2.
        kl_per_sequence = (
            0.5 * (current - reference_logp).square() * token_mask
        ).sum(dim=1) / denom
        policy_loss = policy_per_sequence.mean()
        kl_loss = kl_per_sequence.mean()
        loss = policy_loss + float(cfg["rl"]["policy_kl_coefficient"]) * kl_loss
        (loss * weight).backward()
        total_loss += float(loss.detach()) * weight
        total_policy += float(policy_loss.detach()) * weight
        total_kl += float(kl_loss.detach()) * weight
    grad_norm = float(clip_grad_norm_(av.parameters(), float(cfg["rl"]["max_grad_norm"])))
    finite = math.isfinite(grad_norm)
    if finite:
        optimizer.step()
    else:
        optimizer.zero_grad(set_to_none=True)
    return total_loss, total_policy, total_kl, finite


def _update_ar(
    ar,
    optimizer,
    builder: TokenBuilder,
    explanations: list[str],
    valid: torch.Tensor,
    x: torch.Tensor,
    r0: torch.Tensor,
    stats: DeltaStatistics,
    projection: TargetProjection,
    cfg: dict[str, Any],
) -> tuple[float, float, float, bool, int]:
    kept = valid.nonzero(as_tuple=False).flatten()
    if kept.numel() == 0:
        return math.nan, math.nan, math.nan, True, 0
    ar.train()
    optimizer.zero_grad(set_to_none=True)
    micro = int(cfg["rl"]["ar_micro_batch_size"])
    total = total_mse = total_kl = 0.0
    valid_explanations = [explanations[int(i)] for i in kept.tolist()]
    kept_x, kept_r0 = x[kept], r0[kept]
    n = len(valid_explanations)
    for start in range(0, n, micro):
        end = min(n, start + micro)
        weight = (end - start) / n
        ids, mask = builder.ar_batch(valid_explanations[start:end])
        ids, mask = ids.cuda(), mask.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _ = ar(ids, mask)
        losses = reconstruction_losses(
            pred, kept_x[start:end], kept_r0[start:end], stats, projection,
            float(cfg["delta"]["prediction_loss_weight"]),
        )
        loss = losses.total.mean()
        (loss * weight).backward()
        total += float(loss.detach()) * weight
        total_mse += float(losses.vector_mse.mean().detach()) * weight
        total_kl += float(losses.prediction_kl.mean().detach()) * weight
    grad_norm = float(clip_grad_norm_(ar.parameters(), float(cfg["rl"]["max_grad_norm"])))
    finite = math.isfinite(grad_norm)
    if finite:
        optimizer.step()
    else:
        optimizer.zero_grad(set_to_none=True)
    return total, total_mse, total_kl, finite, n


@torch.no_grad()
def _evaluate(
    av, ar, tokenizer, builder: TokenBuilder, eval_data: dict[str, object],
    stats: DeltaStatistics, projection: TargetProjection, cfg: dict[str, Any], step: int,
) -> dict[str, float]:
    saved_rng = _rng_state()
    seed_everything(int(cfg["seed"]) + 100000 + step)
    n = min(int(cfg["rl"]["evaluation_examples"]), len(eval_data["row_id"]))
    delta = torch.from_numpy(eval_data["delta"][:n]).cuda()
    r0 = torch.from_numpy(eval_data["r0"][:n]).cuda()
    x = stats.standardize(delta)
    generated = generate_actor(
        av, tokenizer, builder.actor_prefix, x, builder.injection_token_id,
        float(cfg["delta"]["injection_alpha"]),
        max_new_tokens=int(cfg["rl"]["response_max_new_tokens"]),
        do_sample=False,
    )
    total, mse, kl, pred = _score_ar(
        ar, builder, generated.explanations, x, r0, stats, projection, cfg
    )
    pred_delta = stats.unstandardize(pred.float())
    cosine = torch.nn.functional.cosine_similarity(pred_delta, delta.float(), dim=-1)
    norm_ratio = pred_delta.norm(dim=-1) / delta.float().norm(dim=-1).clamp_min(1e-8)
    result = {
        "eval_total_loss": float(total.mean()),
        "eval_vector_mse": float(mse.mean()),
        "eval_vector_fve": 1.0 - float(mse.mean()),
        "eval_prediction_kl": float(kl.mean()),
        "eval_delta_cosine": float(cosine.mean()),
        "eval_delta_norm_ratio": float(norm_ratio.mean()),
        "eval_valid_format_rate": float(generated.valid_format.float().mean()),
        "eval_cap_hit_rate": float(generated.cap_hit.float().mean()),
    }
    _restore_rng(saved_rng)
    return result


def _save_checkpoint(
    root: Path, step: int, av, ar, tokenizer, av_optimizer, ar_optimizer,
    cfg: dict[str, Any], best: float, stale: int,
) -> Path:
    checkpoint = root / "checkpoints" / f"step_{step:06d}"
    _save_actor(av, tokenizer, checkpoint / "av", {
        "stage": "rl", "step": step, "role": "av",
        "group_size": int(cfg["rl"]["group_size"]),
    })
    _save_ar(ar, checkpoint / "ar", cfg, {
        "stage": "rl", "step": step, "role": "ar", "full_depth": True,
    })
    state = {
        "step": step,
        "checkpoint": str(checkpoint),
        "av_optimizer": av_optimizer.state_dict(),
        "ar_optimizer": ar_optimizer.state_dict(),
        "best_eval_score": best,
        "stale_evaluations": stale,
        "rng": _rng_state(),
    }
    _atomic_torch_save(state, root / "trainer_state.pt")
    atomic_json(root / "latest.json", {"step": step, "checkpoint": str(checkpoint)})
    return checkpoint


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(int(cfg["seed"]) + 200)
    root = run_dir(cfg) / "rl"
    final = root / "final"
    if (root / "complete.json").exists():
        print("RL already complete.")
        return
    root.mkdir(parents=True, exist_ok=True)
    sft_av = run_dir(cfg) / "sft" / "av" / "final"
    sft_ar = run_dir(cfg) / "sft" / "ar" / "final"
    if not sft_av.exists() or not sft_ar.exists():
        raise FileNotFoundError("both AV-SFT and AR-SFT final checkpoints are required")
    tokenizer = AutoTokenizer.from_pretrained(sft_av)
    builder = TokenBuilder(tokenizer)
    train_data = load_split_vectors(cfg["run_dir"], "rl")
    eval_data = load_split_vectors(cfg["run_dir"], "eval")
    stats = DeltaStatistics.load(cfg["run_dir"], device="cuda")
    projection = TargetProjection.load(cfg["run_dir"], device="cuda")

    state_path = root / "trainer_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        checkpoint = Path(state["checkpoint"])
        av = load_actor(checkpoint / "av", dtype=torch.float32, device="cuda")
        ar = DeltaReconstructor.from_checkpoint(checkpoint / "ar", dtype=torch.float32, device="cuda")
        start_step = int(state["step"])
        best = float(state["best_eval_score"])
        stale = int(state["stale_evaluations"])
    else:
        state = None
        av = load_actor(sft_av, dtype=torch.float32, device="cuda")
        ar = DeltaReconstructor.from_checkpoint(sft_ar, dtype=torch.float32, device="cuda")
        start_step = 0
        best = -math.inf
        stale = 0
    reference = load_actor(sft_av, dtype=torch.bfloat16, device="cuda").eval()
    reference.requires_grad_(False)
    av_optimizer = _adam(
        av.parameters(), float(cfg["rl"]["av_learning_rate"]), float(cfg["rl"]["weight_decay"])
    )
    ar_optimizer = _adam(
        ar.parameters(), float(cfg["rl"]["ar_learning_rate"]), float(cfg["rl"]["weight_decay"])
    )
    if state is not None:
        av_optimizer.load_state_dict(state["av_optimizer"])
        ar_optimizer.load_state_dict(state["ar_optimizer"])
        _restore_rng(state["rng"])

    n_prompts = int(cfg["rl"]["prompts_per_step"])
    group = int(cfg["rl"]["group_size"])
    max_steps = int(cfg["rl"]["max_steps"])
    required = n_prompts * max_steps
    if required > len(train_data["row_id"]):
        raise RuntimeError("RL dataset is smaller than unique-prompt schedule")
    permutation = np.random.default_rng(int(cfg["seed"]) + 300).permutation(
        len(train_data["row_id"])
    )[:required]
    metrics_path = root / "metrics.jsonl"

    if start_step == 0:
        initial_eval = _evaluate(
            av, ar, tokenizer, builder, eval_data, stats, projection, cfg, 0
        )
        _append_metric(metrics_path, {"step": 0, **initial_eval})
        print(json.dumps({"step": 0, **initial_eval}), flush=True)
        best = -initial_eval["eval_total_loss"]

    stopped_early = False
    for step in range(start_step + 1, max_steps + 1):
        prompt_ix = permutation[(step - 1) * n_prompts : step * n_prompts]
        delta = torch.from_numpy(train_data["delta"][prompt_ix]).cuda()
        r0 = torch.from_numpy(train_data["r0"][prompt_ix]).cuda()
        x_prompt = stats.standardize(delta)
        x = x_prompt.repeat_interleave(group, dim=0)
        r0_group = r0.repeat_interleave(group, dim=0)

        generated = generate_actor(
            av, tokenizer, builder.actor_prefix, x, builder.injection_token_id,
            float(cfg["delta"]["injection_alpha"]),
            max_new_tokens=int(cfg["rl"]["response_max_new_tokens"]),
            do_sample=True,
            temperature=float(cfg["rl"]["temperature"]),
            top_p=float(cfg["rl"]["top_p"]),
        )
        score_total, score_mse, score_kl, _ = _score_ar(
            ar, builder, generated.explanations, x, r0_group, stats, projection, cfg
        )
        reward = -score_total
        reward = reward - (~generated.valid_format).float() * float(cfg["rl"]["invalid_format_penalty"])
        reward = reward - generated.cap_hit.float() * float(cfg["rl"]["cap_hit_penalty"])
        grouped = reward.view(n_prompts, group)
        advantages = (
            (grouped - grouped.mean(dim=1, keepdim=True))
            / grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        ).reshape(-1)

        av_loss, policy_loss, policy_kl, av_finite = _update_av(
            av, reference, av_optimizer, generated, x, advantages, builder, cfg
        )
        ar_loss, ar_mse, ar_kl, ar_finite, ar_examples = _update_ar(
            ar, ar_optimizer, builder, generated.explanations, generated.valid_format,
            x, r0_group, stats, projection, cfg,
        )
        record = {
            "step": step,
            "reward_mean": float(reward.mean()),
            "reward_std": float(reward.std(unbiased=False)),
            "score_vector_mse": float(score_mse.mean()),
            "score_prediction_kl": float(score_kl.mean()),
            "valid_format_rate": float(generated.valid_format.float().mean()),
            "cap_hit_rate": float(generated.cap_hit.float().mean()),
            "response_tokens_mean": float(generated.response_mask.sum(dim=1).float().mean()),
            "av_loss": av_loss,
            "policy_loss": policy_loss,
            "policy_kl_k2": policy_kl,
            "av_update_finite": av_finite,
            "ar_loss": ar_loss,
            "ar_vector_mse": ar_mse,
            "ar_prediction_kl": ar_kl,
            "ar_update_finite": ar_finite,
            "ar_examples": ar_examples,
        }
        _append_metric(metrics_path, record)
        print(json.dumps(record), flush=True)

        evaluate_now = step % int(cfg["rl"]["evaluation_interval"]) == 0
        save_now = step % int(cfg["rl"]["checkpoint_interval"]) == 0
        if evaluate_now:
            evaluation = _evaluate(
                av, ar, tokenizer, builder, eval_data, stats, projection, cfg, step
            )
            _append_metric(metrics_path, {"step": step, **evaluation})
            print(json.dumps({"step": step, **evaluation}), flush=True)
            score = -evaluation["eval_total_loss"]
            if score > best + 1e-4:
                best, stale = score, 0
            else:
                stale += 1
        if save_now or evaluate_now:
            _save_checkpoint(
                root, step, av, ar, tokenizer, av_optimizer, ar_optimizer,
                cfg, best, stale,
            )
        if (
            step >= int(cfg["rl"]["minimum_steps"])
            and stale >= int(cfg["rl"]["early_stopping_patience"])
        ):
            stopped_early = True
            break

    final_step = step
    _save_actor(av, tokenizer, final / "av", {
        "stage": "rl", "role": "av", "step": final_step,
        "group_size": group,
    })
    _save_ar(ar, final / "ar", cfg, {
        "stage": "rl", "role": "ar", "step": final_step, "full_depth": True,
    })
    atomic_json(root / "complete.json", {
        "final_step": final_step,
        "maximum_steps": max_steps,
        "stopped_early": stopped_early,
        "best_eval_score": best,
        "final": str(final),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
