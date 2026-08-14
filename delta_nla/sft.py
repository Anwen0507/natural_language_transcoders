"""Full-parameter BF16-autocast SFT for the 0.5B AV and full-depth AR."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer

from delta_nla.config import atomic_json, load_config, run_dir, seed_everything
from delta_nla.data import DeltaStatistics, load_split_vectors, load_teacher_labels
from delta_nla.losses import reconstruction_losses
from delta_nla.models import (
    DeltaReconstructor,
    TargetProjection,
    inject_vectors,
    load_actor,
)
from delta_nla.prompts import INJECTION_CHAR, actor_prompt, ar_prompt, wrap_explanation


def _stable_fraction(row_id: str) -> float:
    value = int.from_bytes(hashlib.sha256(row_id.encode()).digest()[:8], "big")
    return value / 2**64


def _atomic_replace_dir(tmp: Path, final: Path) -> None:
    if not tmp.exists():
        raise FileNotFoundError(f"temporary checkpoint directory vanished: {tmp}")
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)


def _save_actor(model, tokenizer, path: Path, metadata: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(tmp)
    (tmp / "delta_nla_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _atomic_replace_dir(tmp, path)


def _save_ar(model, path: Path, cfg: dict[str, Any], metadata: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_checkpoint(
        tmp,
        base_model=cfg["models"]["ar_init"],
        revision=cfg["models"]["qwen_revision"],
        metadata=metadata,
    )
    _atomic_replace_dir(tmp, path)


def _append_metric(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _pad(sequences: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = 1
    return ids, mask


class TrainingData:
    def __init__(self, cfg: dict[str, Any], split: str, tokenizer):
        arrays = load_split_vectors(cfg["run_dir"], split)
        labels = load_teacher_labels(cfg["run_dir"], split=split, valid_only=True)
        ids = arrays["row_id"]
        keep = np.asarray([row_id in labels for row_id in ids], dtype=bool)
        if not keep.any():
            raise RuntimeError(f"no valid teacher labels for {split}")
        self.row_ids = ids[keep]
        self.r0 = arrays["r0"][keep]
        self.delta = arrays["delta"][keep]
        self.explanations = [labels[row_id] for row_id in self.row_ids]
        stats = DeltaStatistics.load(cfg["run_dir"])
        self.x = ((self.delta - stats.mean.numpy()) / stats.scale).astype(np.float32)
        fraction = float(cfg["sft"]["validation_fraction"])
        if len(self.row_ids) < 2:
            raise RuntimeError("SFT needs at least two labeled rows for train/validation")
        # Select an exact hash-ranked count instead of thresholding hashes.  The
        # latter can produce an empty partition in small smoke tests and makes
        # validation-set size vary needlessly between runs.
        n_validation = min(
            len(self.row_ids) - 1,
            max(1, round(len(self.row_ids) * fraction)),
        )
        ranked = np.argsort(
            np.asarray([_stable_fraction(str(row_id)) for row_id in self.row_ids])
        )
        self.is_validation = np.zeros(len(self.row_ids), dtype=bool)
        self.is_validation[ranked[:n_validation]] = True

    def indices(self, validation: bool) -> np.ndarray:
        return np.flatnonzero(self.is_validation == validation)


class TokenBuilder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.actor_messages = [{"role": "user", "content": actor_prompt()}]
        self.actor_prefix = self.tokenizer.apply_chat_template(
            self.actor_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        marker = self.tokenizer.encode(INJECTION_CHAR, add_special_tokens=False)
        if len(marker) != 1:
            raise RuntimeError(f"injection character is not one token: {marker}")
        self.injection_token_id = marker[0]
        if self.actor_prefix.count(self.injection_token_id) != 1:
            raise RuntimeError("actor prompt does not contain exactly one injection token")

    def actor_batch(self, explanations: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences: list[list[int]] = []
        labels: list[list[int]] = []
        for explanation in explanations:
            messages = self.actor_messages + [
                {"role": "assistant", "content": wrap_explanation(explanation)}
            ]
            full = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_dict=False
            )
            if full[: len(self.actor_prefix)] != self.actor_prefix:
                raise RuntimeError("chat template full sequence does not share actor prefix")
            sequences.append(full)
            labels.append([-100] * len(self.actor_prefix) + full[len(self.actor_prefix) :])
        ids, mask = _pad(sequences, self.tokenizer.pad_token_id)
        lab = torch.full_like(ids, -100)
        for i, seq in enumerate(labels):
            lab[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return ids, mask, lab

    def ar_batch(self, explanations: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [
            self.tokenizer(ar_prompt(explanation), add_special_tokens=True)["input_ids"]
            for explanation in explanations
        ]
        return _pad(encoded, self.tokenizer.pad_token_id)


def _optimizer(parameters, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    kwargs = dict(
        lr=float(cfg["sft"]["learning_rate"]),
        weight_decay=float(cfg["sft"]["weight_decay"]),
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    try:
        return torch.optim.AdamW(parameters, fused=True, **kwargs)
    except TypeError:
        return torch.optim.AdamW(parameters, **kwargs)


def _scheduler(optimizer, total_steps: int, cfg: dict[str, Any]):
    warmup = max(1, round(total_steps * float(cfg["sft"]["warmup_ratio"])))
    minimum = float(cfg["sft"]["min_learning_rate_ratio"])

    def factor(step: int) -> float:
        if step < warmup:
            return max((step + 1) / warmup, 1e-6)
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def _validate_av(model, builder: TokenBuilder, data: TrainingData, cfg: dict[str, Any]) -> dict[str, float]:
    model.eval()
    indices = data.indices(True)[: int(cfg["sft"]["validation_examples"])]
    micro = int(cfg["sft"]["micro_batch_size"])
    real_sum = shuffled_sum = 0.0
    count = 0
    rng = np.random.default_rng(int(cfg["seed"]) + 77)
    shuffled = rng.permutation(indices)
    for offset in range(0, len(indices), micro):
        ix = indices[offset : offset + micro]
        sx = shuffled[offset : offset + micro]
        ids, mask, labels = builder.actor_batch([data.explanations[i] for i in ix])
        ids, mask, labels = ids.cuda(), mask.cuda(), labels.cuda()
        x = torch.from_numpy(data.x[ix]).cuda()
        x_shuf = torch.from_numpy(data.x[sx]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            real = model(
                inputs_embeds=inject_vectors(
                    model, ids, x, builder.injection_token_id,
                    float(cfg["delta"]["injection_alpha"]),
                ),
                attention_mask=mask,
                labels=labels,
                use_cache=False,
            ).loss
            wrong = model(
                inputs_embeds=inject_vectors(
                    model, ids, x_shuf, builder.injection_token_id,
                    float(cfg["delta"]["injection_alpha"]),
                ),
                attention_mask=mask,
                labels=labels,
                use_cache=False,
            ).loss
        real_sum += float(real) * len(ix)
        shuffled_sum += float(wrong) * len(ix)
        count += len(ix)
    model.train()
    return {
        "validation_ce": real_sum / count,
        "validation_shuffled_ce": shuffled_sum / count,
        "real_vs_shuffled_gap": shuffled_sum / count - real_sum / count,
    }


@torch.no_grad()
def _validate_ar(
    model, builder: TokenBuilder, data: TrainingData, cfg: dict[str, Any],
    stats: DeltaStatistics, projection: TargetProjection,
) -> dict[str, float]:
    model.eval()
    indices = data.indices(True)[: int(cfg["sft"]["validation_examples"])]
    micro = int(cfg["sft"]["micro_batch_size"])
    totals = {"total": 0.0, "mse": 0.0, "kl": 0.0}
    count = 0
    for offset in range(0, len(indices), micro):
        ix = indices[offset : offset + micro]
        ids, mask = builder.ar_batch([data.explanations[i] for i in ix])
        ids, mask = ids.cuda(), mask.cuda()
        x = torch.from_numpy(data.x[ix]).cuda()
        r0 = torch.from_numpy(data.r0[ix]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _ = model(ids, mask)
        losses = reconstruction_losses(
            pred, x, r0, stats, projection,
            float(cfg["delta"]["prediction_loss_weight"]),
        )
        totals["total"] += float(losses.total.sum())
        totals["mse"] += float(losses.vector_mse.sum())
        totals["kl"] += float(losses.prediction_kl.sum())
        count += len(ix)
    model.train()
    mse = totals["mse"] / count
    return {
        "validation_total_loss": totals["total"] / count,
        "validation_vector_mse": mse,
        "validation_vector_fve": 1.0 - mse,
        "validation_prediction_kl": totals["kl"] / count,
    }


def train(config_path: str, role: str) -> None:
    if role not in ("av", "ar"):
        raise ValueError(role)
    cfg = load_config(config_path)
    seed_everything(int(cfg["seed"]) + (0 if role == "av" else 1))
    output = run_dir(cfg) / "sft" / role
    final = output / "final"
    if final.exists():
        print(f"{role.upper()} SFT already complete: {final}")
        return
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["models"]["av_init"], revision=cfg["models"]["qwen_revision"]
    )
    builder = TokenBuilder(tokenizer)
    split = "av_sft" if role == "av" else "ar_sft"
    data = TrainingData(cfg, split, tokenizer)
    train_indices = data.indices(False)
    rng = np.random.default_rng(int(cfg["seed"]) + (10 if role == "av" else 20))
    permutation = rng.permutation(train_indices)
    global_batch = int(cfg["sft"]["global_batch_size"])
    micro = int(cfg["sft"]["micro_batch_size"])
    total_steps = math.ceil(len(permutation) / global_batch) * int(cfg["sft"]["epochs"])

    if role == "av":
        model = load_actor(
            cfg["models"]["av_init"], revision=cfg["models"]["qwen_revision"],
            dtype=torch.float32, device="cuda",
        )
        stats = projection = None
    else:
        model = DeltaReconstructor.from_base(
            cfg["models"]["ar_init"], revision=cfg["models"]["qwen_revision"],
            dtype=torch.float32, device="cuda",
        )
        stats = DeltaStatistics.load(cfg["run_dir"], device="cuda")
        projection = TargetProjection.load(cfg["run_dir"], device="cuda")

    optimizer = _optimizer(model.parameters(), cfg)
    scheduler = _scheduler(optimizer, total_steps, cfg)
    model.train()
    step = 0
    metric_path = output / "metrics.jsonl"
    for epoch in range(int(cfg["sft"]["epochs"])):
        if epoch:
            rng = np.random.default_rng(int(cfg["seed"]) + epoch + (10 if role == "av" else 20))
            permutation = rng.permutation(train_indices)
        for start in range(0, len(permutation), global_batch):
            batch_indices = permutation[start : start + global_batch]
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            seen = 0
            for offset in range(0, len(batch_indices), micro):
                ix = batch_indices[offset : offset + micro]
                explanations = [data.explanations[i] for i in ix]
                weight = len(ix) / len(batch_indices)
                if role == "av":
                    ids, mask, labels = builder.actor_batch(explanations)
                    ids, mask, labels = ids.cuda(), mask.cuda(), labels.cuda()
                    x = torch.from_numpy(data.x[ix]).cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        embeds = inject_vectors(
                            model, ids, x, builder.injection_token_id,
                            float(cfg["delta"]["injection_alpha"]),
                        )
                        loss = model(
                            inputs_embeds=embeds,
                            attention_mask=mask,
                            labels=labels,
                            use_cache=False,
                        ).loss
                else:
                    ids, mask = builder.ar_batch(explanations)
                    ids, mask = ids.cuda(), mask.cuda()
                    x = torch.from_numpy(data.x[ix]).cuda()
                    r0 = torch.from_numpy(data.r0[ix]).cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred, _ = model(ids, mask)
                    assert stats is not None and projection is not None
                    loss = reconstruction_losses(
                        pred, x, r0, stats, projection,
                        float(cfg["delta"]["prediction_loss_weight"]),
                    ).total.mean()
                (loss * weight).backward()
                weighted_loss += float(loss.detach()) * weight
                seen += len(ix)
            grad_norm = float(clip_grad_norm_(model.parameters(), float(cfg["sft"]["max_grad_norm"])))
            if not math.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"non-finite {role} SFT gradient norm")
            optimizer.step()
            scheduler.step()
            step += 1
            record = {
                "role": role,
                "step": step,
                "epoch": epoch,
                "examples": seen,
                "train_loss": weighted_loss,
                "grad_norm": grad_norm,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            _append_metric(metric_path, record)
            print(json.dumps(record), flush=True)
            interval = int(cfg["sft"]["checkpoint_interval"])
            if step % interval == 0 or step == total_steps:
                if role == "av":
                    validation = _validate_av(model, builder, data, cfg)
                else:
                    validation = _validate_ar(model, builder, data, cfg, stats, projection)
                record = {"role": role, "step": step, **validation}
                _append_metric(metric_path, record)
                print(json.dumps(record), flush=True)
                checkpoint = output / f"checkpoint_{step:06d}"
                meta = {
                    "stage": "sft",
                    "role": role,
                    "step": step,
                    "full_transformer_delta": True,
                    "injection_alpha": float(cfg["delta"]["injection_alpha"]),
                    "standardization": "fixed train mean and scalar RMS",
                    **validation,
                }
                if role == "av":
                    _save_actor(model, tokenizer, checkpoint, meta)
                else:
                    _save_ar(model, checkpoint, cfg, meta)
                atomic_json(output / "latest.json", {"step": step, "checkpoint": str(checkpoint)})

    meta = {
        "stage": "sft",
        "role": role,
        "step": step,
        "full_transformer_delta": True,
        "injection_alpha": float(cfg["delta"]["injection_alpha"]),
        "standardization": "fixed train mean and scalar RMS",
    }
    if role == "av":
        _save_actor(model, tokenizer, final, meta)
    else:
        _save_ar(model, final, cfg, meta)
    atomic_json(output / "complete.json", {"role": role, "steps": step, "final": str(final)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", required=True, choices=["av", "ar"])
    args = parser.parse_args()
    train(args.config, args.role)


if __name__ == "__main__":
    main()
