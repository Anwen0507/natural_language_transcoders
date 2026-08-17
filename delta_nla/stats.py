"""Compute train-only fixed delta standardization and behavior-loss baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm import tqdm

from delta_nla.config import atomic_json, load_config, run_dir, seed_everything
from delta_nla.data import SPLITS, fixed_list_numpy, iter_split_batches
from delta_nla.models import TargetProjection


TRAIN_SPLITS = ("av_sft", "ar_sft", "rl")


def _kl(true_logits: torch.Tensor, predicted_logits: torch.Tensor) -> torch.Tensor:
    true_logp = F.log_softmax(true_logits.float(), dim=-1)
    true_p = true_logp.exp()
    pred_logp = F.log_softmax(predicted_logits.float(), dim=-1)
    return (true_p * (true_logp - pred_logp)).sum(dim=-1)


def compute(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(int(cfg["seed"]))
    root = run_dir(cfg) / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "delta_stats.json"
    mean_path = root / "delta_mean.safetensors"
    if metadata_path.exists() and mean_path.exists():
        print("Delta statistics already exist.")
        return

    total: np.ndarray | None = None
    total_sq = np.float64(0.0)
    count = 0
    norm_values: list[np.ndarray] = []
    for split in TRAIN_SPLITS:
        for batch in tqdm(
            iter_split_batches(cfg["run_dir"], split, ["delta"], batch_size=4096),
            desc=f"stats:{split}",
        ):
            delta = fixed_list_numpy(batch.column("delta")).astype(np.float64)
            if total is None:
                total = np.zeros(delta.shape[1], dtype=np.float64)
            total += delta.sum(axis=0)
            total_sq += np.square(delta).sum(dtype=np.float64)
            count += delta.shape[0]
            if sum(len(x) for x in norm_values) < 100000:
                norm_values.append(np.linalg.norm(delta, axis=1))
    if total is None or count == 0:
        raise RuntimeError("no training deltas")
    mean = total / count
    centered_sum_sq = total_sq - count * np.dot(mean, mean)
    if centered_sum_sq <= 0:
        raise RuntimeError("delta variance is not positive")
    d_model = mean.shape[0]
    scale = math.sqrt(centered_sum_sq / (count * d_model))

    # Baseline for the auxiliary prediction loss: replace each true delta with
    # the train-set mean delta while retaining that example's r0.
    projection = TargetProjection.load(cfg["run_dir"], device=cfg["runtime"]["device"])
    mean_gpu = torch.from_numpy(mean.astype(np.float32)).to(cfg["runtime"]["device"])
    baseline_target = int(cfg["delta"]["prediction_baseline_examples"])
    kl_sum = 0.0
    kl_count = 0
    with torch.inference_mode():
        for split in TRAIN_SPLITS:
            if kl_count >= baseline_target:
                break
            for batch in iter_split_batches(
                cfg["run_dir"], split, ["r0", "delta"], batch_size=32
            ):
                r0 = torch.from_numpy(fixed_list_numpy(batch.column("r0"))).to(
                    cfg["runtime"]["device"]
                )
                delta = torch.from_numpy(fixed_list_numpy(batch.column("delta"))).to(
                    cfg["runtime"]["device"]
                )
                take = min(r0.shape[0], baseline_target - kl_count)
                r0, delta = r0[:take], delta[:take]
                true_logits = projection(r0 + delta)
                mean_logits = projection(r0 + mean_gpu)
                values = _kl(true_logits, mean_logits)
                kl_sum += float(values.sum())
                kl_count += take
                if kl_count >= baseline_target:
                    break
    prediction_baseline = kl_sum / kl_count
    if not math.isfinite(prediction_baseline) or prediction_baseline <= 0:
        raise RuntimeError(f"invalid prediction KL baseline {prediction_baseline}")

    save_file({"mean": torch.from_numpy(mean.astype(np.float32))}, str(mean_path))
    sampled_norms = np.concatenate(norm_values) if norm_values else np.array([], dtype=np.float64)
    atomic_json(metadata_path, {
        "count": count,
        "d_model": d_model,
        "mean_norm": float(np.linalg.norm(mean)),
        "scale": scale,
        "centered_mse_baseline": centered_sum_sq / (count * d_model),
        "standardized_mse_baseline": centered_sum_sq / (count * d_model * scale * scale),
        "prediction_kl_baseline": prediction_baseline,
        "prediction_kl_baseline_examples": kl_count,
        "sample_delta_norm_mean": float(sampled_norms.mean()) if sampled_norms.size else None,
        "sample_delta_norm_std": float(sampled_norms.std()) if sampled_norms.size else None,
        "statistics_splits": list(TRAIN_SPLITS),
        "excluded_split": "eval",
        "definition": "x=(delta-mean)/global_scalar_rms",
    })
    print(json.dumps(json.loads(metadata_path.read_text()), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    compute(args.config)


if __name__ == "__main__":
    main()
