#!/usr/bin/env python3
"""Plot training and held-out curves from a delta-NLA RL metrics JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#16a34a",
    "orange": "#ea580c",
    "purple": "#7c3aed",
    "red": "#dc2626",
    "slate": "#475569",
    "gold": "#ca8a04",
}


def load_rows(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    train = pd.DataFrame(row for row in rows if "reward_mean" in row).sort_values("step")
    heldout = pd.DataFrame(row for row in rows if "eval_total_loss" in row).sort_values("step")
    if train.empty or heldout.empty:
        raise ValueError("metrics must contain both per-step and held-out records")
    return train.reset_index(drop=True), heldout.reset_index(drop=True)


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(1, window // 5)).mean()


def raw_and_smooth(
    ax: plt.Axes,
    frame: pd.DataFrame,
    key: str,
    label: str,
    color: str,
    window: int,
) -> None:
    ax.plot(frame["step"], frame[key], color=color, alpha=0.12, linewidth=0.65)
    ax.plot(
        frame["step"],
        rolling(frame[key], window),
        color=color,
        linewidth=2.0,
        label=f"{label} ({window}-step mean)",
    )


def polish(ax: plt.Axes, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xlabel("RL step")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, color="#cbd5e1", linewidth=0.6, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def final_rl_fve(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text())
    value = payload.get("rl", {}).get("raw_delta_fve")
    return None if value is None else float(value)


def dashboard(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    output: Path,
    window: int,
    exhaustive_fve: float | None,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 16), constrained_layout=True)
    fig.suptitle(
        "Delta NLA RL training curves — full-transformer delta",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.979,
        f"3,000 RL steps on one H100 · raw per-step traces beneath trailing {window}-step means · held-out evaluation every 250 steps",
        ha="center",
        color=COLORS["slate"],
        fontsize=10,
    )

    ax = axes[0, 0]
    ax.plot(
        heldout["step"], heldout["eval_vector_fve"], marker="o", linewidth=2.4,
        color=COLORS["blue"], label="512-example held-out FVE",
    )
    if exhaustive_fve is not None:
        ax.axhline(
            exhaustive_fve, color=COLORS["green"], linestyle="--", linewidth=1.5,
            label=f"Final 5,000-example FVE = {exhaustive_fve:.3f}",
        )
    ax.annotate(
        f"{heldout.iloc[0].eval_vector_fve:.3f}",
        (heldout.iloc[0].step, heldout.iloc[0].eval_vector_fve),
        xytext=(8, -18), textcoords="offset points", fontsize=9,
    )
    ax.annotate(
        f"{heldout.iloc[-1].eval_vector_fve:.3f}",
        (heldout.iloc[-1].step, heldout.iloc[-1].eval_vector_fve),
        xytext=(-38, 10), textcoords="offset points", fontsize=9,
    )
    polish(ax, "Held-out reconstruction improves 0.121 → 0.501", "Fraction of variance explained")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    ax.plot(
        heldout["step"], heldout["eval_total_loss"], marker="o", linewidth=2.2,
        color=COLORS["blue"], label="Reconstruction objective",
    )
    ax.plot(
        heldout["step"], heldout["eval_selection_loss"], marker="s", linewidth=2.0,
        color=COLORS["orange"], label="Format-aware selection loss",
    )
    polish(ax, "Held-out objective (lower is better)", "Loss")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    raw_and_smooth(ax, train, "score_vector_mse", "Reward-time MSE", COLORS["blue"], window)
    raw_and_smooth(ax, train, "ar_vector_mse", "AR-update MSE", COLORS["cyan"], window)
    polish(ax, "Training reconstruction error", "Standardized-vector MSE")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    raw_and_smooth(ax, train, "score_prediction_kl", "Reward-time KL", COLORS["purple"], window)
    raw_and_smooth(ax, train, "ar_prediction_kl", "AR-update KL", COLORS["gold"], window)
    ax.scatter(
        heldout["step"], heldout["eval_prediction_kl"], color=COLORS["red"],
        marker="D", s=30, zorder=3, label="Held-out KL",
    )
    polish(ax, "Next-token prediction fidelity", "Prediction KL")
    ax.legend(frameon=False)

    ax = axes[2, 0]
    raw_and_smooth(ax, train, "valid_format_rate", "Valid format", COLORS["green"], window)
    raw_and_smooth(ax, train, "cap_hit_rate", "Token-cap hit", COLORS["red"], window)
    ax.scatter(
        heldout["step"], heldout["eval_valid_format_rate"], color=COLORS["green"],
        edgecolor="white", linewidth=0.7, s=35, zorder=4, label="Held-out valid",
    )
    ax.scatter(
        heldout["step"], heldout["eval_cap_hit_rate"], color=COLORS["red"],
        edgecolor="white", linewidth=0.7, s=35, zorder=4, label="Held-out cap hit",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    polish(ax, "Explanation format and truncation", "Rate")
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[2, 1]
    raw_and_smooth(ax, train, "reward_mean", "Mean reward", COLORS["orange"], window)
    polish(ax, "Training reward", "Reward")
    ax.legend(frameon=False)

    ax = axes[3, 0]
    raw_and_smooth(ax, train, "policy_kl_k2", "Policy KL", COLORS["purple"], window)
    raw_and_smooth(ax, train, "av_sft_anchor_loss", "AV SFT anchor CE", COLORS["gold"], window)
    polish(ax, "Policy drift and supervised anchor", "Loss / divergence")
    ax.legend(frameon=False)

    ax = axes[3, 1]
    raw_and_smooth(ax, train, "response_tokens_mean", "Response length", COLORS["slate"], window)
    ax2 = ax.twinx()
    ax2.plot(
        train["step"], rolling(train["ar_examples"], window),
        color=COLORS["cyan"], linewidth=2.0,
        label=f"Valid AR examples ({window}-step mean)",
    )
    ax2.set_ylabel("Valid AR examples per step")
    ax2.spines["top"].set_visible(False)
    polish(ax, "Generation length and AR training coverage", "Mean response tokens")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")

    save_figure(fig, output)


def heldout_detail(
    heldout: pd.DataFrame, output: Path, exhaustive_fve: float | None
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Held-out RL evaluations", fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(heldout.step, heldout.eval_vector_fve, "o-", color=COLORS["blue"], linewidth=2.3)
    if exhaustive_fve is not None:
        ax.axhline(exhaustive_fve, color=COLORS["green"], linestyle="--", linewidth=1.4)
    polish(ax, "Vector FVE", "FVE")

    ax = axes[0, 1]
    ax.plot(heldout.step, heldout.eval_prediction_kl, "o-", color=COLORS["purple"], label="Prediction KL")
    ax2 = ax.twinx()
    ax2.plot(heldout.step, heldout.eval_delta_cosine, "s-", color=COLORS["green"], label="Delta cosine")
    ax2.plot(heldout.step, heldout.eval_delta_norm_ratio, "^-", color=COLORS["gold"], label="Norm ratio")
    ax2.set_ylabel("Cosine / norm ratio")
    ax2.spines["top"].set_visible(False)
    polish(ax, "Prediction and vector fidelity", "Prediction KL")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False)

    ax = axes[1, 0]
    ax.plot(heldout.step, heldout.eval_total_loss, "o-", color=COLORS["blue"], label="Total")
    ax.plot(heldout.step, heldout.eval_selection_loss, "s-", color=COLORS["orange"], label="Selection")
    polish(ax, "Objective and format-aware selection", "Loss")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(heldout.step, heldout.eval_valid_format_rate, "o-", color=COLORS["green"], label="Valid")
    ax.plot(heldout.step, heldout.eval_cap_hit_rate, "s-", color=COLORS["red"], label="Cap hit")
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    polish(ax, "Generation contract", "Rate")
    ax.legend(frameon=False)

    save_figure(fig, output)


def training_detail(train: pd.DataFrame, output: Path, window: int) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    fig.suptitle(
        f"Per-step RL metrics (raw + trailing {window}-step mean)",
        fontsize=17,
        fontweight="bold",
    )
    specs = [
        ("reward_mean", "Mean reward", COLORS["orange"], "Reward", "Reward"),
        ("score_vector_mse", "Reward-time vector MSE", COLORS["blue"], "Vector reconstruction", "MSE"),
        ("score_prediction_kl", "Reward-time prediction KL", COLORS["purple"], "Prediction fidelity", "KL"),
        ("policy_kl_k2", "Policy KL", COLORS["purple"], "Policy drift", "KL"),
        ("av_sft_anchor_loss", "AV SFT anchor CE", COLORS["gold"], "Supervised anchor", "Cross-entropy"),
        ("response_tokens_mean", "Response length", COLORS["slate"], "Generated explanation length", "Tokens"),
    ]
    for ax, (key, label, color, title, ylabel) in zip(axes.flat, specs, strict=True):
        raw_and_smooth(ax, train, key, label, color, window)
        polish(ax, title, ylabel)
        ax.legend(frameon=False)
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--final-evaluation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()
    if args.window <= 0:
        raise ValueError("window must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, heldout = load_rows(args.metrics)
    exhaustive_fve = final_rl_fve(args.final_evaluation)
    dashboard(train, heldout, args.output_dir / "rl_training_dashboard.png", args.window, exhaustive_fve)
    heldout_detail(heldout, args.output_dir / "rl_heldout_curves.png", exhaustive_fve)
    training_detail(train, args.output_dir / "rl_per_step_curves.png", args.window)
    heldout.to_csv(args.output_dir / "heldout_metrics.csv", index=False)
    train.to_csv(args.output_dir / "per_step_metrics.csv", index=False)


if __name__ == "__main__":
    main()
