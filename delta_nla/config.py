"""Configuration loading and provenance helpers for the full-delta experiment."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("schema_version") != 1:
        raise ValueError(f"unsupported config schema: {cfg.get('schema_version')!r}")
    cfg["_config_path"] = str(path)
    cfg["run_dir"] = str(Path(cfg["run_dir"]).expanduser().resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    quotas = cfg["data"]["quotas"]
    if set(quotas) != {"av_sft", "ar_sft", "rl", "eval"}:
        raise ValueError("data.quotas must contain av_sft, ar_sft, rl, and eval")
    if any(int(v) <= 0 for v in quotas.values()):
        raise ValueError("all split quotas must be positive")
    if cfg["rl"]["group_size"] < 2:
        raise ValueError("GRPO group_size must be at least two")
    if cfg["rl"]["prompts_per_step"] * cfg["rl"]["max_steps"] > quotas["rl"]:
        raise ValueError("RL quota cannot supply unique prompts for every configured step")
    if cfg["delta"]["injection_alpha"] <= 0:
        raise ValueError("injection_alpha must be positive")


def run_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["run_dir"])


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_revision(repo: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def write_run_manifest(cfg: dict[str, Any]) -> None:
    root = run_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    config_text = Path(cfg["_config_path"]).read_text()
    manifest = {
        "experiment_name": cfg["experiment_name"],
        "config_path": cfg["_config_path"],
        "config_sha256": sha256_text(config_text),
        "git_revision": git_revision(Path(__file__).resolve().parents[1]),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    atomic_json(root / "manifest.json", manifest)
    destination = root / "config.lock.yaml"
    if destination.exists() and destination.read_text() != config_text:
        raise RuntimeError(
            f"locked run config differs from {cfg['_config_path']}; use a new run_dir"
        )
    destination.write_text(config_text)
