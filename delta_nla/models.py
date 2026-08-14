"""AV injection, full-depth AR, and the frozen target readout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoModelForCausalLM


def _input_embedding(model: nn.Module) -> nn.Module:
    embed = model.get_input_embeddings()
    if embed is None:
        raise RuntimeError(f"{type(model).__name__} has no input embeddings")
    return embed


def inject_vectors(
    model: nn.Module,
    input_ids: torch.Tensor,
    vectors: torch.Tensor,
    injection_token_id: int,
    alpha: float,
) -> torch.Tensor:
    """Replace exactly one marked embedding per row with fixed-alpha vectors."""
    sites = input_ids.eq(injection_token_id)
    counts = sites.sum(dim=1)
    if not torch.equal(counts, torch.ones_like(counts)):
        raise RuntimeError(f"expected one injection site per row, got {counts.tolist()}")
    embeds = _input_embedding(model)(input_ids)
    if vectors.shape != (input_ids.shape[0], embeds.shape[-1]):
        raise ValueError(
            f"vector shape {tuple(vectors.shape)} incompatible with "
            f"batch={input_ids.shape[0]}, width={embeds.shape[-1]}"
        )
    embeds = embeds.clone()
    embeds[sites] = (float(alpha) * vectors).to(embeds.device, embeds.dtype)
    return embeds


class DeltaReconstructor(nn.Module):
    """Complete 24-block Qwen backbone, no final RMSNorm, plus affine vector head."""

    def __init__(self, backbone: nn.Module, d_model: int):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(d_model, d_model, bias=True)
        with torch.no_grad():
            self.value_head.weight.copy_(torch.eye(d_model))
            self.value_head.bias.zero_()

    @classmethod
    def from_base(
        cls,
        model_name: str,
        *,
        revision: str = "main",
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> "DeltaReconstructor":
        backbone = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        if not hasattr(backbone, "norm"):
            raise RuntimeError(f"{type(backbone).__name__} has no final norm attribute")
        d_model = int(backbone.config.hidden_size)
        # The vector head consumes the raw output of the last transformer block.
        backbone.norm = nn.Identity()
        return cls(backbone, d_model).to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        indices = attention_mask.long().sum(dim=1) - 1
        if (indices < 0).any():
            raise RuntimeError("AR received an empty token sequence")
        batch = torch.arange(input_ids.shape[0], device=input_ids.device)
        hidden = out.last_hidden_state[batch, indices]
        return self.value_head(hidden), hidden

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        base_model: str,
        revision: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors = {
            key: value.detach().cpu().contiguous()
            for key, value in self.state_dict().items()
        }
        tmp = path / "model.safetensors.tmp"
        save_file(tensors, str(tmp))
        os.replace(tmp, path / "model.safetensors")
        payload = {
            "architecture": "DeltaReconstructor",
            "base_model": base_model,
            "revision": revision,
            "d_model": int(self.value_head.in_features),
            "full_depth": True,
            "final_norm": "Identity",
            "head": "Linear(d_model,d_model,bias=True)",
            **(metadata or {}),
        }
        tmp_json = path / "delta_ar_config.json.tmp"
        tmp_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_json, path / "delta_ar_config.json")

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> "DeltaReconstructor":
        path = Path(path)
        metadata = json.loads((path / "delta_ar_config.json").read_text())
        model = cls.from_base(
            metadata["base_model"],
            revision=metadata.get("revision", "main"),
            dtype=dtype,
            device="cpu",
        )
        state = load_file(str(path / "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"AR checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        return model.to(device)


class TargetProjection(nn.Module):
    """Frozen Qwen final RMSNorm and unembedding, without loading the target LM."""

    def __init__(self, norm_weight: torch.Tensor, unembedding_weight: torch.Tensor, eps: float):
        super().__init__()
        self.register_buffer("norm_weight", norm_weight.float())
        self.register_buffer("unembedding_weight", unembedding_weight.to(torch.bfloat16))
        self.eps = float(eps)

    @classmethod
    def load(cls, run_dir: str | Path, device: str | torch.device = "cuda") -> "TargetProjection":
        root = Path(run_dir) / "artifacts"
        values = load_file(str(root / "target_projection.safetensors"))
        metadata = json.loads((root / "target_projection.json").read_text())
        return cls(
            values["norm_weight"], values["unembedding_weight"], metadata["rms_norm_eps"]
        ).to(device).eval()

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        x = residual.float()
        variance = x.square().mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x * self.norm_weight
        return F.linear(x.to(torch.bfloat16), self.unembedding_weight).float()


def load_actor(
    model_name_or_path: str | Path,
    *,
    revision: str = "main",
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cuda",
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        str(model_name_or_path),
        revision=revision if not Path(str(model_name_or_path)).exists() else None,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model.to(device)
