"""Injected AV generation and response-token policy log probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from delta_nla.models import inject_vectors
from delta_nla.prompts import parse_explanation


@dataclass
class GeneratedBatch:
    full_ids: torch.Tensor
    full_attention_mask: torch.Tensor
    response_ids: torch.Tensor
    response_mask: torch.Tensor
    raw_text: list[str]
    explanations: list[str]
    valid_format: torch.Tensor
    cap_hit: torch.Tensor
    prefix_length: int


def _eos_set(model, tokenizer) -> set[int]:
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        return {eos}
    return {int(x) for x in eos}


@torch.no_grad()
def generate_actor(
    model,
    tokenizer,
    prefix_ids: Sequence[int],
    vectors: torch.Tensor,
    injection_token_id: int,
    alpha: float,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> GeneratedBatch:
    model.eval()
    device = next(model.parameters()).device
    batch = vectors.shape[0]
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1)
    prefix_mask = torch.ones_like(prefix)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        embeds = inject_vectors(model, prefix, vectors, injection_token_id, alpha)
        kwargs = {
            "input_ids": prefix,
            "inputs_embeds": embeds,
            "attention_mask": prefix_mask,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "use_cache": True,
            "return_dict_in_generate": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": model.generation_config.eos_token_id or tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = float(temperature)
            kwargs["top_p"] = float(top_p)
        output = model.generate(**kwargs)
    sequences = output.sequences
    prefix_length = len(prefix_ids)
    if sequences.shape[1] < prefix_length:
        raise RuntimeError(f"generate returned shape {tuple(sequences.shape)} shorter than prompt")
    if not torch.equal(sequences[:, :prefix_length], prefix):
        raise RuntimeError(
            "generate(input_ids + inputs_embeds) did not preserve bookkeeping prefix; "
            "the installed transformers version is incompatible with injected generation"
        )
    response = sequences[:, prefix_length:]
    eos_ids = _eos_set(model, tokenizer)
    mask = torch.zeros_like(response, dtype=torch.bool)
    cap_hit = torch.ones(batch, dtype=torch.bool, device=device)
    raw_text: list[str] = []
    explanations: list[str] = []
    valid: list[bool] = []
    for i in range(batch):
        length = response.shape[1]
        for j, token in enumerate(response[i].tolist()):
            if token in eos_ids:
                length = j + 1
                cap_hit[i] = False
                break
        mask[i, :length] = True
        decoded = tokenizer.decode(response[i, :length], skip_special_tokens=True).strip()
        explanation, is_valid = parse_explanation(decoded)
        raw_text.append(decoded)
        explanations.append(explanation)
        valid.append(is_valid)
    full_mask = torch.cat([prefix_mask.bool(), mask], dim=1)
    return GeneratedBatch(
        full_ids=sequences,
        full_attention_mask=full_mask,
        response_ids=response,
        response_mask=mask,
        raw_text=raw_text,
        explanations=explanations,
        valid_format=torch.tensor(valid, dtype=torch.bool, device=device),
        cap_hit=cap_hit,
        prefix_length=prefix_length,
    )


def response_log_probs(
    model,
    full_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    vectors: torch.Tensor,
    injection_token_id: int,
    alpha: float,
    prefix_length: int,
) -> torch.Tensor:
    """Log p(response token) without materializing prompt-position vocabulary logits."""
    # Only the marker in the fixed prefix is an injection site. If the policy
    # happens to emit the same Unicode token later, that response token must
    # retain its ordinary learned embedding rather than becoming a second
    # vector site (or tripping the exact-one-site assertion).
    prefix_embeds = inject_vectors(
        model, full_ids[:, :prefix_length], vectors, injection_token_id, alpha
    )
    response_embeds = model.get_input_embeddings()(full_ids[:, prefix_length:])
    embeds = torch.cat([prefix_embeds, response_embeds], dim=1)
    transformer = getattr(model, "model", None)
    if transformer is None:
        raise RuntimeError(f"unsupported AV wrapper {type(model).__name__}: no .model")
    hidden = transformer(
        inputs_embeds=embeds,
        attention_mask=full_attention_mask.long(),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    response_length = response_ids.shape[1]
    # Hidden at prefix_length-1 predicts response token 0.
    predictive = hidden[:, prefix_length - 1 : prefix_length - 1 + response_length]
    logits = model.get_output_embeddings()(predictive)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    selected = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
    return selected.masked_fill(~response_mask, 0.0)
