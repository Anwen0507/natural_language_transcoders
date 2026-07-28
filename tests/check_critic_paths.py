"""Miles-free critic checks — the structural part of rl_preflight that doesn't
need the training stack. rl_preflight imports nla.train_actor (→ miles); this
runs anywhere with transformers (Qwen2-capable) + torch + a GPU.

Validates a critic checkpoint (autoencoder OR transcoder — identical code):
  layers   num_hidden_layers == sidecar extraction_layer_index + 1
  head     value_head is square d×d
  scale    prediction per-dim norm > 0.1  (head not stuck at random init)
  paths    reward-path (padded batch) MSE == training-path (thd-packed) MSE

NOTE the 'paths' check is a STANDALONE REPLICA using transformers' position_ids
packing, not miles' flash-attn varlen, so its tolerance is looser than
rl_preflight's production 1e-3. It rules out the left-pad / packing-detection BUG
(ratio ~0.5–1.0), not bf16 GEMM noise. Always run the real rl_preflight on the
training box before an RL run.

    python tests/check_critic_paths.py --critic-hf-dir OUT/critic_init_LM
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nla.config import load_nla_config  # noqa: E402
from nla.models import NLACriticModel  # noqa: E402
from nla.schema import normalize_activation  # noqa: E402

_DUMMIES = [
    "explain: cat",
    "explain: the quick brown fox jumps over the lazy dog repeatedly",
    "explain: " + "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 3,
    "explain: singular",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--critic-hf-dir", required=True)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--bug-floor", type=float, default=0.1,
                   help="abort if reward/train MSE deviates by more than this (the bug regime)")
    args = p.parse_args()
    D = args.critic_hf_dir

    tok = AutoTokenizer.from_pretrained(D)
    cfg = load_nla_config(D, tok)
    d_model, mse_scale = cfg.d_model, cfg.mse_scale
    print(f"tokenizer={type(tok).__name__} padding_side={tok.padding_side!r}  "
          f"d_model={d_model} mse_scale={mse_scale:.3f}")

    n_layers = json.load(open(f"{D}/config.json"))["num_hidden_layers"]
    k = (yaml.safe_load(open(f"{D}/nla_meta.yaml")).get("critic") or {}).get("extraction_layer_index")
    assert k is not None and n_layers == k + 1, f"num_hidden_layers {n_layers} != extraction_layer_index {k} + 1"
    print(f"[layers] PASS  num_hidden_layers={n_layers} == extraction_layer_index={k} + 1")

    m = NLACriticModel.from_pretrained(D, torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()
    vh = tuple(m.value_head.weight.shape)
    assert vh == (d_model, d_model), f"value_head {vh} != ({d_model}, {d_model})"
    print(f"[head]   PASS  value_head square {vh}")

    n = len(_DUMMIES)
    gold = torch.randn(n, d_model)

    # reward path: padded batch, left/right-pad-safe last-token index
    enc = tok(_DUMMIES, add_special_tokens=True, padding=True, return_tensors="pt")
    ids, mask = enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device)
    last = mask.cumsum(1).argmax(1)
    with torch.no_grad():
        v = m(input_ids=ids, attention_mask=mask, use_cache=False).values
        pred_rwd = v[torch.arange(n, device=ids.device), last].float().cpu()

    # training path: thd packing (concat, position_ids reset, attention_mask=None)
    per = [tok(d, add_special_tokens=True, return_tensors="pt")["input_ids"][0] for d in _DUMMIES]
    lens = [int(x.shape[0]) for x in per]
    off = torch.tensor([0] + lens[:-1]).cumsum(0)
    packed = torch.cat(per).unsqueeze(0).to(args.device)
    pos = torch.cat([torch.arange(l) for l in lens]).unsqueeze(0).to(args.device)
    with torch.no_grad():
        v = m(input_ids=packed, position_ids=pos, attention_mask=None, use_cache=False).values
        pred_trn = v[0, (off + torch.tensor(lens) - 1).to(args.device)].float().cpu()

    ppd = pred_rwd.norm(dim=1).mean().item() / math.sqrt(d_model)
    assert ppd > 0.1, f"pred per-dim {ppd:.3f} <= 0.1 — head likely random-init"
    print(f"[scale]  PASS  pred per-dim (normalized) = {ppd:.3f} > 0.1")

    def mse(x):
        return ((normalize_activation(x, mse_scale) - normalize_activation(gold, mse_scale)) ** 2).mean(1)

    ratio = (mse(pred_rwd) / mse(pred_trn)).numpy()
    dev = abs(ratio - 1.0).max()
    assert dev < args.bug_floor, f"paths diverge like the bug ({dev:.2f}): {ratio}"
    tag = "tight (≈production)" if dev < 2e-3 else f"agree (replica-noise {dev:.1%} ≪ {args.bug_floor:.0%} bug floor)"
    print(f"[paths]  PASS  reward/train MSE max|r-1| = {dev:.2e}  {tag}")
    print("\ncritic checks passed")


if __name__ == "__main__":
    main()
