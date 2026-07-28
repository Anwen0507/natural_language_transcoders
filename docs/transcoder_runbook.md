# Transcoder runbook (layer N → M), end-to-end

The ordered, copy-pasteable version of [`docs/transcoder.md`](transcoder.md) for
an actual run. Read that for the *why*, [`docs/setup.md`](setup.md) for install
detail, and [`configs/TRAINING_NOTES.md`](../configs/TRAINING_NOTES.md) for the
Qwen2.5-7B hyperparameter case study. Real SFT/RL needs a multi-GPU box — a
single GPU is enough for datagen + the preflight checks only.

## 0. Prerequisites

- **GPUs** — Qwen2.5-7B reference: SFT on 2×H100-80GB, RL on 2×8×H100. Smaller
  works (see GPU layout in §6); one GPU is not enough for RL.
- **`ANTHROPIC_API_KEY`** — stage-2 generates the AV explanations (the `av_sft`
  and `ar_sft` data). The `rl` split needs no API. **No key?** Stage 2 is
  provider-pluggable: `nla.datagen.providers.OpenAICompatProvider` speaks to any
  OpenAI-compatible endpoint — a local explainer (`vllm serve
  Qwen/Qwen2.5-14B-Instruct`, zero keys) or a hosted one (OpenAI / Gemini
  compat, `$OPENAI_API_KEY`). Set `stage2.provider_cls` + `provider_kwargs`
  (`base_url`, `model`) in the datagen config.
- **`HF_TOKEN`** — only for gated bases (Gemma-3, Llama-3.3); Qwen2.5 is ungated.
- **`/dev/shm` ≥ 8 GB** — RL writes ~1 GB/step of embed dumps there (run with
  `--shm-size=8g`, or point `NLA_EMBED_DUMP_DIR` at a disk path).

## 1. Install the stack (no-conda venv path)

```bash
python -m venv .venv && . .venv/bin/activate && pip install -U uv
export NLA_REPO=$PWD/natural_language_autoencoders
# miles (pinned commit) + integration patches + flash-attn
git clone https://github.com/radixark/miles && cd miles
git checkout $(cut -d@ -f2 $NLA_REPO/nla/miles_patches/UPSTREAM_PIN)
uv pip install -e . && git apply $NLA_REPO/nla/miles_patches/*.patch
uv pip install flash-attn --no-build-isolation && cd ..
# SGLang (patched from source — needed for the training rollout transport)
git clone https://github.com/sgl-project/sglang
bash $NLA_REPO/patches/apply_sglang_patches.sh ./sglang
uv pip install -e "./sglang/python[all]"
# this package
uv pip install -e "$NLA_REPO"
python -c "import miles, sglang, nla; print('ok')"
```

`docs/setup.md` has the conda variant, the cu124 torch pin, and the Megatron
backend (Llama-70B only). For Gemma-3 launch SGLang with `--attention-backend fa3`.

## 2. Pick the experiment

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct ; N=16 ; M=20 ; OUT=/data/xcoder
export NLA_TRANSCODER_DELTA=1     # delta: gold = v_M − v_N (this run). UNSET for absolute v_M
```

- **N=16 → M=20 (locked, from measured Qwen2.5-7B geometry):** cos(v₁₆, v₂₀)=0.77
  — the four blocks do real work (‖Δ‖ ≈ ‖v_N‖), both layers sit in the rich band,
  and M=20 is the released NLA layer (a target known to be language-reconstructible).
  Avoid M ≥ 25 (the last block collapses toward the unembedding, cos(v₂₆, v₂₇)=0.39)
  and gaps ≤ 3 (near-identity). Re-verify on real data with §4 before training.
- **absolute vs delta** — `NLA_TRANSCODER_DELTA` must be **`export`ed** so every
  Ray worker (rollout / reward / critic) *and* the FVE baseline read the same
  mode. Setting it inline (un-exported) silently desyncs the FVE denominator from
  the gold; the RL step-0 assert won't catch that. A/B both from one paired dataset.

## 3. Data — full pipeline at N, stage-0 at M, then pair

```bash
# Source layer N → av_sft / ar_sft / rl (carry v_N + N-context explanations)
python -m nla.datagen.run_pipeline --config configs/datagen/qwen7b_fineweb_1M.yaml \
    --override output_dir=$OUT/N layer_index=$N
# Target layer M, stage-0 only — SAME config ⇒ per-doc keyed RNG samples the
# IDENTICAL (doc_id, position) set (no manual seed-matching).
python -m nla.datagen.run_pipeline --config configs/datagen/qwen7b_fineweb_1M.yaml \
    --override output_dir=$OUT/M layer_index=$M --stages 0
# Attach v_M to the critic-gold parquets (ar_sft + rl); av_sft is untouched.
for S in ar_sft rl; do
  python -m nla.datagen.stage_pair_target \
      --input $OUT/N/$S.parquet --target-base $OUT/M/base.parquet \
      --target-layer $M --output $OUT/${S}_xcoder.parquet
done
```

## 4. Verify the pairing BEFORE training (cheap, CPU)

```bash
python tests/verify_paired_parquet.py \
    --paired $OUT/rl_xcoder.parquet --target-base $OUT/M/base.parquet
```

Confirms full coverage + correct keying, and prints the **identity baseline**
MSE(v_N, v_M) — the bar the transcoder must beat. If cos(v_N, v_M) ≈ 1.0 the gap
is too small: pick a larger M−N or use delta.

## 5. Critic init (M+1 layers) + preflight

```bash
python -m nla.scripts.prepare_critic_checkpoint \
    --base-model $MODEL --num-layers $M \
    --dataset-sidecar $OUT/ar_sft_xcoder.parquet --output $OUT/critic_init_LM
python tests/check_critic_paths.py --critic-hf-dir $OUT/critic_init_LM   # miles-free smoke
# real preflight, on the training stack, before RL:
python -m nla.scripts.rl_preflight --critic-hf-dir $OUT/critic_init_LM --actor-hf-dir <av_sft_ckpt>
```

`--num-layers M` is the **target** layer; the script anchors the critic to M
(`num_hidden_layers = M+1`, sidecar `extraction_layer_index = M`) regardless of
the source N in the paired sidecar.

## 6. Train — AR-SFT, AV-SFT, RL

```bash
# AR-SFT (critic): paired parquet + M+1-layer init. Honours NLA_TRANSCODER_DELTA.
AR_SFT_PARQUET=$OUT/ar_sft_xcoder.parquet CRITIC_INIT_CKPT=$OUT/critic_init_LM \
  SAVE_DIR=$OUT/critic_sft  bash configs/critic_sft.sh
# AV-SFT (actor): unchanged single-layer (N). INJ_SCALE is calibrated to layer N.
AV_SFT_PARQUET=$OUT/N/av_sft.parquet INSTRUCT_MODEL=$MODEL INJ_SCALE=sqrt_d_model \
  SAVE_DIR=$OUT/actor_sft  bash configs/actor_sft.sh
# RL — paired parquet; export the flag so Ray workers inherit it (see §2).
RL_PARQUET=$OUT/rl_xcoder.parquet INSTRUCT_MODEL=$MODEL \
  ACTOR_SFT_CKPT=$OUT/actor_sft/iter_XXXX CRITIC_SL_CKPT=$OUT/critic_sft/iter_XXXX/hf \
  RUN_DIR=$OUT/rl  bash configs/rl.sh
```

On a single 8-GPU node, RL defaults (8+4+4) exceed devices — set
`ACTOR_GPUS=4 CRITIC_GPUS=2 ROLLOUT_GPUS=2` or Ray hangs on placement.

## 7. Evaluate

Round-trip on held-out **paired** positions (`nla_inference.py`): inject v_N → AV
explanation → AR reconstruct → compare to true v_M (or v_M−v_N for delta). Compare
against (a) **identity** `v̂_M = v_N` (the §4 bar) and (b) a **shuffled** target
floor. `fve_nrm` logged during critic training already uses the correct gold.

## Gotchas (the ones that silently cost a run)

- **`export NLA_TRANSCODER_DELTA`** — not just set. Un-exported desyncs the FVE
  denominator from the gold.
- **`INJ_SCALE` / `--nla-injection-scale` is keyed to the SOURCE layer N** —
  residual norms grow with depth (dry-run: ‖v_10‖≈15 vs ‖v_18‖≈36). `sqrt_d_model`
  is the safe default.
- **Grep generated text for CJK** — the loudest injection-failure smoke test
  (see `CLAUDE.md` / `docs/inference.md`).
- **The critic must be M+1 blocks** — `prepare_critic_checkpoint --num-layers M`;
  `train_actor` asserts `num_hidden_layers == critic_num_layers + 1` at load.
