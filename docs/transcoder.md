# Natural-language transcoders (layer N → layer M)

A **natural-language transcoder** reads the residual stream at layer *N*,
verbalizes it to text, then reconstructs the residual stream at a *later* layer
*M* (M > N) from that text alone:

```
v_N  ──AV──▶  natural-language explanation  ──AR──▶  v̂_M  ≈  v_M
```

It is the mechanistic-interpretability *transcoder* idea
(<https://transformer-circuits.pub/2026/nla/> for the autoencoder it builds on)
expressed through a natural-language bottleneck: instead of reconstructing an
activation back to itself, the language description is optimized to carry exactly
the information about layer *N* needed to predict layer *M*. The bottleneck is
the explanation, so whatever computation the model performs between *N* and *M*
must be expressible in words for the reconstruction to succeed.

## Why this is a small change

NLA already factors into two halves bridged by language, and **neither half is
tied to a layer in its code** — the layer is just which vectors you feed it:

- **AV (actor / verbalizer)** injects an activation and generates an explanation.
  It is layer-agnostic (`train_actor.py` injects whatever `activation_vector`
  it's handed; `nla_generate.py` likewise). Reading layer *N* needs **no change**.
- **AR (critic / reconstructor)** is the base model truncated to `M+1` blocks
  with a `d×d` value head, predicting the residual stream at the extraction layer
  from the explanation text — it never sees the input vector. Writing layer *M*
  is just `prepare_critic_checkpoint --num-layers M`.

The autoencoder ties both halves to one layer *K*. The transcoder unties that
knot: AV reads *N*, AR writes *M*.

| Piece | Autoencoder (layer K) | Transcoder (N → M) | Code impact |
|---|---|---|---|
| AV / actor | reads & writes K | reads **N** | none (layer-agnostic) |
| AR / critic | `K+1` blocks, predicts K | `M+1` blocks, predicts **M** | `--num-layers M` only |
| AV-SFT data | `activation_vector` = v_K | `activation_vector` = **v_N** | ordinary single-layer build |
| AR-SFT data | gold = v_K | gold = **v_M** (or **v_M − v_N**) | + paired target column |
| RL data | one vector (inject = gold) | inject **v_N**, gold **v_M** | + paired target column |

The only genuinely new data is a **second raw vector per row** (`v_M` at the same
position as `v_N`) in the AR-SFT and RL parquets. Everything else is the existing
pipeline run at two layers.

## Two objectives: absolute vs. delta

Both modes are coded; pick at training time. Data is identical (both raw vectors)
— flip the env var, no rebuild.

| Mode | Critic gold | Reconstruct v_M as | Interpretation |
|---|---|---|---|
| **absolute** (default) | `v_M` | `AR(text)` | "describe v_N well enough to name the layer-M state" |
| **delta** (`NLA_TRANSCODER_DELTA=1`) | `v_M − v_N` | `v_N + AR(text)` | "describe what the layers between N and M *add*" — closest to a classic MLP/sublayer transcoder |

Both inherit the sidecar's `mse_scale`: with the default `sqrt_d_model`, pred and
gold are L2-normalized before MSE, so the objective is **direction-only** (MSE =
2(1−cos)). For delta this means "predict the *direction* the residual stream
moves." Set `extraction.mse_scale: null` in the critic sidecar for raw-magnitude
MSE if you want the critic to fit the delta's length too.

### Why a env var and not a CLI flag

The training entrypoint is `train.py` from **miles** (upstream — we don't add
argparse flags there). The mode must be read identically in four processes:
`NLADataSource` and `nla_generate` (RolloutManager), `nla_rm` (reward worker), and
the critic trainer's FVE baseline. An **exported** env var is inherited by every
Ray worker; a CLI flag would not reach them. All gold-producing sites route
through one helper (`schema.transcoder_gold` + `schema.transcoder_delta_mode`), so
the reward gold and the training gold cannot diverge — and if they somehow did,
`train_actor._assert_reward_train_paths_agree` fires at RL step 0.

The **target layer itself** *is* a CLI flag — on the datagen scripts, which are
ours: `stage0_extract --layer-index M` and `stage_pair_target --target-layer M`.

## Data pipeline

Run the standard pipeline once at *N*, extract once more at *M*, then attach the
*M* vectors to the AR-SFT and RL parquets. AV-SFT needs no pairing (the actor only
injects the source layer).

```bash
export PYTHONPATH=/path/to/natural_language_autoencoders:${PYTHONPATH:-}
MODEL=Qwen/Qwen2.5-7B-Instruct
CFG=configs/datagen/qwen7b_fineweb_1M.yaml
N=10; M=18                       # source / target layers — pick a MEANINGFUL gap
OUT=/tmp/xcoder

# 1) Full pipeline at the SOURCE layer N → av_sft / ar_sft / rl (all carry v_N
#    and N-context explanations — exactly what the actor will verbalize).
#    --override sets the top-level layer_index + output_dir for this run.
python -m nla.datagen.run_pipeline --config $CFG \
    --override output_dir=$OUT/N layer_index=$N

# 2) Stage 0 ONLY at the TARGET layer M, from the SAME config — identical corpus,
#    seed, and positions_per_doc, so the per-doc keyed RNG samples the IDENTICAL
#    (doc_id, n_raw_tokens) positions (stage0_extract._sample_positions). No
#    manual seed-matching to get wrong.
python -m nla.datagen.run_pipeline --config $CFG \
    --override output_dir=$OUT/M layer_index=$M --stages 0

# 3) Attach v_M to the critic-gold parquets (ar_sft + rl). Joins on
#    (doc_id, n_raw_tokens); asserts full coverage. av_sft is left untouched.
python -m nla.datagen.stage_pair_target \
    --input $OUT/N/ar_sft.parquet --target-base $OUT/M/base.parquet \
    --target-layer $M --output $OUT/ar_sft_xcoder.parquet
python -m nla.datagen.stage_pair_target \
    --input $OUT/N/rl.parquet --target-base $OUT/M/base.parquet \
    --target-layer $M --output $OUT/rl_xcoder.parquet

# 4) Critic init truncated to M+1 blocks (last_hidden_state in layer-M space).
python -m nla.scripts.prepare_critic_checkpoint \
    --base-model $MODEL --num-layers $M \
    --dataset-sidecar $OUT/ar_sft_xcoder.parquet --output $OUT/critic_init_LM
```

The paired parquets store **both vectors raw** (`norm="none"`), so the same file
drives absolute and delta. Provenance: the paired sidecar records
`extraction.layer_index = N` and `extraction.target_layer_index = M`.

> **Note on `activation_vector` in paired AR-SFT.** It holds **v_N** (the source),
> and the rows carry the *N-context* explanations the actor learns to produce — so
> the critic is trained on the same explanation distribution it will see at serve
> time. The gold is `target_activation_vector` (v_M), not `activation_vector`.

## Training

```bash
# AV-SFT — unchanged, single-layer (layer N). INJ_SCALE is calibrated to layer N.
AV_SFT_PARQUET=$OUT/N/av_sft.parquet  INSTRUCT_MODEL=$MODEL  INJ_SCALE=sqrt_d_model \
  SAVE_DIR=$OUT/actor_sft  bash configs/actor_sft.sh

# AR-SFT — paired parquet + M+1-layer critic init. NLA_TRANSCODER_DELTA optional.
AR_SFT_PARQUET=$OUT/ar_sft_xcoder.parquet  CRITIC_INIT_CKPT=$OUT/critic_init_LM \
  SAVE_DIR=$OUT/critic_sft  bash configs/critic_sft.sh
# delta variant:
NLA_TRANSCODER_DELTA=1  AR_SFT_PARQUET=$OUT/ar_sft_xcoder.parquet ... bash configs/critic_sft.sh

# RL — paired parquet; export the flag so Ray workers inherit it.
export NLA_TRANSCODER_DELTA=1   # or leave unset for absolute
RL_PARQUET=$OUT/rl_xcoder.parquet  INSTRUCT_MODEL=$MODEL \
  ACTOR_SFT_CKPT=$OUT/actor_sft/iter_XXXX  CRITIC_SL_CKPT=$OUT/critic_sft/iter_XXXX/hf \
  RUN_DIR=$OUT/rl  bash configs/rl.sh
```

To A/B the two modes, train two critics from the same paired data — one with the
flag, one without — and keep them paired with the same actor.

## Evaluation

Round-trip on held-out **paired** positions: inject v_N, get the AV explanation,
reconstruct, compare to the true v_M. With `nla_inference.py`
(`NLAClient.generate` + `NLACritic`):

```python
text = client.generate(v_N)                 # AV: v_N → explanation
pred = critic.reconstruct(text)             # AR: explanation → v̂  (raw)
v_M_hat = v_N + pred if delta else pred     # delta adds the source back
# direction MSE = 2(1 - cos); critic.score normalizes both to mse_scale
mse, cos = critic.score(text, v_M if not delta else (v_M - v_N))
```

Useful baselines: (a) **identity** `v̂_M = v_N` (how much does layer M differ from
layer N at all — the bar the transcoder must clear), and (b) **shuffled** target
(`stage_shuffle`-style) for a chance floor. `fve_nrm` logged during critic
training already uses the correct gold distribution (target or delta) as its
denominator.

## Code map

| File | Change |
|---|---|
| `nla/schema.py` | `TARGET_ACTIVATION_COLUMN`; `transcoder_delta_mode()`; `transcoder_gold(source,target,delta)`; `load_predict_mean_baselines(..., target_column, delta)` |
| `nla/datagen/stage_pair_target.py` | **new** — keyed join attaching `target_activation_vector` to an ar_sft/rl parquet |
| `nla/datagen/sidecar.py` | `NLAExtractionMeta.target_layer_index` |
| `nla/data_source.py` | read `target_activation_vector` as numpy → sample metadata |
| `nla/rollout/sft_critic.py` | critic-SFT gold via `transcoder_gold` |
| `nla/rollout/nla_generate.py` | RL online-critic gold via `transcoder_gold` (inject still = source) |
| `nla/reward.py` | reward gold via `transcoder_gold` (same helper → matches train) |
| `nla/train_actor.py` | FVE baseline on the gold distribution |

All changes are **backward-compatible**: with no `target_activation_vector`
column and `NLA_TRANSCODER_DELTA` unset, `transcoder_gold` returns the lone
activation and every path is the original autoencoder.

## Design notes / gotchas

- **Pick a meaningful gap.** Adjacent layers (M = N+1) differ by one block, so an
  N→N+1 transcoder is nearly an identity and shows little. Try N→N+4/N+8, or use
  **delta** mode, to force real computation through the language bottleneck.
- **The critic must be `M+1` blocks.** `prepare_critic_checkpoint --num-layers M`;
  `train_actor.py` asserts `num_hidden_layers == critic_num_layers + 1` at load.
- **`injection_scale` is calibrated to the source layer N.** Residual-stream norms
  grow with depth — set `--nla-injection-scale` from layer-N's norm (or
  `sqrt_d_model`). `mse_scale` is layer-agnostic (direction-only by default).
- **Export the env var.** `NLA_TRANSCODER_DELTA` must be *exported* so Ray workers
  inherit it; setting it inline before `train.py` without `export` only affects the
  driver. Absolute is the default.
- **Pairing is exact but memory-bounded by the input bucket.** The join keys on
  `(doc_id, n_raw_tokens)` and asserts full coverage, so a seed/corpus mismatch
  fails loud. The target lookup holds only the ar_sft/rl bucket's vectors; for very
  large buckets, pair per-shard and concatenate.
