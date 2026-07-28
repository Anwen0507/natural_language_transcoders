# Tests

Checks for the natural-language **transcoder** (layer N → layer M) paths. See
[`docs/transcoder.md`](../docs/transcoder.md) for the design and
[`docs/transcoder_runbook.md`](../docs/transcoder_runbook.md) for the end-to-end run.

| File | Needs | What it checks |
|---|---|---|
| `test_transcoder.py` | nothing (CPU; no model/network) | **Unit tests** for the new logic: `stage_pair_target` keyed join (incl. chunked batches, duplicate target keys, empty input, every input-validation assert) + coverage assertion + N→M provenance; `transcoder_gold` (autoencoder / absolute / delta); `transcoder_delta_mode` env contract; `load_predict_mean_baselines` gold-column + delta branch + autoencoder auto-detect fallback; sidecar `target_layer_index` round-trip + old-sidecar back-compat; `train_actor`'s baseline call contract (AST-level); `injection_token_cache.yaml` shape. Builds its own tiny parquets. |
| `test_transcoder_training_paths.py` | nothing (CPU; no model/network — `miles`/`ray` are stubbed in sys.modules) | **Unit tests** for the transcoder wiring in the training/rollout paths: `NLADataSource` paired-parquet loading (target column via the numpy fast path into sample metadata); `sft_critic` critic gold per mode; `reward._prep_batch` gold ≡ training gold + `_lazy_init` delta caching; `nla_generate.generate` — actor always injects the scaled SOURCE `v_N` while the stashed RAW gold switches per mode, critic tokens attached, extraction-miss handling, `_lazy_init` delta announcement. |
| `test_openai_compat_provider.py` | nothing (CPU; no network — the httpx client is a scripted fake; `anthropic`/`httpx` stubbed if absent) | **Unit tests** for `OpenAICompatProvider` (keyless local-explainer path for stage 2): payload/auth-header construction (explicit key / `$OPENAI_API_KEY` / none), completion order + length contract, retry policy (408/429/5xx/transport → backoff → success or None-drop; other statuses raise), `content_filter`/empty drops vs `length` pass-through, `finish_reason` contract, semaphore concurrency bound, and the sidecar provenance attrs stage 2 reads. |
| `verify_paired_parquet.py` | a paired parquet (+ optional layer-M base) | Structure & provenance of a real `stage_pair_target` output, keyed-join cross-check, and the **identity baseline** MSE(v_N, v_M) — the bar a transcoder must beat — plus the predict-mean FVE denominators. |
| `check_critic_paths.py` | a critic checkpoint + GPU + Qwen2-capable transformers | The miles-free subset of `rl_preflight`: layer truncation, square value head, prediction scale, reward-path ≡ training-path MSE. **Complements** `rl_preflight` (which needs the training stack), it does not replace it. |

```bash
# Unit tests — run anywhere, CI-safe:
python tests/test_transcoder.py          # or: pytest tests/test_transcoder.py

# Against real artifacts (examples):
python tests/verify_paired_parquet.py --paired OUT/rl_xcoder.parquet --target-base OUT/M/base.parquet
python tests/check_critic_paths.py --critic-hf-dir OUT/critic_init_LM
```

> `check_critic_paths.py`'s path-equivalence uses transformers' `position_ids`
> packing (not miles' flash-attn varlen), so its tolerance is looser than
> `rl_preflight`'s production `1e-3`. It rules out the left-pad / packing **bug**
> (~0.5–1.0), not GEMM noise. Always run the real `rl_preflight` on the training
> box before an RL run.
