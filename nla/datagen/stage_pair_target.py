"""Transcoder pairing: attach the TARGET-layer activation to a built parquet.

A natural-language transcoder reads layer N and reconstructs layer M (M > N).
The actor/AV side is unchanged — it only verbalizes layer-N activations, so its
av_sft parquet is an ordinary single-layer (layer N) build with no target. The
critic/AR side must reconstruct layer M, so its training rows need the layer-M
activation at the SAME (doc_id, position) as the layer-N vector the actor saw.

This stage takes a built ar_sft / rl parquet (activation_vector = v_N, the
SOURCE) and a Stage-0 base parquet extracted at the TARGET layer M (v_M at every
sampled position), and writes a copy with one extra column:

    target_activation_vector   list[float32]   RAW v_M at the same position

Both vectors stay RAW (norm="none") — data-gen never normalizes or transforms,
so a SINGLE paired parquet serves BOTH objectives; absolute (gold = v_M) vs.
delta (gold = v_M − v_N) is a training-time toggle (NLA_TRANSCODER_DELTA). See
docs/transcoder.md.

Why this join is exact: Stage-0 samples positions with an RNG keyed on
(seed, doc_id) (stage0_extract._sample_positions), so two runs over the same
corpus/seed at different layers sample the IDENTICAL (doc_id, n_raw_tokens)
positions. We still join on the key (never on row order) and assert full
coverage, so a seed/corpus mismatch fails loud instead of silently mis-pairing.

Memory: the target lookup holds only the |input| positions (the ar_sft or rl
bucket), one contiguous fp32 array — |input| × d_model × 4 bytes — not the full
target corpus. For very large buckets, pair per-shard and concatenate.

Usage:
    python -m nla.datagen.stage_pair_target \
        --input        OUT_N/rl.parquet \
        --target-base  OUT_M/base.parquet \
        --target-layer M \
        --output       OUT/rl_transcoder.parquet
"""

import argparse
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import add_storage_args, make_storage
from nla.datagen.sidecar import read_sidecar, write_sidecar
from nla.schema import ACTIVATION_COLUMN, TARGET_ACTIVATION_COLUMN

_CHUNK_SIZE = 16384


def _keys(batch: pa.RecordBatch) -> list[tuple[str, int]]:
    # doc_id + n_raw_tokens uniquely identify a sampled position. Small columns
    # (string + int) → to_pylist is fine; the heavy vector column never does.
    return list(zip(
        batch.column("doc_id").to_pylist(),
        batch.column("n_raw_tokens").to_pylist(),
    ))


def _vectors(batch: pa.RecordBatch, column: str, d_model: int) -> np.ndarray:
    col = batch.column(column)
    flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    return flat.reshape(len(col), d_model)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="built ar_sft or rl parquet (activation_vector = SOURCE layer N)")
    p.add_argument("--target-base", required=True,
                   help="Stage-0 base parquet extracted at the TARGET layer M")
    p.add_argument("--target-layer", type=int, required=True,
                   help="TARGET layer index M — asserted against --target-base's extraction layer")
    p.add_argument("--output", required=True, help="output parquet path")
    add_storage_args(p)
    args = p.parse_args()

    storage = make_storage(args)

    in_meta = read_sidecar(storage, args.input)
    tgt_meta = read_sidecar(storage, args.target_base)

    assert in_meta.stage in ("ar_sft", "rl"), (
        f"--input must be an ar_sft or rl parquet (the critic-gold stages), got "
        f"stage={in_meta.stage!r}. av_sft needs no target — the actor only injects "
        f"the source layer."
    )
    assert in_meta.extraction.norm == "none" and tgt_meta.extraction.norm == "none", (
        f"both inputs must hold RAW vectors (norm='none'); got input="
        f"{in_meta.extraction.norm!r} target={tgt_meta.extraction.norm!r}. "
        f"Normalization is a training-side decision."
    )
    assert in_meta.extraction.base_model == tgt_meta.extraction.base_model, (
        f"base_model mismatch: input={in_meta.extraction.base_model!r} "
        f"target={tgt_meta.extraction.base_model!r}. Source and target layers must "
        f"come from the SAME model (one residual stream)."
    )
    assert in_meta.extraction.d_model == tgt_meta.extraction.d_model, (
        f"d_model mismatch: input={in_meta.extraction.d_model} "
        f"target={tgt_meta.extraction.d_model}."
    )
    assert tgt_meta.extraction.layer_index == args.target_layer, (
        f"--target-layer {args.target_layer} != --target-base extraction layer "
        f"{tgt_meta.extraction.layer_index}. Point --target-base at the layer-M base.parquet."
    )
    assert args.target_layer != in_meta.extraction.layer_index, (
        f"target layer {args.target_layer} == source layer "
        f"{in_meta.extraction.layer_index}: that's an autoencoder, not a transcoder. "
        f"Use the ordinary single-layer pipeline."
    )
    d_model = in_meta.extraction.d_model
    print(f"pairing {in_meta.stage}: source L{in_meta.extraction.layer_index} → target L{args.target_layer}")

    # 1) Which (doc_id, n_raw_tokens) positions does the input need a target for?
    in_pf = pq.ParquetFile(storage.open_read(args.input))
    needed: set[tuple[str, int]] = set()
    for batch in in_pf.iter_batches(batch_size=_CHUNK_SIZE, columns=["doc_id", "n_raw_tokens"]):
        needed.update(_keys(batch))

    # 2) Pull just those vectors from the target base. Bounded by |needed| (the
    #    input bucket), not the full target corpus: one contiguous array + a
    #    key→row index. Early-exit once every needed key is found.
    tgt_pf = pq.ParquetFile(storage.open_read(args.target_base))
    index: dict[tuple[str, int], int] = {}
    chunks: list[np.ndarray] = []
    filled = 0
    for batch in tgt_pf.iter_batches(batch_size=_CHUNK_SIZE):
        keys = _keys(batch)
        hits = [j for j, k in enumerate(keys) if k in needed and k not in index]
        if hits:
            chunks.append(_vectors(batch, ACTIVATION_COLUMN, d_model)[hits])
            for j in hits:
                index[keys[j]] = filled
                filled += 1
            if len(index) == len(needed):
                break
    missing = len(needed) - len(index)
    assert missing == 0, (
        f"{missing}/{len(needed)} input positions have no target-layer vector in "
        f"{args.target_base!r}. The target base must cover the same (doc_id, position) "
        f"set — re-extract layer {args.target_layer} with the SAME --seed, --corpus, "
        f"--corpus-start/--corpus-length and --positions-per-doc as the source run."
    )
    target_lookup = (
        np.concatenate(chunks, axis=0) if chunks else np.zeros((0, d_model), np.float32)
    )

    # 3) Stream the input through, appending target_activation_vector per row.
    out_schema = in_pf.schema_arrow.append(
        pa.field(TARGET_ACTIVATION_COLUMN, pa.list_(pa.float32(), d_model))
    )
    storage.ensure_parent(args.output)
    row_count = 0
    with pq.ParquetWriter(storage.open_write(args.output), out_schema) as writer:
        for batch in in_pf.iter_batches(batch_size=_CHUNK_SIZE):
            keys = _keys(batch)
            rows = (
                np.stack([target_lookup[index[k]] for k in keys])
                if keys else np.zeros((0, d_model), np.float32)
            )
            tgt_arr = pa.FixedSizeListArray.from_arrays(
                pa.array(rows.reshape(-1), type=pa.float32()), d_model
            )
            out = {name: batch.column(name) for name in batch.schema.names}
            out[TARGET_ACTIVATION_COLUMN] = tgt_arr
            writer.write_table(pa.table(out, schema=out_schema))
            row_count += batch.num_rows

    out_meta = replace(
        in_meta,
        dataset_id=f"{in_meta.dataset_id}__xcoder_L{in_meta.extraction.layer_index}_to_L{args.target_layer}",
        extraction=replace(in_meta.extraction, target_layer_index=args.target_layer),
        row_count=row_count,
        parent_datasets=[in_meta.dataset_id, tgt_meta.dataset_id],
        created_by="nla.datagen.stage_pair_target",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)
    print(f"wrote {row_count} paired rows → {args.output}")
    print(f"  source layer N={in_meta.extraction.layer_index} → target layer M={args.target_layer}")
    print(f"  sidecar → {args.output}.nla_meta.yaml")


if __name__ == "__main__":
    main()
