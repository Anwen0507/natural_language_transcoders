"""Verify a transcoder paired parquet + report the identity baseline.

A paired parquet (stage_pair_target output) carries `activation_vector` (v_N, the
SOURCE the actor injects) and `target_activation_vector` (v_M, the critic GOLD).
This checks structure + provenance and — given the layer-M base parquet — that
the join is keyed correctly on real data. It then prints the identity baseline
MSE(v_N, v_M) (the bar a transcoder must beat) and the predict-mean FVE
denominators for both absolute and delta.

    python tests/verify_paired_parquet.py --paired OUT/rl_xcoder.parquet \
        [--target-base OUT/M/base.parquet]
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nla.datagen.sidecar import read_sidecar  # noqa: E402
from nla.schema import (  # noqa: E402
    ACTIVATION_COLUMN, TARGET_ACTIVATION_COLUMN, load_predict_mean_baselines,
    normalize_activation,
)
from nla.storage import LocalStorage  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paired", required=True, help="stage_pair_target output parquet")
    p.add_argument("--target-base", default=None,
                   help="layer-M base parquet; if given, cross-checks the join keying")
    args = p.parse_args()
    st = LocalStorage()

    meta = read_sidecar(st, args.paired)
    d_model, n_src, m_tgt = meta.extraction.d_model, meta.extraction.layer_index, meta.extraction.target_layer_index
    assert meta.extraction.norm == "none", f"expected raw vectors, got norm={meta.extraction.norm!r}"
    assert m_tgt is not None, "sidecar has no target_layer_index — not a paired (transcoder) parquet"
    assert m_tgt != n_src, f"source == target layer ({n_src}); that's an autoencoder"

    t = pq.read_table(args.paired)
    assert TARGET_ACTIVATION_COLUMN in t.column_names, "missing target_activation_vector column"
    vN = np.array(t[ACTIVATION_COLUMN].to_pylist(), np.float32)
    vM = np.array(t[TARGET_ACTIVATION_COLUMN].to_pylist(), np.float32)
    assert vN.shape == vM.shape == (t.num_rows, d_model), (vN.shape, vM.shape)
    print(f"OK  {t.num_rows} rows  d_model={d_model}  N(source)={n_src} → M(target)={m_tgt}  norm=none")

    if args.target_base:
        mb = pq.read_table(args.target_base)
        idx = {k: i for i, k in enumerate(zip(mb["doc_id"].to_pylist(), mb["n_raw_tokens"].to_pylist()))}
        mv = np.array(mb[ACTIVATION_COLUMN].to_pylist(), np.float32)
        for i, k in enumerate(zip(t["doc_id"].to_pylist(), t["n_raw_tokens"].to_pylist())):
            assert k in idx, f"row {k} absent from target base (coverage hole)"
            assert np.allclose(vM[i], mv[idx[k]], atol=1e-5), f"target mismatch at {k}"
        print(f"OK  cross-check: every target == layer-{m_tgt} base activation at the same position")

    tN, tM = torch.tensor(vN), torch.tensor(vM)
    cos = torch.nn.functional.cosine_similarity(tN, tM, dim=1)
    sd = float(np.sqrt(d_model))
    dir_mse = ((normalize_activation(tM, sd) - normalize_activation(tN, sd)) ** 2).mean().item()
    print("\nidentity baseline  v̂_M = v_N  (the bar the transcoder must clear):")
    print(f"  cos(v_N, v_M): mean={cos.mean():.3f}  [min {cos.min():.3f}, max {cos.max():.3f}]")
    print(f"  direction MSE = 2(1−cos) = {dir_mse:.3f}   raw MSE = {((tM - tN) ** 2).mean():.3f}")
    print(f"  ‖v_N‖={tN.norm(dim=1).mean():.1f}  ‖v_M‖={tM.norm(dim=1).mean():.1f}")
    for name, sc in (("sqrt_d", sd), ("raw", None)):
        _, a = load_predict_mean_baselines(args.paired, sc, target_column=TARGET_ACTIVATION_COLUMN, delta=False)
        _, dl = load_predict_mean_baselines(args.paired, sc, target_column=TARGET_ACTIVATION_COLUMN, delta=True)
        print(f"  predict-mean baseline [{name}]: absolute(v_M)={a:.4f}  delta(v_M−v_N)={dl:.4f}")
    print("\nverification passed")


if __name__ == "__main__":
    main()
