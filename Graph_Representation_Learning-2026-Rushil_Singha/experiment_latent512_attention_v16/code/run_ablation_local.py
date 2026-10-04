#!/usr/bin/env python
"""
SCRIPT 1 — "LOCAL" ARM  (NO global attention anywhere)
======================================================

This is the PRIMARY experiment.  Every global-mixing shortcut is removed, so
the ONLY path by which two distant particles can influence each other is the
encoder's message passing.  Whatever long-range structure survives the
512-d bottleneck therefore had to come from the receptive field.

REMOVED                          REPLACED WITH
-------                          -------------
Set2Set pooling (LSTM+attention)  StatsPool: [sum | mean | max | std] -> MLP
Decoder TransformerEncoder        Per-slot FiLM residual MLP (3 blocks)

Why StatsPool is the right replacement:
  * permutation invariant (a jet is an unordered set)
  * handles variable multiplicity (sum keeps "how much", mean keeps "what
    kind", max keeps the leading/hardest particle, std keeps the spread)
  * it is a FIXED symmetric function — no learned node-to-node routing, so it
    cannot manufacture correlations the convolutions did not already build
  * 4 x 256 = 1024 raw statistics -> projected to the SAME 512-d bottleneck
    used by the Set2Set arm, so the two arms are directly comparable

THREE RUNS (same kNN graph, same everything else):
Receptive field = layers x (K-1) hops, with layers = 2:
  1. sage       strictly 1-hop        -> 1 hop/layer  ->  2 hops total
  2. cheb K=4   K-1 = 3 hops/layer    -> 3 hops/layer ->  6 hops total
  3. cheb K=8   K-1 = 7 hops/layer    -> 7 hops/layer -> 14 hops total

Measured graph geometry at kNN k=6 on JetNet-150: median diameter 7,
p90 diameter 10, ~8% of jets fragmented.  So sage (2) is deeply starved,
K=4 (6) is just short of the typical jet, and K=8 (14) spans everything.

Run:   python run_ablation_local.py
"""
import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ae_core import run_one, SAVE_ROOT   # noqa: E402

# ══════════════════════════════════════════════════════════════════════════
# SHARED SETTINGS — identical for every run so only the encoder differs
# ══════════════════════════════════════════════════════════════════════════
KNN_K            = 6       # graph degree. CONSTANT. See README for why 6.
LAYERS           = 2       # conv layers (4 saturates the K sweep — see doc)
TOTAL_PARTICLES  = 150     # JetNet-150
EPOCHS           = 200     # per model, all 3 models identical
BATCH_SIZE       = 512

# TRAINING DATA  = JetNet split="train"  (70% of the dataset)
# EVALUATION DATA= JetNet split="test"   (15%, never seen during training)
# Both are None => use ALL of the respective official split.
MAX_JETS_PER_TYPE = None   # cap on TRAIN jets per type; e.g. 20000 = smoke run
MAX_TEST_PER_TYPE = None   # cap on TEST  jets per type
N_EVAL            = 0      # 0 = evaluate on the ENTIRE test split

COMMON = dict(
    knn=KNN_K, layers=LAYERS, total_particles=TOTAL_PARTICLES,
    epochs=EPOCHS, batch_size=BATCH_SIZE, n_eval=N_EVAL,
    max_jets_per_type=MAX_JETS_PER_TYPE,
    max_test_per_type=MAX_TEST_PER_TYPE,
    pool="stats",      # <-- NO Set2Set
    use_attn=False,    # <-- NO decoder self-attention
)

RUNS = [
    dict(COMMON, tag=f"local_sage_knn{KNN_K}_L{LAYERS}_N{TOTAL_PARTICLES}",
         conv="sage", K=1),
    dict(COMMON, tag=f"local_chebK4_knn{KNN_K}_L{LAYERS}_N{TOTAL_PARTICLES}",
         conv="cheb", K=4),
    dict(COMMON, tag=f"local_chebK8_knn{KNN_K}_L{LAYERS}_N{TOTAL_PARTICLES}",
         conv="cheb", K=8),
]


def main():
    print("=" * 74)
    print("ABLATION ARM 1 — LOCAL (no Set2Set, no decoder attention)")
    print(f"  {len(RUNS)} runs | kNN k={KNN_K} | layers={LAYERS} | "
          f"JetNet-{TOTAL_PARTICLES} | {EPOCHS} epochs")
    print("=" * 74)

    results = []
    for i, cfg in enumerate(RUNS, 1):
        print(f"\n\n########## [{i}/{len(RUNS)}]  {cfg['tag']}  ##########\n")
        try:
            results.append(run_one(cfg))
        except Exception:
            print(f"!! RUN FAILED: {cfg['tag']}")
            traceback.print_exc()

    out = os.path.join(SAVE_ROOT, "summary_local.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nArm 1 complete. Summary -> {out}")

    print(f"\n{'tag':44s} {'LR-relerr':>10s} {'ptD(ctrl)':>10s} {'EMD':>10s}")
    for r in results:
        print(f"{r['tag']:44s} {r.get('longrange_relerr_mean', float('nan')):10.4f} "
              f"{r.get('control_relerr_ptD', float('nan')):10.4f} "
              f"{r.get('emd_mean', float('nan')):10.4f}")


if __name__ == "__main__":
    main()
