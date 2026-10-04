#!/usr/bin/env python
"""
SCRIPT 2 — "ATTENTION-POOLING" ARM   (Set2Set back ON, decoder unchanged)
========================================================================

Identical to Script 1 in EXACTLY ONE respect: the pooling.

  local  arm : StatsPool  [sum | mean | max | std]  -- fixed, no routing
  global arm : Set2Set    (LSTM + softmax attention, 4 steps) -- learned
                           content-based global routing

WHY DECODER ATTENTION IS *NOT* TURNED BACK ON HERE
--------------------------------------------------
v15/v15.2 had two global-mixing mechanisms.  Switching both on at once would
make this a COMPOUND treatment: if the two arms differ, you could not say
whether it was the pooling or the decoder that caused it.

The two are also not equally relevant to the thesis:

  * Set2Set sits in the ENCODER, BEFORE the bottleneck.  It is a direct
    competitor to receptive field -- both are ways of getting information
    from distant particles into the latent.  This is the thing worth testing.

  * Decoder self-attention sits AFTER the bottleneck, where all 150 slots
    already receive the same z.  It can invent inter-particle structure that
    the latent never encoded, which partially rescues ANY encoder and
    COMPRESSES the differences between sage / K=4 / K=8 -- exactly the
    differences being measured.  It also costs ~20-30% more compute.

So the decoder is held CONSTANT (per-slot FiLM residual MLP) across all six
runs, and pooling is the single variable.  The result is then attributable:
"learned attention pooling vs fixed statistics pooling, at each receptive
field".

To test decoder attention as well, flip use_attn=True below -- but treat that
as a separate 3-run experiment, not as part of this comparison.

WHY THIS ARM EXISTS
-------------------
Running the same three encoders with and without attention lets you answer
the question directly:

    "How much of the work that global attention does can a wider spectral
     receptive field do instead?"

For each encoder, define the ATTENTION GAP:

    gap(encoder) = relerr_local(encoder) - relerr_global(encoder)

  * gap large for 1-hop, small for ChebNet K=8
        -> ChebNet substitutes for attention.  THIS IS THE THESIS RESULT.
  * gap the same size for every encoder
        -> attention adds a constant benefit that K cannot buy.
  * gap ~ 0 everywhere
        -> attention was never doing anything here (also a real result).

THREE RUNS (mirroring Script 1 exactly):
  1. sage       strictly 1-hop  ->  2 hops total
  2. cheb K=4                    ->  6 hops total
  3. cheb K=8                    -> 14 hops total

Run:   python run_ablation_global.py
"""
import os
import sys
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ae_core import run_one, SAVE_ROOT   # noqa: E402

# ══════════════════════════════════════════════════════════════════════════
# MUST MATCH run_ablation_local.py EXACTLY (except pool / use_attn)
# ══════════════════════════════════════════════════════════════════════════
KNN_K             = 6
LAYERS            = 2
TOTAL_PARTICLES   = 150
EPOCHS            = 200
BATCH_SIZE        = 512
MAX_JETS_PER_TYPE = None   # TRAIN = JetNet split="train" (70%)
MAX_TEST_PER_TYPE = None   # TEST  = JetNet split="test"  (15%)
N_EVAL            = 0      # 0 = entire test split

COMMON = dict(
    knn=KNN_K, layers=LAYERS, total_particles=TOTAL_PARTICLES,
    epochs=EPOCHS, batch_size=BATCH_SIZE, n_eval=N_EVAL,
    max_jets_per_type=MAX_JETS_PER_TYPE,
    max_test_per_type=MAX_TEST_PER_TYPE,
    pool="set2set",   # <-- THE ONLY DIFFERENCE from the local arm
    use_attn=False,   # <-- decoder attention stays OFF.  See note above.
)

RUNS = [
    dict(COMMON, tag=f"global_sage_knn{KNN_K}_L{LAYERS}_N{TOTAL_PARTICLES}",
         conv="sage", K=1),
    dict(COMMON, tag=f"global_chebK4_knn{KNN_K}_L{LAYERS}_N{TOTAL_PARTICLES}",
         conv="cheb", K=4),
    dict(COMMON, tag=f"global_chebK8_knn{KNN_K}_L{LAYERS}_N{TOTAL_PARTICLES}",
         conv="cheb", K=8),
]


def main():
    print("=" * 74)
    print("ABLATION ARM 2 — GLOBAL (Set2Set pooling; decoder identical to arm 1)")
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

    out = os.path.join(SAVE_ROOT, "summary_global.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nArm 2 complete. Summary -> {out}")

    print(f"\n{'tag':44s} {'LR-relerr':>10s} {'ptD(ctrl)':>10s} {'EMD':>10s}")
    for r in results:
        print(f"{r['tag']:44s} {r.get('longrange_relerr_mean', float('nan')):10.4f} "
              f"{r.get('control_relerr_ptD', float('nan')):10.4f} "
              f"{r.get('emd_mean', float('nan')):10.4f}")


if __name__ == "__main__":
    main()
