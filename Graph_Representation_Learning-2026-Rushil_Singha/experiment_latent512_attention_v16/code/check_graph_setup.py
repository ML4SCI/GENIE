#!/usr/bin/env python
"""
check_graph_setup.py — run this FIRST, before spending GPU hours.
=================================================================

Measures the jet-graph diameter for several candidate kNN k values.

WHY IT MATTERS
--------------
Chebyshev order K buys you (K-1) hops per layer.  If the graph is only
3 hops wide, then K=4 already reaches every particle and K=10 has nothing
left to do — the K sweep comes out FLAT and the experiment cannot answer
the question.  You want:

        layers x (K_small - 1)   <   median diameter   <=  layers x (K_big - 1)

so the small-K model is genuinely starved and the big-K model is not.

Prints a table you can paste into the paper as setup justification.

Run:  python check_graph_setup.py
      python check_graph_setup.py --ks 4 6 8 10 16 --num-particles 150
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ae_core import (load_jetnet_data, build_graphs, check_graph_diameter,   # noqa
                     CACHE_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 6, 8, 10, 16])
    ap.add_argument("--num-particles", type=int, default=150)
    ap.add_argument("--n-jets", type=int, default=4000,
                    help="jets per type to sample (fast)")
    ap.add_argument("--n-sample", type=int, default=400,
                    help="graphs to measure diameter on")
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()

    pdata, _, ptypes = load_jetnet_data(args.num_particles, args.n_jets,
                                        split="train")

    rows = []
    for k in args.ks:
        cache = os.path.join(
            CACHE_ROOT, f"diagcheck_np{args.num_particles}_knn{k}"
                        f"_mj{args.n_jets}.pkl")
        graphs, _, _ = build_graphs(pdata, ptypes, k, args.num_particles, cache)
        st = check_graph_diameter(graphs, n_sample=args.n_sample)
        rows.append((k, st))

    print("\n" + "=" * 78)
    print(f"JetNet-{args.num_particles}   |   {args.layers} conv layers")
    print("=" * 78)
    hdr = (f"{'kNN k':>6} {'med diam':>9} {'p90 diam':>9} {'avg path':>9} "
           f"{'n_comp':>7} | {'K needed to span':>17}")
    print(hdr); print("-" * len(hdr))
    for k, st in rows:
        d = st.get("median_diameter")
        # K such that layers*(K-1) >= diameter  ->  K >= diam/layers + 1
        kneed = (int(np.ceil(d / args.layers)) + 1) if d else None
        print(f"{k:>6} {d if d else '-':>9} "
              f"{st.get('p90_diameter', '-'):>9} "
              f"{st.get('mean_avg_shortest_path', 0):>9.2f} "
              f"{st.get('mean_n_components', 0):>7.2f} | {str(kneed):>17}")
    print("\nRead it like this: a K BELOW the last column is starved "
          "(good, it is the baseline);")
    print("a K well ABOVE it is saturated (the sweep will look flat there).")


if __name__ == "__main__":
    main()
