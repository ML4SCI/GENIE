"""The Geant4-vs-Geant4 floors in our own feature spaces, over disjoint repeats.

The floors in DEVLOG sections 11 and 15, quoted in every meeting document so
far, came from a **single** N=8000 comparison. Their AUC error bar came from
retraining the classifier on that one pair, which measures classifier noise, not
the sampling noise that actually limits a comparison. This script measures them
the way `validate_official.py` measures the 362-feature floor: over disjoint
pairs, drawn both inside one file and across the two.

Run from the ``Tina`` folder with both ds2 files present:

    python -m pinnde_eval.validate_floors              # 7 core observables
    python -m pinnde_eval.validate_floors --per-layer  # 187 columns (slower)
"""

import argparse
import os
import time

import numpy as np

from .floors import (pairs_across_files, pairs_within_file, report, save,
                     score_pairs)
from .observables import observables_from_file


def load_observables(data_dir=None, n_load=100000, per_layer=False):
    data_dir = data_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")
    feats, names = [], None
    for which in (1, 2):
        path = os.path.join(data_dir, f"dataset_2_{which}.hdf5")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}\nDownload ds2 from "
                             f"https://zenodo.org/records/6366271")
        start = time.time()
        block, names, _ = observables_from_file(path, n_load, per_layer=per_layer)
        print(f"  file {which}: {block.shape[0]} showers x {block.shape[1]} "
              f"observables in {time.time() - start:.1f} s")
        feats.append(block)
    return feats[0], feats[1], names


def main(n=8000, repeats=10, n_load=100000, per_layer=False, data_dir=None):
    print(f"extracting {'187 per-layer columns' if per_layer else '7 observables'} "
          f"from {n_load} showers per file")
    a, b, names = load_observables(data_dir=data_dir, n_load=n_load,
                                   per_layer=per_layer)
    space = f"{len(names)} observables"

    print(f"\nsame-file pairs of {n} showers (pure null)")
    within = score_pairs(pairs_within_file(a, b, n, repeats), label="same file ")
    print(f"\ndifferent-file pairs of {n} showers (the floor to quote)")
    across = score_pairs(pairs_across_files(a, b, n, repeats), label="across ")
    report(within, across, n, repeats, space)
    save(os.path.join(data_dir or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), ".."), "floor_results",
        f"d{len(names)}_n{n}_r{repeats}.npz"), within, across, n, repeats, space)

    assert abs(np.nanmean(within["auc"]) - 0.5) < 0.05, (
        f"same-file AUC {np.nanmean(within['auc'])} is not ~0.5; the null itself "
        f"is broken")
    print("\nThese replace the single-comparison floors of DEVLOG sections 11 "
          "and 15.\nThe +/- is now sampling noise across disjoint pairs, which "
          "is what limits a\ncomparison at this sample size.")
    return within, across


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-layer", action="store_true",
                        help="the 187-column space instead of the 7 observables")
    args = parser.parse_args()
    main(per_layer=args.per_layer)
