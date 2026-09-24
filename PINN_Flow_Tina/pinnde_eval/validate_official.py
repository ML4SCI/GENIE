"""Floors in the CaloChallenge's 362-feature space, and what they can detect.

Two questions, and the second is what makes the first worth quoting:

1. **What does a perfect generator score here?** Measured over 10 disjoint
   pairs of Geant4 samples, drawn both inside one file (sampling noise alone)
   and across the two files (the floor a model is scored against).
2. **How large a difference would this test have seen?** The two ds2 files
   differ by 0.018 in mean log E_inc. Concluding that the difference "does not
   matter" is empty unless the test would have caught a larger one, so
   ``--sensitivity`` resamples one side to carry a known energy shift and finds
   the size at which the test starts to notice.

Run from the ``Tina`` folder with both ds2 files present (the first run also
extracts and caches the features, a few minutes):

    python -m pinnde_eval.validate_official                # the two floors
    python -m pinnde_eval.validate_official --sensitivity  # what it can detect
"""

import argparse
import os
import time

import numpy as np

from .calochallenge import official_features_from_file
from .evaluate import evaluate
from .floors import (mean_and_spread, pairs_across_files, pairs_within_file,
                     report, save, score_pairs)

# The measured difference between the two ds2 files, over all 100,000 showers
# each: 0.018 in mean log E_inc, 2.0 sigma, KS on E_inc p = 0.10.
FILE_TO_FILE_SHIFT = 0.018


def load_features(data_dir=None, n_load=100000, cache_dir=None):
    """Official features for both ds2 files, cached as .npz after the first run."""
    data_dir = data_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")
    cache_dir = cache_dir or os.path.join(data_dir, "calochallenge_cache")
    feats, energies, names = [], [], None
    for which in (1, 2):
        path = os.path.join(data_dir, f"dataset_2_{which}.hdf5")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}\nDownload ds2 from "
                             f"https://zenodo.org/records/6366271")
        cache = os.path.join(cache_dir, f"ds2_{which}_official_{n_load}.npz")
        start = time.time()
        block, names, e_inc = official_features_from_file(path, n_load, cache=cache)
        print(f"  file {which}: {block.shape[0]} showers x {block.shape[1]} "
              f"features in {time.time() - start:.1f} s")
        feats.append(block)
        energies.append(e_inc)
    return feats[0], feats[1], names, energies[0], energies[1]


def tilted_sample(feats, log_e, n, shift, rng):
    """``n`` showers whose mean log E_inc is about ``shift`` higher.

    Exponential tilting: weight shower i by ``exp(lam * log_e[i])`` with
    ``lam = shift / var(log_e)``, which moves the mean by roughly ``shift``.
    Only real showers are used -- nothing in the features is perturbed by hand,
    so the difference the test sees is exactly what a real energy shift does.
    Selection is without replacement (Gumbel top-k), so no shower is duplicated.
    """
    if shift == 0.0:
        idx = rng.choice(len(feats), n, replace=False)
    else:
        lam = shift / np.var(log_e)
        keys = lam * log_e + rng.gumbel(size=len(log_e))
        idx = np.argpartition(-keys, n)[:n]
    return feats[idx], log_e[idx]


def shift_sensitivity(feats, e_inc, shifts=(0.0, FILE_TO_FILE_SHIFT, 0.05, 0.1, 0.2),
                      n=8000, repeats=3, seed=0, n_classifier=3):
    """AUC against a known incident-energy shift, within a single file.

    Both samples come from the same file, so with ``shift = 0`` this is an exact
    null. Larger shifts say how big a difference the 362-feature test can see at
    this sample size.
    """
    log_e = np.log(np.asarray(e_inc).ravel())
    half = len(feats) // 2
    pool_size = (len(feats) - half) // repeats
    if pool_size < n or half < repeats * n:
        raise ValueError("not enough showers for this many repeats")

    print(f"\nsensitivity: same file, one side resampled to a known shift "
          f"({repeats} repeats of {n} showers)\n")
    print(f"{'shift asked':>12} | {'shift realised':>14} | {'auc':>17} | {'sep_mean':>9}")
    rows = []
    for shift in shifts:
        aucs, seps, realised = [], [], []
        for r in range(repeats):
            rng = np.random.default_rng(seed + 1000 * r + int(round(shift * 1000)))
            reference = feats[r * n:(r + 1) * n]
            lo = half + r * pool_size
            other, other_log_e = tilted_sample(feats[lo:lo + pool_size],
                                               log_e[lo:lo + pool_size], n, shift, rng)
            realised.append(other_log_e.mean() - log_e[r * n:(r + 1) * n].mean())
            res = evaluate(reference, other, tier="full", standardize=True,
                           seed=seed + r, n_classifier=n_classifier)
            aucs.append(res["auc"][0])
            seps.append(res["sep_mean"])
        mean, spread = mean_and_spread(np.array(aucs))
        print(f"{shift:12.3f} | {np.mean(realised):14.3f} | "
              f"{mean:8.4f} +/-{spread:6.4f} | {np.mean(seps):9.5f}")
        rows.append((shift, float(np.mean(realised)), mean, spread,
                     float(np.mean(seps))))

    null = rows[0]
    print(f"\nThe two ds2 files differ by {FILE_TO_FILE_SHIFT:.3f} in mean "
          f"log E_inc. Read the table as: a shift\nis visible once its AUC "
          f"stands clear of the {null[2]:.4f} +/-{null[3]:.4f} that the same "
          f"file scores\nagainst itself. If the {FILE_TO_FILE_SHIFT:.3f} row "
          f"sits on the null and a larger row does not, the\nfile-to-file "
          f"difference is below what this test can see, which is the honest\n"
          f"version of \"the files agree\".")
    return rows


def main(n=8000, repeats=10, n_load=100000, data_dir=None, sensitivity=False,
         shifts=None):
    print("extracting the CaloChallenge's 362 features (cached after the first run)")
    a, b, names, e_a, e_b = load_features(data_dir=data_dir, n_load=n_load)
    space = f"{len(names)} official features"

    print(f"\nsame-file pairs of {n} showers (pure null)")
    within = score_pairs(pairs_within_file(a, b, n, repeats), label="same file ")
    print(f"\ndifferent-file pairs of {n} showers (the floor to quote)")
    across = score_pairs(pairs_across_files(a, b, n, repeats), label="across ")
    report(within, across, n, repeats, space)
    save(os.path.join(data_dir or os.path.join(os.path.dirname(
        os.path.abspath(__file__)), ".."), "floor_results",
        f"official_n{n}_r{repeats}.npz"), within, across, n, repeats, space)

    assert abs(np.nanmean(within["auc"]) - 0.5) < 0.05, (
        f"same-file AUC {np.nanmean(within['auc'])} is not ~0.5; the null itself "
        f"is broken")

    rows = None
    if sensitivity:
        extra = {"shifts": shifts} if shifts else {}
        rows = shift_sensitivity(a, e_a, n=n, **extra)
    print("\nFloors measured. Quote a model number next to this table, never alone.")
    return within, across, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", action="store_true",
                        help="also measure how large an energy shift the test can see")
    parser.add_argument("--shifts", default=None,
                        help="comma-separated shifts in mean log E_inc to try, "
                             "e.g. 0,0.1,0.2,0.35,0.5")
    parser.add_argument("--only-sensitivity", action="store_true",
                        help="skip the floors and measure sensitivity alone")
    args = parser.parse_args()
    shifts = (tuple(float(s) for s in args.shifts.split(","))
              if args.shifts else None)
    if args.only_sensitivity:
        a, _, _, e_a, _ = load_features()
        shift_sensitivity(a, e_a, **({"shifts": shifts} if shifts else {}))
    else:
        main(sensitivity=args.sensitivity or bool(shifts), shifts=shifts)
