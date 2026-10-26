"""The floor a conditional model actually faces: Geant4 pairs at matched energies.

`validate_floors.py` and `validate_official.py` measure the floor between two
independent Geant4 samples. A conditional model is not in that position. It
generates **at the evaluation set's own incident energies**, so its sample and
the real one share energies exactly, while two independent Geant4 samples differ
by the energy draw as well. That extra variance is something the model never
pays, so an unmatched floor is mildly generous to it — and at a floor this tight
that is exactly what lets a model appear to score *better than Geant4*
(the GPU run of 2026-09-16 scored chi2 0.83 against an unmatched floor of 1.06).

This script removes the difference: for each shower in the evaluation file, take
the shower from the other file whose incident energy is closest, without reuse.
The pair is then Geant4-vs-Geant4 with the model's own advantage built in, which
is the number a model should be quoted against.

Run from the ``Tina`` folder with both ds2 files present:

    python -m pinnde_eval.validate_matched             # the 7 observables
    python -m pinnde_eval.validate_matched --official  # the 362 features
"""

import argparse
import os

import numpy as np

from .floors import (METRICS, mean_and_spread, pairs_across_files,
                     pairs_matched_energy, score_pairs)


def load(official=False, data_dir=None, n_load=100000):
    """Features and incident energies for both ds2 files."""
    data_dir = data_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..")
    paths = [os.path.join(data_dir, f"dataset_2_{i}.hdf5") for i in (1, 2)]
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}\nDownload ds2 from "
                             f"https://zenodo.org/records/6366271")
    feats, energies, names = [], [], None
    for which, path in enumerate(paths, start=1):
        if official:
            from .calochallenge import official_features_from_file
            cache = os.path.join(data_dir, "calochallenge_cache",
                                 f"ds2_{which}_official_{n_load}.npz")
            block, names, e_inc = official_features_from_file(path, n_load,
                                                              cache=cache)
        else:
            from .observables import observables_from_file
            block, names, e_inc = observables_from_file(path, n_load)
        feats.append(block)
        energies.append(np.log(np.asarray(e_inc).ravel()))
    return feats[0], energies[0], feats[1], energies[1], names


def main(n=8000, repeats=10, official=False, n_load=100000, data_dir=None):
    print(f"loading {'362 official features' if official else '7 observables'} "
          f"for both files")
    a, log_a, b, log_b, names = load(official=official, data_dir=data_dir,
                                     n_load=n_load)
    space = f"{len(names)} {'official features' if official else 'observables'}"

    # The model trains on file 1 and is scored against file 2, generating at
    # file 2's energies, so the matched partner comes from file 1.
    print(f"\nmatched-energy pairs of {n} showers "
          f"(file 2 against file 1 at the same energies)")
    matched = score_pairs(pairs_matched_energy(b, log_b, a, log_a, n, repeats),
                          label="matched ")
    print(f"\nindependent pairs of {n} showers (the usual floor)")
    independent = score_pairs(pairs_across_files(a, b, n, repeats), label="independent ")

    print(f"\nfloors over {repeats} disjoint pairs of {n} showers each, {space}, "
          f"standardized\n")
    print(f"{'metric':>10} | {'matched energies':>19} | {'independent':>19} | {'gap':>8}")
    for metric in METRICS:
        cells, values = [], []
        for source in (matched, independent):
            mean, spread = mean_and_spread(source[metric])
            values.append((mean, spread, len(source[metric])))
            cells.append(f"{'n/a':>19}" if np.isnan(mean)
                         else f"{mean:10.5f} +/-{spread:7.5f}")
        (m1, s1, k1), (m2, s2, k2) = values
        error = np.sqrt(s1 ** 2 / k1 + s2 ** 2 / k2)
        shown = "     n/a" if (np.isnan(m1) or np.isnan(m2) or error == 0) \
            else f"{(m2 - m1) / error:+8.1f}"
        print(f"{metric:>10} | {cells[0]} | {cells[1]} | {shown}")

    print("\ngap = (independent) - (matched), in standard errors. A positive gap "
          "means the\nusual floor sits above the one a conditional model "
          "actually faces, so scoring a\nmodel against it flatters the model by "
          "that much.")
    print("\nQuote conditional-model numbers against the matched column.")
    return matched, independent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", action="store_true",
                        help="the CaloChallenge's 362 features instead of our 7")
    args = parser.parse_args()
    main(official=args.official)
