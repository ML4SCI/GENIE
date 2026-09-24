"""How much of a 362-feature comparison is about correlations, not marginals?

A generator can reproduce every conditional marginal exactly and still fail,
if its columns are not tied together the way real showers tie them. This
measures what that alone costs, with no model involved: take real Geant4
showers, destroy chosen correlations while keeping the marginals exactly, and
score the result against untouched Geant4 at matched incident energies.

Three variants, from no damage to the most:

* **intact** -- real showers, nothing moved. This is the floor.
* **layers shuffled** -- all eight quantities of a layer move together to
  another shower of nearly the same energy. Structure *within* a layer
  survives; structure *across* layers is destroyed.
* **columns shuffled** -- every column moves independently to another shower of
  nearly the same energy. Only the conditional marginals survive.

If shuffled columns already score near AUC 1 in some energy range, then any
generator that learns marginals but not correlations fails there too, and the
model's failure in that range points at correlations rather than at the shapes
of individual distributions.

Run from the ``Tina`` folder with both ds2 files present:

    python -m pinnde_eval.validate_structure
"""

import argparse
import os

import numpy as np

from .calochallenge import official_features_from_file
from .evaluate import evaluate
from .floors import match_by_energy

PER_LAYER = ("logE_layer", "EC_eta", "EC_phi", "width_eta", "width_phi",
             "sparsity", "EC_R", "width_R")


def shuffle_within_energy(feats, log_e, groups, bin_size, rng):
    """Permute each group of columns across showers of nearly the same energy.

    ``groups`` is a list of column-index lists; each group moves as one block,
    independently of the others. Showers are sorted by incident energy and cut
    into bins of ``bin_size``, and permutations never cross a bin, so every
    conditional marginal is preserved exactly.
    """
    out = feats.copy()
    order = np.argsort(log_e)
    for start in range(0, len(order), bin_size):
        members = order[start:start + bin_size]
        for columns in groups:
            moved = members[rng.permutation(len(members))]
            out[np.ix_(members, columns)] = feats[np.ix_(moved, columns)]
    return out


def restore_total(feats, names):
    """Recompute log10 E_tot from the (shuffled) layer energies.

    E_tot is the sum of the layer energies, a hard constraint that any shuffle
    of layers breaks exactly. Restoring it separates two things a classifier
    can use: that single identity, and the softer correlations of how a shower
    develops from layer to layer.
    """
    column = {name: i for i, name in enumerate(names)}
    layers = [column[f"logE_layer_{l}"] for l in range(45)]
    energy = np.clip(10.0 ** feats[:, layers] - 1e-8, 0.0, None).sum(axis=1)
    out = feats.copy()
    out[:, column["logE_tot"]] = np.log10(energy + 1e-8)
    return out


def quartile_auc(real, other, log_e, seed, n_classifier):
    """AUC in the lowest and highest incident-energy quartiles."""
    edges = np.quantile(log_e, [0.25, 0.75])
    low, high = log_e <= edges[0], log_e >= edges[1]
    return tuple(evaluate(real[m], other[m], tier="full", standardize=True,
                          seed=seed, n_classifier=n_classifier)["auc"][0]
                 for m in (low, high))


def main(n=8000, bin_size=50, seed=0, n_classifier=3, data_dir=None):
    data_dir = data_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    cache = os.path.join(data_dir, "calochallenge_cache")
    train, names, e_train = official_features_from_file(
        os.path.join(data_dir, "dataset_2_1.hdf5"), 100000,
        cache=os.path.join(cache, "ds2_1_official_100000.npz"))
    real, _, e_real = official_features_from_file(
        os.path.join(data_dir, "dataset_2_2.hdf5"), n,
        cache=os.path.join(cache, f"ds2_2_official_{n}.npz"))
    log_real = np.log(e_real.ravel())

    # The "generator": real showers from the other file at matched energies,
    # so the intact variant is exactly the matched floor.
    picks, worst = match_by_energy(log_real, np.log(e_train.ravel()))
    base, log_base = train[picks], np.log(e_train.ravel())[picks]
    print(f"{n} showers per side, energies matched to {worst:.1e} in log E_inc; "
          f"shuffles stay inside bins of {bin_size} showers of similar energy\n")

    column = {name: i for i, name in enumerate(names)}
    keep = [column["logE_inc"]]                      # the condition never moves
    layers = [[column[f"{q}_{l}"] for q in PER_LAYER] for l in range(45)]
    singles = [[i] for i in range(len(names)) if i not in keep]

    rng = np.random.default_rng(seed)
    by_layer = shuffle_within_energy(base, log_base,
                                     layers + [[column["logE_tot"]]], bin_size, rng)
    by_column = shuffle_within_energy(base, log_base, singles, bin_size, rng)
    rebuilt = restore_total(base, names)
    gap = np.abs(rebuilt[:, column["logE_tot"]] - base[:, column["logE_tot"]]).max()
    print(f"sanity: log10 E_tot rebuilt from the layers differs from the stored "
          f"value by at most {gap:.1e}\n")
    variants = {
        "intact": base,
        "layers shuffled": by_layer,
        "layers, total kept": restore_total(by_layer, names),
        "columns shuffled": by_column,
        "columns, total kept": restore_total(by_column, names),
    }

    print(f"{'variant':>20} | {'pooled auc':>10} | {'lowest E auc':>12} | "
          f"{'highest E auc':>13} | {'sep_mean':>9}")
    rows = {}
    for label, other in variants.items():
        res = evaluate(real, other, tier="full", standardize=True, seed=seed,
                       n_classifier=n_classifier)
        low, high = quartile_auc(real, other, log_real, seed, n_classifier)
        rows[label] = (res["auc"][0], low, high, res["sep_mean"])
        print(f"{label:>20} | {res['auc'][0]:10.4f} | {low:12.4f} | "
              f"{high:13.4f} | {res['sep_mean']:9.5f}")

    print("\nsep_mean should barely move between variants: shuffling inside an "
          "energy bin\nkeeps every marginal, so any change in AUC is the price "
          "of the lost correlations.")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-size", type=int, default=50,
                        help="showers per energy bin the shuffles stay inside")
    args = parser.parse_args()
    main(bin_size=args.bin_size)
