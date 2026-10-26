"""Where does a generator fail? Break a saved comparison down piece by piece.

``demo_calo --save-samples PATH`` writes the evaluation showers and the
generated ones. This scores them part by part, so a failure can be located
without training again:

* by **column family** -- layer energies, eta/phi centres, eta/phi widths,
  sparsity, radial centre and width, the totals -- each on its own columns;
* by **depth** -- the front, middle and back thirds of the calorimeter, each on
  all quantities of its layers;
* by **incident energy** -- all showers, the lowest quartile, the highest.

Every cell is a classifier AUC on just those columns. 0.5 means that part of the
shower cannot be told from Geant4; the cells far above it are where the model is
wrong. A family can look right on its own and still fail jointly with the
others, so read this beside the pooled AUC, not instead of it.

    python -m pinnde_eval.diagnose_samples path/to/samples.npz
"""

import argparse
import re

import numpy as np

from .tier1 import classifier_two_sample_test

FAMILIES = {
    "layer energies": ("logE_layer",),
    "eta/phi centres": ("EC_eta", "EC_phi"),
    "eta/phi widths": ("width_eta", "width_phi"),
    "sparsity": ("sparsity",),
    "radial centre+width": ("EC_R", "width_R"),
}
TOTALS = ("logE_inc", "logE_tot")


def _layer_of(name):
    match = re.search(r"_(\d+)$", name)
    return int(match.group(1)) if match else None


def column_groups(names):
    """``{label: column indices}`` for every family and depth third."""
    groups = {}
    for label, prefixes in FAMILIES.items():
        cols = [i for i, n in enumerate(names)
                if any(n.startswith(p + "_") for p in prefixes)]
        if cols:
            groups[label] = cols
    totals = [i for i, n in enumerate(names) if n in TOTALS]
    if totals:
        groups["totals"] = totals

    layers = [_layer_of(n) for n in names]
    n_layers = max((l for l in layers if l is not None), default=-1) + 1
    if n_layers >= 3:
        cuts = np.linspace(0, n_layers, 4).round().astype(int)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            cols = [i for i, l in enumerate(layers) if l is not None and lo <= l < hi]
            groups[f"layers {lo}-{hi - 1}"] = cols
    return groups


def diagnose(real, generated, e_inc, names, n_classifier=3, seed=0):
    """AUC per column group, pooled and in the lowest and highest energy quartiles."""
    log_e = np.log(np.asarray(e_inc, dtype=np.float64).ravel())
    edges = np.quantile(log_e, [0.25, 0.75])
    slices = {"all": np.ones(len(log_e), dtype=bool),
              "lowest E": log_e <= edges[0],
              "highest E": log_e >= edges[1]}
    table = {}
    for label, cols in column_groups(list(names)).items():
        table[label] = {
            where: classifier_two_sample_test(real[mask][:, cols],
                                              generated[mask][:, cols],
                                              k=n_classifier, seed=seed)[0]
            for where, mask in slices.items()}
    return table


def with_true_energy(generated, e_inc, names):
    """The generated sample with its log10 E_inc column set to the true value.

    The official features contain log10 E_inc, which the model was handed as its
    condition and still has to regenerate -- a point mass a continuous flow
    cannot hit. Comparing the pooled AUC before and after this substitution
    measures what that built-in defect alone costs, without retraining.
    """
    fixed = np.array(generated, dtype=np.float64, copy=True)
    if "logE_inc" in names:
        fixed[:, names.index("logE_inc")] = np.log10(
            np.asarray(e_inc, dtype=np.float64).ravel())
    return fixed


def main(path, n_classifier=3, seed=0):
    data = np.load(path, allow_pickle=False)
    names = [str(n) for n in data["names"]]
    real, generated, e_inc = data["real"], data["generated"], data["e_inc"]
    print(f"{len(real)} real vs {len(generated)} generated showers, "
          f"{len(names)} columns\n")

    if "logE_inc" in names:
        column = names.index("logE_inc")
        error = np.abs(generated[:, column] - np.log10(e_inc.ravel()))
        before = classifier_two_sample_test(real, generated, k=n_classifier, seed=seed)[0]
        after = classifier_two_sample_test(real, with_true_energy(generated, e_inc, names),
                                           k=n_classifier, seed=seed)[0]
        print(f"regenerated log10 E_inc is off by a median {np.median(error):.4f}")
        print(f"pooled AUC, all columns as generated      : {before:.4f}")
        print(f"pooled AUC, log10 E_inc set to true value : {after:.4f}\n")

    table = diagnose(real, generated, e_inc, names,
                     n_classifier=n_classifier, seed=seed)
    print("classifier AUC per part of the shower (0.5 = indistinguishable)\n")
    print(f"{'part':>20} | {'columns':>7} | {'all':>6} | {'lowest E':>8} | {'highest E':>9}")
    groups = column_groups(names)
    for label, row in table.items():
        print(f"{label:>20} | {len(groups[label]):7d} | {row['all']:6.3f} | "
              f"{row['lowest E']:8.3f} | {row['highest E']:9.3f}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", help=".npz written by demo_calo --save-samples")
    args = parser.parse_args()
    main(args.samples)
