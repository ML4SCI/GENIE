"""Reproduce the measurements the report rests on.

Three claims in the report came from one-off analyses rather than from the
scoring path, which makes them hard for a reader to check. This script runs
them again from the data.

    python -m pinnde_eval.validate_claims                       # 1 and 2
    python -m pinnde_eval.validate_claims runs/samples.npz      # also 3

Claim 1: no layer deposits between zero and about 15 keV, which is what makes
    the empty band safe to snap (report, section 3).
Claim 2: reading an observable family against 0.5 is fair, because two real
    samples score 0.5 on every family (report, section 1 and the family tables).
Claim 3: a sample carrying the model's marginals inside Geant4's correlation
    structure is nearly as detectable as the model itself, so per-feature
    accuracy rather than correlation is the binding constraint (section 6).
    Needs a file written by ``demo_calo --save-samples``.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pinnde_eval.calochallenge import official_features_from_file
from pinnde_eval.diagnose_samples import column_groups
from pinnde_eval.floors import match_by_energy
from pinnde_eval.tier1 import classifier_two_sample_test

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def smallest_deposit(n=100000):
    """Claim 1: the gap between an empty layer and the faintest real one."""
    import h5py
    print("\n1. smallest non-zero energy in any layer")
    worst = np.inf
    for name in ("dataset_2_1.hdf5", "dataset_2_2.hdf5"):
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            print("   %s not found, skipped" % name)
            continue
        with h5py.File(path, "r") as f:
            v = f["showers"][:n].astype(np.float64)
        layer = v.reshape(len(v), 45, 144).sum(axis=2)
        nz = layer[layer > 0]
        in_gap = int(np.sum((layer > 0) & (layer < 0.015)))
        worst = min(worst, nz.min())
        print("   %s: %d showers, smallest %.3f keV, layers in (0, 15 keV): %d"
              % (name, len(v), nz.min() * 1e3, in_gap))
    if np.isfinite(worst):
        print("   -> the band from empty up to %.2f keV is unused by real data"
              % (worst * 1e3))


def family_floors(n=8000):
    """Claim 2: what a family scores when both samples are real."""
    print("\n2. per-family null floor, two disjoint real samples")
    a_path = os.path.join(DATA, "dataset_2_1.hdf5")
    b_path = os.path.join(DATA, "dataset_2_2.hdf5")
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        print("   both ds2 files needed, skipped")
        return
    x1, names, e1 = official_features_from_file(a_path, 2 * n)
    x2, _, e2 = official_features_from_file(b_path, n)
    picks, worst = match_by_energy(np.log(e2), np.log(e1))
    a, b = x2, x1[picks]
    print("   energies matched to %.1e in log E" % worst)
    values = []
    for label, cols in column_groups(names).items():
        auc = float(classifier_two_sample_test(a[:, cols], b[:, cols], k=3, seed=0)[0])
        values.append(auc)
        print("   %-22s %d columns  AUC %.4f" % (label, len(cols), auc))
    print("   -> floors span %.4f to %.4f, so reading a family against 0.5 is "
          "fair to about +-%.3f" % (min(values), max(values),
                                    max(abs(np.array(values) - 0.5))))


def marginals_versus_correlations(path, n_layers=10):
    """Claim 3: how much of the detectability is per-feature error.

    Builds a surrogate with the model's marginals and Geant4's rank
    correlations. Restricted to the front layers, where no point mass couples
    the columns: with atoms present the surrogate breaks the empty-layer rule
    and the classifier detects the construction instead of the marginals.
    """
    print("\n3. marginals versus correlations, layers 0-%d" % (n_layers - 1))
    d = np.load(path, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    real, gen = d["real"], d["generated"]
    ok = np.isfinite(gen).all(axis=1)
    real, gen = real[ok], gen[ok]
    tags = ("logE_layer", "EC_eta", "EC_phi", "width_eta", "width_phi",
            "EC_R", "width_R", "sparsity")
    cols = [names.index("%s_%d" % (t, L)) for L in range(n_layers) for t in tags]
    e = [names.index("logE_layer_%d" % L) for L in range(n_layers)]
    lit_r, lit_g = (real[:, e] > -8 + 1e-9).all(axis=1), (gen[:, e] > -8 + 1e-9).all(axis=1)
    r, g = real[lit_r][:, cols], gen[lit_g][:, cols]
    m = min(len(r), len(g))
    r, g = r[:m], g[:m]
    surrogate = np.empty_like(r)
    for j in range(r.shape[1]):
        surrogate[:, j] = np.sort(g[:, j])[np.argsort(np.argsort(r[:, j]))]
    full = float(classifier_two_sample_test(r, g, k=3, seed=0)[0])
    marg = float(classifier_two_sample_test(r, surrogate, k=3, seed=0)[0])
    print("   %d showers, %d columns" % (m, len(cols)))
    print("   the model                                  AUC %.4f" % full)
    print("   its marginals with Geant4's correlations   AUC %.4f" % marg)
    print("   -> correlations add %.3f on top of the marginal error" % (full - marg))


if __name__ == "__main__":
    smallest_deposit()
    family_floors()
    for p in sys.argv[1:]:
        marginals_versus_correlations(p)
    if len(sys.argv) == 1:
        print("\n(pass a file from `demo_calo --save-samples` to run claim 3)")
