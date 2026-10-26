"""The diagnosis must point at the part that is actually broken, and only there."""

import numpy as np

from pinnde_eval.diagnose_samples import column_groups, diagnose, with_true_energy

QUANTITIES = ("logE_layer", "EC_eta", "EC_phi", "width_eta", "width_phi",
              "sparsity", "EC_R", "width_R")


def _names(n_layers=6):
    names = ["logE_inc"]
    for q in QUANTITIES:
        names += [f"{q}_{l}" for l in range(n_layers)]
    return names + ["logE_tot"]


def test_groups_cover_families_and_depth_thirds():
    names = _names()
    groups = column_groups(names)
    assert len(groups["layer energies"]) == 6
    assert len(groups["eta/phi widths"]) == 12
    assert groups["totals"] == [0, len(names) - 1]
    assert set(groups) >= {"layers 0-1", "layers 2-3", "layers 4-5"}
    assert len(groups["layers 4-5"]) == 2 * len(QUANTITIES)


def test_true_energy_substitution_touches_only_that_column():
    names = _names()
    rng = np.random.default_rng(1)
    generated = rng.normal(size=(50, len(names)))
    e_inc = np.exp(rng.uniform(np.log(1e3), np.log(1e6), 50))
    fixed = with_true_energy(generated, e_inc, names)
    np.testing.assert_allclose(fixed[:, 0], np.log10(e_inc))
    np.testing.assert_array_equal(fixed[:, 1:], generated[:, 1:])
    assert not np.shares_memory(fixed, generated)


def test_only_the_corrupted_family_is_flagged():
    rng = np.random.default_rng(0)
    names = _names()
    n = 3000
    real = rng.normal(size=(n, len(names)))
    generated = rng.normal(size=(n, len(names)))       # same distribution...
    widths = column_groups(names)["eta/phi widths"]
    generated[:, widths] += 1.0                        # ...except the widths
    e_inc = np.exp(rng.uniform(np.log(1e3), np.log(1e6), n))

    table = diagnose(real, generated, e_inc, names, n_classifier=2)
    assert table["eta/phi widths"]["all"] > 0.8
    for clean in ("layer energies", "eta/phi centres", "sparsity", "totals"):
        assert abs(table[clean]["all"] - 0.5) < 0.05
