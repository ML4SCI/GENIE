"""Each check must fire on the defect it is for, and stay quiet otherwise.

A check that never fires is worse than no check, so every test here breaks one
physical rule on purpose and asserts that exactly that check catches it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pinnde_eval.calochallenge import official_feature_names
from pinnde_eval.validate_physical import (CHECKS, EMPTY_LOG_ENERGY,
                                           SPARSITY_SPACING, run_checks)


def _valid_showers(n=50, seed=0, empty_layers=(40, 41)):
    """Rows that obey every rule, including the empty-layer convention."""
    rng = np.random.default_rng(seed)
    names = official_feature_names()
    x = np.zeros((n, len(names)))
    index = {m: i for i, m in enumerate(names)}
    x[:, index["logE_inc"]] = rng.uniform(3.0, 6.0, n)
    for layer in range(45):
        energy = rng.uniform(1.0, 4.0, n)           # log10 MeV
        x[:, index[f"logE_layer_{layer}"]] = energy
        for tag in ("EC_eta", "EC_phi"):
            x[:, index[f"{tag}_{layer}"]] = rng.normal(0, 0.05, n)
        for tag in ("width_eta", "width_phi", "EC_R", "width_R"):
            x[:, index[f"{tag}_{layer}"]] = rng.uniform(0.01, 0.2, n)
        lit = rng.integers(1, 144, n)
        x[:, index[f"sparsity_{layer}"]] = 1.0 - lit * SPARSITY_SPACING
    for layer in empty_layers:                      # emptied consistently
        x[:, index[f"logE_layer_{layer}"]] = EMPTY_LOG_ENERGY
        for tag in ("EC_eta", "EC_phi", "width_eta", "width_phi", "EC_R", "width_R"):
            x[:, index[f"{tag}_{layer}"]] = 0.0
        x[:, index[f"sparsity_{layer}"]] = 1.0
    # Scale the layers so the shower deposits ~78% of the beam energy, the way
    # a real one does, then make the total the sum of them. Without this the
    # fixture deposits more energy than it receives and the conservation check
    # rightly fires on it.
    cols = [index[f"logE_layer_{i}"] for i in range(45)]
    linear = np.clip(10.0 ** x[:, cols] - 1e-8, 0, None)
    wanted = 0.78 * 10.0 ** x[:, index["logE_inc"]]
    linear *= (wanted / np.maximum(linear.sum(axis=1), 1e-12))[:, None]
    x[:, cols] = np.log10(linear + 1e-8)        # an empty layer stays at -8
    x[:, index["logE_tot"]] = np.log10(linear.sum(axis=1) + 1e-8)
    return x, names, index


def test_valid_showers_pass_every_check():
    x, names, _ = _valid_showers()
    for label, value in run_checks(x, names).items():
        assert value == 0.0, f"{label} fired on valid showers ({value})"


@pytest.mark.parametrize("label,column,value", [
    ("widths and radial centres >= 0", "width_eta_3", -0.01),
    ("layer energy >= empty", "logE_layer_3", -9.0),
    ("sparsity within [0, 1]", "sparsity_3", 1.5),
    ("sparsity on the 1/144 lattice", "sparsity_3", 0.5123),
])
def test_each_check_fires_on_its_own_defect(label, column, value):
    x, names, index = _valid_showers()
    x[0, index[column]] = value
    fired = {k for k, v in run_checks(x, names).items() if v > 0}
    assert label in fired, f"{label} missed {column}={value}"


def test_empty_layer_with_a_live_centre_is_caught():
    """The defect that cost 0.076 in AUC: no energy, but a centre anyway."""
    x, names, index = _valid_showers()
    x[0, index["EC_eta_41"]] = 0.3            # layer 41 is empty
    checks = run_checks(x, names)
    assert checks["empty layer empty in every column"] == pytest.approx(1 / 50)
    assert checks["widths and radial centres >= 0"] == 0.0


def test_a_lit_layer_with_no_lit_voxel_is_caught():
    """The mirror of the empty-layer check: energy has to be somewhere."""
    x, names, index = _valid_showers()
    x[0, index["sparsity_10"]] = 1.0          # layer 10 has energy
    checks = run_checks(x, names)
    assert checks["lit layer has a lit voxel"] == pytest.approx(1 / 50)
    assert checks["empty layer empty in every column"] == 0.0


def test_an_empty_layer_with_sparsity_one_is_not_flagged():
    """Sparsity 1 is correct when the layer really is empty."""
    x, names, index = _valid_showers(empty_layers=(40, 41))
    assert run_checks(x, names)["lit layer has a lit voxel"] == 0.0


def test_total_energy_must_be_the_sum_of_the_layers():
    x, names, index = _valid_showers()
    x[0, index["logE_tot"]] += 0.3            # 2x too much energy overall
    checks = run_checks(x, names)
    assert checks["E_tot = sum of layer energies"] == pytest.approx(1 / 50)


def test_non_finite_rows_are_counted_not_crashed_on():
    x, names, index = _valid_showers()
    x[0, index["logE_layer_5"]] = np.nan
    x[1, index["width_eta_5"]] = np.inf
    checks = run_checks(x, names)
    assert checks["all values finite"] == pytest.approx(2 / 50)
    # a NaN must not silently pass the arithmetic checks either
    assert checks["E_tot = sum of layer energies"] > 0


def test_every_check_has_a_test():
    """Guard against adding a check with no coverage."""
    covered = {
        "all values finite", "widths and radial centres >= 0",
        "layer energy >= empty", "sparsity within [0, 1]",
        "sparsity on the 1/144 lattice",
        "empty layer empty in every column", "lit layer has a lit voxel",
        "E_tot = sum of layer energies", "deposits less energy than it got",
    }
    assert {label for label, _ in CHECKS} == covered


def test_a_shower_cannot_deposit_more_than_it_received():
    """The check that was missing: consistency is not conservation."""
    x, names, index = _valid_showers()
    # a shower that deposits 10^6 times the beam energy, consistently
    x[0, index["logE_tot"]] = x[0, index["logE_inc"]] + 6.0
    checks = run_checks(x, names)
    assert checks["deposits less energy than it got"] == pytest.approx(1 / 50)


def test_normal_showers_pass_the_conservation_check():
    x, names, index = _valid_showers()
    # put every total just under the beam energy, as a real shower is
    x[:, index["logE_tot"]] = x[:, index["logE_inc"]] - 0.11
    assert run_checks(x, names)["deposits less energy than it got"] == 0.0
