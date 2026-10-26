"""Tests for the sample-size stability study and the local discrepancy maps.

Run from the project folder:  pytest pinnde_eval/tests -q
"""

import os
import sys

import numpy as np
import pytest

# Make the package importable when pytest is run from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pinnde_eval.local import binned_residual_map, classifier_discrepancy, mmd_witness
from pinnde_eval.stability import min_resolvable_n, separation_z, stability_study


# ---------- stability study ----------

@pytest.fixture(scope="module")
def small_study():
    # Small grid without the classifier so the whole module stays fast.
    return stability_study(d=2, k=4, n_grid=(200, 800, 3200), n_repeats=4,
                           kind="mean", eps=0.3, seed=0, include_auc=False)


def test_stability_study_shapes(small_study):
    assert list(small_study["n_grid"]) == [200, 800, 3200]
    for name in ("mmd", "swd", "w1_mean", "chi2_mean"):
        assert small_study[name]["null"].shape == (3, 4)
        assert small_study[name]["signal"].shape == (3, 4)


def test_null_floor_shrinks_with_n(small_study):
    # The SWD/W1 null floor is a finite-N artifact: it must fall as N grows.
    for name in ("swd", "w1_mean"):
        floor = small_study[name]["null"].mean(axis=1)
        assert floor[0] > floor[-1], f"{name} null floor did not shrink: {floor}"
    # The chi2 null stays ~1 at every N (it is already sample-size corrected).
    chi2 = small_study["chi2_mean"]["null"].mean(axis=1)
    assert np.all(np.abs(chi2 - 1.0) < 0.5), chi2


def test_signal_separates_at_large_n(small_study):
    # eps=0.3 mean shift: by the largest N every distance separates cleanly.
    z = separation_z(small_study)
    for name in ("mmd", "swd", "w1_mean"):
        assert z[name][-1] > 2.0, f"{name} z at max N: {z[name]}"


def test_min_resolvable_n(small_study):
    res = min_resolvable_n(small_study, z_min=2.0)
    # The transport distances must resolve this shift somewhere on the grid.
    assert res["swd"] is not None
    assert res["w1_mean"] is not None
    # An absurd threshold is resolvable nowhere -> None, not a crash.
    res_hard = min_resolvable_n(small_study, z_min=1e9)
    assert all(v is None for v in res_hard.values())


def test_stability_study_is_deterministic():
    a = stability_study(d=2, k=4, n_grid=(300,), n_repeats=2, seed=0,
                        include_auc=False)
    b = stability_study(d=2, k=4, n_grid=(300,), n_repeats=2, seed=0,
                        include_auc=False)
    assert np.array_equal(a["swd"]["null"], b["swd"]["null"])
    assert np.array_equal(a["mmd"]["signal"], b["mmd"]["signal"])


# ---------- mmd_witness ----------

def test_witness_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1500, 2))
    y = rng.normal(size=(1500, 2))
    w = mmd_witness(x, y, seed=0)
    assert w.shape == (3000,)
    assert np.abs(w).max() < 0.05


def test_witness_localizes_a_shifted_cluster():
    # real has a cluster at +4 that gen lacks; gen has one at -4 real lacks.
    rng = np.random.default_rng(1)
    shared_r = rng.normal(size=(1000, 2))
    shared_g = rng.normal(size=(1000, 2))
    real = np.vstack([shared_r, rng.normal(size=(500, 2)) + 4.0])
    gen = np.vstack([shared_g, rng.normal(size=(500, 2)) - 4.0])

    grid = np.array([[4.0, 4.0], [-4.0, -4.0], [0.0, 0.0]])
    w = mmd_witness(real, gen, points=grid, seed=0)
    assert w[0] > 0.02          # real over-dense at +4
    assert w[1] < -0.02         # gen over-dense at -4
    assert abs(w[2]) < abs(w[0]) and abs(w[2]) < abs(w[1])  # shared region ~ok


def test_witness_dim_checks():
    with pytest.raises(ValueError):
        mmd_witness(np.zeros((10, 2)), np.zeros((10, 3)))
    with pytest.raises(ValueError):
        mmd_witness(np.zeros((10, 2)), np.zeros((10, 2)), points=np.zeros((5, 3)))


# ---------- classifier_discrepancy ----------

def test_classifier_discrepancy_null_and_signal():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(600, 2))
    same = rng.normal(size=(600, 2))
    p_real, p_gen, auc = classifier_discrepancy(x, same, n_folds=3, seed=0)
    assert p_real.shape == (600,) and p_gen.shape == (600,)
    assert abs(auc - 0.5) < 0.08
    assert abs(p_real.mean() - 0.5) < 0.1
    assert abs(p_gen.mean() - 0.5) < 0.1

    far = rng.normal(size=(600, 2)) + 4.0
    p_real, p_gen, auc = classifier_discrepancy(x, far, n_folds=3, seed=0)
    assert auc > 0.95
    assert p_real.mean() > 0.8      # real confidently real
    assert p_gen.mean() < 0.2       # fakes confidently fake


def test_classifier_discrepancy_ranks_bad_samples():
    # gen = half in-distribution, half hallucinated at +5: the hallucinated
    # half must dominate the lowest P(real) scores.
    rng = np.random.default_rng(3)
    real = rng.normal(size=(800, 2))
    good = rng.normal(size=(400, 2))
    bad = rng.normal(size=(400, 2)) + 5.0
    gen = np.vstack([good, bad])

    _, p_gen, _ = classifier_discrepancy(real, gen, n_folds=3, seed=0)
    worst = np.argsort(p_gen)[:400]
    assert (worst >= 400).mean() > 0.9


# ---------- binned_residual_map ----------

def test_residual_map_null_is_standard_normal():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(8000, 2))
    b = rng.normal(size=(8000, 2))
    res, edges = binned_residual_map(a, b, features=(0, 1), bins=10)
    assert res.shape == (10, 10)
    assert len(edges) == 2 and len(edges[0]) == 11
    # Under the null each bin is ~N(0,1): std near 1, no extreme outliers.
    assert res.std() < 1.6
    assert np.abs(res).max() < 5.0


def test_residual_map_localizes_the_bad_region():
    # gen misses the +5 cluster and hallucinates one at -5 instead.
    rng = np.random.default_rng(5)
    shared_r = rng.normal(size=(3000, 2))
    shared_g = rng.normal(size=(3000, 2))
    real = np.vstack([shared_r, rng.normal(scale=0.5, size=(1000, 2)) + 5.0])
    gen = np.vstack([shared_g, rng.normal(scale=0.5, size=(1000, 2)) - 5.0])

    common = [np.linspace(-8, 8, 9), np.linspace(-8, 8, 9)]
    res, _ = binned_residual_map(real, gen, features=(0, 1), bins=8, edges=common)
    # +5 region (last bins) -> strong real excess; -5 region -> strong gen excess.
    assert res[6:, 6:].max() > 5.0
    assert res[:2, :2].min() < -5.0
    # Shared central region stays consistent with statistical fluctuation.
    assert np.abs(res[3:5, 3:5]).max() < 4.0


def test_residual_map_1d_and_validation():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(2000, 3))
    b = rng.normal(size=(2000, 3))
    res, edges = binned_residual_map(a, b, features=1, bins=15)
    assert res.shape == (15,)
    assert edges.shape == (16,)

    with pytest.raises(ValueError):
        binned_residual_map(a, b, features=(0, 1, 2))
    with pytest.raises(ValueError):
        binned_residual_map(a, b, features=(0, 7))
