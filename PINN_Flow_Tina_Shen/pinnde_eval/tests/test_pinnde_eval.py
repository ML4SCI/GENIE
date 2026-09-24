"""Lightweight edge-case tests for pinnde_eval.

Run from the project folder:  pytest pinnde_eval/tests -q
"""

import os
import sys
import importlib

import numpy as np
import pytest
import torch

# Make the package importable when pytest is run from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pinnde_eval as pe
from pinnde_eval.data import gmm_params, perturb_params, sample_gmm
from pinnde_eval.tier1 import _two_sample_chi2


# ---------- input handling ----------

def test_accepts_numpy_and_torch():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 2))
    b = rng.normal(size=(500, 2))
    m_np = pe.mmd(a, b, seed=0)
    m_th = pe.mmd(torch.tensor(a), torch.tensor(b), seed=0)
    assert np.isfinite(m_np) and np.isfinite(m_th)
    assert abs(m_np - m_th) < 1e-5


def test_handles_1d_input():
    rng = np.random.default_rng(1)
    a = rng.normal(size=1000)        # 1D -> treated as (N, 1)
    b = rng.normal(size=1000)
    assert np.isfinite(pe.mmd(a, b))
    assert np.isfinite(pe.swd(a, b))
    w = pe.wasserstein_per_feature(a, b)
    assert w.shape == (1,)


def test_mismatched_dim_raises():
    with pytest.raises(ValueError):
        pe.mmd(np.zeros((10, 2)), np.zeros((10, 3)))


def test_mmd_needs_two_samples():
    with pytest.raises(ValueError):
        pe.mmd(np.zeros((1, 2)), np.zeros((5, 2)))


# ---------- Tier 3: identical -> ~0, shifted -> larger ----------

def test_mmd_swd_identical_near_zero():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1000, 3))
    # SWD between a set and its copy is exactly 0 (sorted projections match).
    assert pe.swd(x, x.copy(), seed=0) < 1e-6
    # The unbiased MMD^2 estimate is ~0 (and may be slightly negative -- that is
    # expected for the unbiased estimator and is evidence it is unbiased).
    assert abs(pe.mmd(x, x.copy(), seed=0)) < 5e-3
    # Two independent draws from the same distribution: still ~0.
    y = rng.normal(size=(1000, 3))
    assert abs(pe.mmd(x, y, seed=0)) < 5e-3


def test_mmd_swd_increase_with_shift():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1500, 3))
    y_small = x + 0.2
    y_big = x + 1.0
    assert pe.mmd(x, y_small, seed=0) < pe.mmd(x, y_big, seed=0)
    assert pe.swd(x, y_small, seed=0) < pe.swd(x, y_big, seed=0)


def test_swd_handles_unequal_sizes():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(900, 2))
    b = rng.normal(size=(1300, 2))
    assert np.isfinite(pe.swd(a, b, seed=0))


def test_distances_are_symmetric_and_order_invariant():
    rng = np.random.default_rng(9)
    a = rng.normal(size=(600, 3))
    b = rng.normal(size=(600, 3)) + np.array([0.3, -0.2, 0.1])
    a_perm = a[rng.permutation(len(a))]

    assert np.isclose(pe.mmd(a, b, seed=0), pe.mmd(b, a, seed=0), atol=1e-6)
    assert np.isclose(pe.swd(a, b, seed=0), pe.swd(b, a, seed=0), atol=1e-6)
    assert np.allclose(pe.wasserstein_per_feature(a, b),
                       pe.wasserstein_per_feature(b, a))

    assert np.isclose(pe.mmd(a, b, seed=0), pe.mmd(a_perm, b, seed=0), atol=1e-6)
    assert np.isclose(pe.swd(a, b, seed=0), pe.swd(a_perm, b, seed=0), atol=1e-6)


# ---------- Tier 1 ----------

def test_chi2_identical_near_one():
    rng = np.random.default_rng(5)
    a = rng.normal(size=20000)
    b = rng.normal(size=20000)
    edges = np.linspace(-4, 4, 51)
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)
    chi2 = _two_sample_chi2(ha, hb)
    assert 0.5 < chi2 < 1.6


def test_classifier_auc_separates():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(1500, 2))
    same = rng.normal(size=(1500, 2))
    far = rng.normal(size=(1500, 2)) + 4.0
    auc_same, _ = pe.classifier_two_sample_test(x, same, k=3, seed=0)
    auc_far, _ = pe.classifier_two_sample_test(x, far, k=3, seed=0)
    assert abs(auc_same - 0.5) < 0.07
    assert auc_far > 0.95


# ---------- evaluate() interface ----------

def test_evaluate_monitor_keys():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(800, 3))
    b = rng.normal(size=(800, 3))
    res = pe.evaluate(a, b, tier="monitor", seed=0)
    assert set(res) == {"mmd", "swd"}


def test_evaluate_full_keys():
    params = gmm_params(d=3, k=6, seed=0)
    real = sample_gmm(params, 1500, seed=1)
    gen = sample_gmm(params, 1500, seed=2)
    res = pe.evaluate(real, gen, tier="full", n_classifier=2, seed=0)
    for key in ("mmd", "swd", "auc", "chi2_per_feature", "chi2_mean",
                "w1_per_feature", "w1_mean", "fpd", "kpd"):
        assert key in res
    assert isinstance(res["auc"], tuple)
    assert res["chi2_per_feature"].shape == (3,)


def test_evaluate_features_fn_hook():
    # features_fn that keeps only the first coordinate -> metrics run on (N,1)
    rng = np.random.default_rng(8)
    a = rng.normal(size=(600, 3))
    b = rng.normal(size=(600, 3))
    res = pe.evaluate(a, b, tier="full", n_classifier=2,
                      features_fn=lambda x: np.asarray(x)[:, :1], seed=0)
    assert res["chi2_per_feature"].shape == (1,)


def test_features_fn_can_compare_physics_observables():
    # Four calorimeter-like radial cells: same total energy, different width.
    real = np.tile(np.array([2.0, 2.0, 0.0, 0.0]), (256, 1))
    gen = np.tile(np.array([2.0, 0.0, 2.0, 0.0]), (256, 1))

    def shower_observables(cells):
        cells = np.asarray(cells)
        radii = np.arange(cells.shape[1], dtype=float)
        total = cells.sum(axis=1)
        width = (cells * radii).sum(axis=1) / total
        return np.column_stack([total, width])

    obs_real = shower_observables(real)
    obs_gen = shower_observables(gen)
    w1 = pe.wasserstein_per_feature(obs_real, obs_gen)
    assert w1[0] == 0.0
    assert w1[1] > 0.4

    res = pe.evaluate(real, gen, tier="monitor", features_fn=shower_observables, seed=0)
    assert res["swd"] > 0.2


def test_evaluate_by_condition_finds_bad_slice():
    rng = np.random.default_rng(11)
    real_lo = rng.normal(size=(300, 2))
    real_hi = rng.normal(size=(300, 2))
    gen_lo = rng.normal(size=(300, 2))
    gen_hi = rng.normal(size=(300, 2)) + 1.0

    real = np.vstack([real_lo, real_hi])
    gen = np.vstack([gen_lo, gen_hi])
    labels = np.array(["low"] * 300 + ["high"] * 300)

    res = pe.evaluate_by_condition(real, gen, labels, tier="monitor", seed=0)
    assert set(res) == {"low", "high"}
    assert res["low"]["n_real"] == 300
    assert res["low"]["n_gen"] == 300
    assert res["high"]["swd"] > 3 * res["low"]["swd"]


def test_evaluate_by_condition_uses_separate_generated_labels():
    rng = np.random.default_rng(12)
    real = rng.normal(size=(6, 2))
    gen = rng.normal(size=(7, 2))
    real_labels = np.array([0, 0, 0, 1, 1, 1])
    gen_labels = np.array([0, 0, 1, 1, 1, 2, 2])

    res = pe.evaluate_by_condition(real, gen, real_labels, gen_labels,
                                   tier="monitor", min_count=2, seed=0)
    assert set(res) == {0, 1}
    assert res[0]["n_real"] == 3
    assert res[0]["n_gen"] == 2
    assert res[1]["n_real"] == 3
    assert res[1]["n_gen"] == 3


def test_evaluate_by_condition_checks_label_lengths():
    with pytest.raises(ValueError):
        pe.evaluate_by_condition(np.zeros((4, 2)), np.zeros((4, 2)),
                                 np.zeros(3), tier="monitor")


def test_evaluate_bad_tier_raises():
    with pytest.raises(ValueError):
        pe.evaluate(np.zeros((10, 2)), np.zeros((10, 2)), tier="bogus")


def test_evaluate_degrades_when_jetnet_is_missing(monkeypatch):
    eval_mod = importlib.import_module("pinnde_eval.evaluate")

    def missing_jetnet(*args, **kwargs):
        raise ImportError("jetnet unavailable")

    monkeypatch.setattr(eval_mod, "fpd", missing_jetnet)
    monkeypatch.setattr(eval_mod, "kpd", missing_jetnet)
    monkeypatch.setattr(eval_mod, "classifier_two_sample_test", lambda *args, **kwargs: (0.5, 0.0))

    rng = np.random.default_rng(10)
    real = rng.normal(size=(400, 2))
    gen = rng.normal(size=(400, 2))
    res = pe.evaluate(real, gen, tier="full", n_classifier=1, seed=0)
    assert res["fpd"] is None
    assert res["kpd"] is None
    assert np.isfinite(res["mmd"])
    assert np.isfinite(res["swd"])


# ---------- perturbations move the metrics ----------

def test_perturbation_increases_distance():
    params = gmm_params(d=3, k=8, seed=0)
    real = sample_gmm(params, 4000, seed=1)
    gen0 = sample_gmm(params, 4000, seed=2)
    gen1 = sample_gmm(perturb_params(params, "mean", 0.4), 4000, seed=2)
    assert pe.swd(real, gen1) > pe.swd(real, gen0)
    assert pe.wasserstein_per_feature(real, gen1).mean() > \
        pe.wasserstein_per_feature(real, gen0).mean()
