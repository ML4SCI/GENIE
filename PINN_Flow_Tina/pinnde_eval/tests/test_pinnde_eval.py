"""Lightweight edge-case tests for pinnde_eval.

Run from the project folder:  pytest pinnde_eval/tests -q
"""

import os
import sys

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


def test_evaluate_bad_tier_raises():
    with pytest.raises(ValueError):
        pe.evaluate(np.zeros((10, 2)), np.zeros((10, 2)), tier="bogus")


# ---------- perturbations move the metrics ----------

def test_perturbation_increases_distance():
    params = gmm_params(d=3, k=8, seed=0)
    real = sample_gmm(params, 4000, seed=1)
    gen0 = sample_gmm(params, 4000, seed=2)
    gen1 = sample_gmm(perturb_params(params, "mean", 0.4), 4000, seed=2)
    assert pe.swd(real, gen1) > pe.swd(real, gen0)
    assert pe.wasserstein_per_feature(real, gen1).mean() > \
        pe.wasserstein_per_feature(real, gen0).mean()
