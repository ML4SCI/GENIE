"""Tests for the classical two-sample tests and the Sinkhorn divergence."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pinnde_eval as pe


# --- classical tests --------------------------------------------------------

def test_identical_samples_give_large_pvalues():
    x = np.random.default_rng(0).normal(size=(500, 3))
    r = pe.two_sample_tests(x, x.copy(), tests=("ks", "cvm"))
    assert np.all(r["ks"]["pvalue"] > 0.99)
    assert np.all(r["cvm"]["pvalue"] > 0.99)


def test_shifted_samples_give_small_pvalues():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2000, 3)), rng.normal(loc=0.5, size=(2000, 3))
    r = pe.two_sample_tests(a, b, tests=("ks", "cvm"))
    assert np.all(r["ks"]["pvalue"] < 1e-6)
    assert pe.combine_pvalues(r["ks"]["pvalue"]) < 1e-10  # Bonferroni


def test_pvalues_are_uniform_under_a_permutation_null():
    """The property that makes these tests worth having: calibration."""
    rng = np.random.default_rng(2)
    pooled = rng.normal(size=(4000, 2))
    pvals = []
    for _ in range(40):
        idx = rng.permutation(len(pooled))
        a, b = pooled[idx[:2000]], pooled[idx[2000:]]
        pvals.append(pe.two_sample_tests(a, b, tests=("ks",))["ks"]["pvalue"])
    _, unif_p, frac = pe.pvalue_uniformity(np.concatenate(pvals))
    assert unif_p > 0.01
    assert frac < 0.15


def test_combine_pvalues_defaults_to_bonferroni():
    """Fisher assumes independence; shower observables are correlated."""
    p = np.array([0.5, 0.5, 0.5, 0.02])
    assert pe.combine_pvalues(p) == pytest.approx(0.08)          # Bonferroni
    assert pe.combine_pvalues(p, "fisher") != pe.combine_pvalues(p)


def test_combine_pvalues_methods():
    p = np.array([0.5, 0.5, 0.5, 0.02])
    assert pe.combine_pvalues(p, "bonferroni") == pytest.approx(0.08)
    fisher = pe.combine_pvalues(p, "fisher")
    assert 0.0 < fisher < 1.0
    with pytest.raises(ValueError, match="unknown method"):
        pe.combine_pvalues(p, "nope")


def test_unknown_test_name_raises():
    x = np.zeros((10, 2))
    with pytest.raises(ValueError, match="unknown tests"):
        pe.two_sample_tests(x, x, tests=("ks", "banana"))


def test_max_samples_subsamples_deterministically():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(3000, 2)), rng.normal(size=(3000, 2))
    r1 = pe.two_sample_tests(a, b, tests=("ks",), max_samples=500, seed=7)
    r2 = pe.two_sample_tests(a, b, tests=("ks",), max_samples=500, seed=7)
    assert np.allclose(r1["ks"]["pvalue"], r2["ks"]["pvalue"])


# --- Sinkhorn ---------------------------------------------------------------

def test_sinkhorn_is_zero_for_identical_samples():
    """The debiased divergence must vanish, unlike the raw entropic cost.

    Tolerance is float32 epsilon (~1.2e-7), since the transport plan is built
    in single precision; a real signal on these clouds is ~0.07, four orders
    of magnitude above this.
    """
    x = np.random.default_rng(4).normal(size=(600, 3))
    assert pe.sinkhorn(x, x.copy()) == pytest.approx(0.0, abs=1e-5)
    assert pe.sinkhorn(x, x.copy(), debias=False) > 0.1


def test_sinkhorn_grows_with_separation():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(600, 2))
    vals = [pe.sinkhorn(a, rng.normal(loc=m, size=(600, 2)))
            for m in (0.0, 0.5, 1.0, 2.0)]
    assert vals == sorted(vals)


def test_sinkhorn_rejects_mismatched_dimension():
    with pytest.raises(ValueError, match="share feature dimension"):
        pe.sinkhorn(np.zeros((10, 2)), np.zeros((10, 3)))


def test_sinkhorn_is_scale_dependent_so_standardize_matters():
    """Documented behaviour: like swd, it needs features on a common scale."""
    rng = np.random.default_rng(6)
    a, b = rng.normal(size=(400, 3)), rng.normal(loc=0.4, size=(400, 3))
    blown_a, blown_b = a.copy(), b.copy()
    blown_a[:, 0] *= 1e4
    blown_b[:, 0] *= 1e4
    assert pe.sinkhorn(blown_a, blown_b) > 100 * pe.sinkhorn(a, b)


def test_evaluate_reports_sinkhorn_and_ks():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(400, 2)), rng.normal(size=(400, 2))
    res = pe.evaluate(a, b, tier="full")
    assert "sinkhorn" in res and np.isfinite(res["sinkhorn"])
    assert res["ks_pvalue"].shape == (2,)
    assert 0.0 <= res["ks_p_combined"] <= 1.0
