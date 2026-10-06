"""The pairing rules behind every floor: disjoint, and matched when asked.

These are the assumptions the floors rest on. If pairs silently share showers,
the spread across repeats shrinks and every floor looks tighter than it is.
"""

import numpy as np
import pytest

from pinnde_eval.floors import (METRICS, match_by_energy, pairs_across_files,
                                pairs_matched_energy, pairs_within_file,
                                score_pairs)


def test_match_by_energy_takes_the_nearest_free_shower():
    pool = np.array([0.0, 1.0, 2.0, 3.0])
    picks, worst = match_by_energy(np.array([2.9, 1.1]), pool)
    np.testing.assert_array_equal(pool[picks], [3.0, 1.0])
    assert worst == pytest.approx(0.1)


def test_match_by_energy_never_reuses_a_shower():
    """Two targets at the same energy must take different partners."""
    pool = np.array([0.0, 0.5, 1.0])
    picks, _ = match_by_energy(np.array([0.4, 0.4]), pool)
    assert len(set(picks.tolist())) == 2


def test_match_by_energy_beats_the_file_to_file_shift():
    rng = np.random.default_rng(0)
    pool, target = rng.uniform(0, 7, 40000), rng.uniform(0, 7, 4000)
    picks, worst = match_by_energy(target, pool)
    assert len(set(picks.tolist())) == len(picks)
    # the two ds2 files differ by 1.8e-2 in mean log E_inc; matching must be
    # far tighter than that or it is not removing the energy difference
    assert worst < 1e-2


def test_pairs_are_disjoint():
    a = np.arange(1000).reshape(-1, 1).astype(float)
    b = (np.arange(1000) + 10000).reshape(-1, 1).astype(float)

    across = pairs_across_files(a, b, n=100, repeats=5)
    seen = np.concatenate([np.concatenate([x.ravel(), y.ravel()])
                           for x, y in across])
    assert len(set(seen.tolist())) == len(seen)

    within = pairs_within_file(a, b, n=100, repeats=4)
    seen = np.concatenate([np.concatenate([x.ravel(), y.ravel()])
                           for x, y in within])
    assert len(set(seen.tolist())) == len(seen)


def test_matched_pairs_are_disjoint_and_aligned_in_energy():
    rng = np.random.default_rng(1)
    # The pool must be dense enough that a near-exact partner exists at all:
    # the worst match can never beat the largest gap between pool energies,
    # which is about range * log(size) / size.
    log_pool = np.sort(rng.uniform(0, 7, 20000))
    pool = log_pool.reshape(-1, 1)
    log_target = rng.uniform(0, 7, 900)
    target = log_target.reshape(-1, 1)

    pairs = pairs_matched_energy(target, log_target, pool, log_pool, n=300,
                                 repeats=3)
    partners = np.concatenate([y.ravel() for _, y in pairs])
    assert len(set(partners.tolist())) == len(partners)
    for (x, y) in pairs:
        assert np.abs(x.ravel() - y.ravel()).max() < 1e-2


def test_score_pairs_reports_every_metric():
    rng = np.random.default_rng(2)
    pairs = [(rng.normal(size=(200, 3)), rng.normal(size=(200, 3)))]
    got = score_pairs(pairs, tier="monitor", quiet=True)
    assert set(got) == set(METRICS)
    assert np.isfinite(got["swd"][0])          # monitor tier computes this
    assert np.isnan(got["auc"][0])             # and leaves the rest unset
