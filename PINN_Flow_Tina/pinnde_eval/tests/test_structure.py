"""The shuffle behind validate_structure must keep marginals and move blocks whole.

If it leaked showers across energy bins or split a block, the experiment would
be measuring its own damage rather than the cost of lost correlations.
"""

import numpy as np

from pinnde_eval.validate_structure import shuffle_within_energy


def _rows(n=200):
    idx = np.arange(n, dtype=float)
    # columns 0 and 1 are locked together; column 2 is its own block
    return np.column_stack([idx, 10 * idx, 100 * idx]), idx


def test_marginals_inside_each_energy_bin_are_untouched():
    feats, log_e = _rows()
    out = shuffle_within_energy(feats, log_e, [[0, 1], [2]], bin_size=20,
                                rng=np.random.default_rng(0))
    for start in range(0, 200, 20):
        block = slice(start, start + 20)
        for c in range(3):
            assert sorted(out[block, c]) == sorted(feats[block, c])


def test_a_group_moves_as_one_block():
    feats, log_e = _rows()
    out = shuffle_within_energy(feats, log_e, [[0, 1], [2]], bin_size=20,
                                rng=np.random.default_rng(1))
    np.testing.assert_array_equal(out[:, 1], 10 * out[:, 0])


def test_separate_groups_are_decoupled():
    feats, log_e = _rows()
    out = shuffle_within_energy(feats, log_e, [[0, 1], [2]], bin_size=20,
                                rng=np.random.default_rng(2))
    assert np.mean(out[:, 2] != 100 * out[:, 0]) > 0.5
