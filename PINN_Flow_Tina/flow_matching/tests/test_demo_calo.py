"""Checks on the pieces of ``demo_calo`` that a silent bug would corrupt.

A generator that scores badly is only informative if everything between the
data and the score is known to be right. These tests pin down that plumbing:
the feature transform must be lossless, dequantization must stay inside the
lattice, the floors must cover every space, the energy conditioning must map
correctly, and real showers pushed through the whole harness must score as a
perfect generator does.
"""

import os

import numpy as np
import pytest

from flow_matching.demo_calo import (BIN_NAMES, NULL_FLOORS,
                                     OFFICIAL_SPARSITY_SPACING, FeatureTransform,
                                     drop_non_finite, energy_bin_labels,
                                     check_parameterisation,
                                     find_atoms, from_relative_energy,
                                     from_sqrt_fraction, main,
                                     normalized_log_energy,
                                     out_of_range_fraction, to_relative_energy,
                                     to_sqrt_fraction)
from pinnde_eval.calochallenge import official_feature_names
from pinnde_eval.observables import GEOMETRIES

DS2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "dataset_2_1.hdf5")


def _mixed_features(n=2000, seed=0):
    """A log-scaled column, a lattice column and two continuous ones."""
    rng = np.random.default_rng(seed)
    e_tot = np.exp(rng.uniform(np.log(50.0), np.log(8e5), n))
    spacing = 1.0 / 144
    sparsity = rng.integers(0, 145, n) * spacing
    other = rng.normal(size=(n, 2))
    names = ["E_tot", "sparsity_3", "a", "b"]
    return np.column_stack([e_tot, sparsity, other]), names, spacing


def test_transform_round_trip_is_lossless():
    x, names, spacing = _mixed_features()
    tf = FeatureTransform(x, names, spacings={"sparsity_3": spacing})
    back = tf.inverse(tf.forward(x, dequantize=False), quantize=False)
    np.testing.assert_allclose(back, x, rtol=1e-12, atol=1e-9)


def test_dequantized_values_return_to_their_lattice_point():
    """Spread over the cell for training, snapped back exactly for sampling."""
    x, names, spacing = _mixed_features()
    tf = FeatureTransform(x, names, spacings={"sparsity_3": spacing})
    z = tf.forward(x, dequantize=True)
    spread = tf.inverse(z, quantize=False)[:, 1]
    assert np.all(spread >= x[:, 1] - 1e-9)
    assert np.all(spread < x[:, 1] + spacing + 1e-9)
    snapped = tf.inverse(z, quantize=True)[:, 1]
    np.testing.assert_allclose(snapped, x[:, 1], atol=1e-9)


def _data_like_lattice(n=4000, seed=3):
    """Sparsity computed the way the data computes it: 1 - lit / 144."""
    rng = np.random.default_rng(seed)
    lit = rng.integers(0, 145, n)
    sparsity = 1.0 - lit / 144.0
    other = rng.normal(size=n)
    return np.column_stack([sparsity, other]), ["sparsity_7", "a"]


def test_quantize_returns_the_data_values_bit_for_bit():
    """A perfect generator must come back identical, not equal to 1e-16.

    ``floor(x / spacing) * spacing`` differs from ``1 - lit / 144`` in the last
    bit for about 40% of lattice points, and KS on ds2 sparsity then rejects a
    perfect generator at p = 5e-25. Only an exact match is safe.
    """
    x, names = _data_like_lattice()
    tf = FeatureTransform(x, names, spacings={"sparsity_7": 1.0 / 144})
    back = tf.inverse(tf.forward(x, dequantize=True), quantize=True)
    assert np.array_equal(back[:, 0], x[:, 0])


def test_quantize_leaves_impossible_values_visible():
    """Snapping must not pull a sparsity above 1 back into range."""
    x, names = _data_like_lattice()
    tf = FeatureTransform(x, names, spacings={"sparsity_7": 1.0 / 144})
    z = tf.forward(np.array([[1.02, 0.0], [0.5, 0.0]]), dequantize=False)
    back = tf.inverse(z, quantize=True)
    assert back[0, 0] > 1.0
    assert back[1, 0] == pytest.approx(0.5, abs=1.0 / 144)


def test_range_check_ignores_floating_point_noise_but_not_overshoots():
    real = np.array([[0.0, -8.0], [0.2, 3.0], [0.1, 1.0]])
    noise = np.array([[-7e-18, -8.000000000000002], [0.2, 3.0]])
    overshoot = np.array([[-0.01, -8.3], [0.2, 3.0]])
    np.testing.assert_array_equal(out_of_range_fraction(real, noise), [0.0, 0.0])
    np.testing.assert_array_equal(out_of_range_fraction(real, overshoot), [0.5, 0.5])


def test_diverged_showers_are_dropped_with_their_real_partners():
    """Rows stay aligned: generated shower i was made at real shower i's energy."""
    real = np.arange(12, dtype=float).reshape(6, 2)
    gen = real + 100.0
    gen[1, 0] = np.nan
    gen[4, 1] = np.inf
    cond = np.arange(6, dtype=float).reshape(6, 1)
    e_inc = np.arange(6, dtype=float) * 10

    r, g, c, e, dropped = drop_non_finite(real, gen, cond, e_inc)
    assert dropped == 2
    np.testing.assert_array_equal(g, r + 100.0)          # still paired row by row
    np.testing.assert_array_equal(c.ravel(), [0, 2, 3, 5])
    np.testing.assert_array_equal(e, [0, 20, 30, 50])


def test_nothing_is_dropped_from_a_finite_sample():
    real = np.ones((5, 3))
    _, g, _, _, dropped = drop_non_finite(real, real * 2, np.ones((5, 1)), np.ones(5))
    assert dropped == 0 and len(g) == 5


def test_log_column_is_logged_and_others_are_not():
    x, names, spacing = _mixed_features()
    tf = FeatureTransform(x, names, spacings={"sparsity_3": spacing})
    assert tf.log_idx == [names.index("E_tot")]


def _widths_with_a_spike_at_zero(n=600, seed=5):
    rng = np.random.default_rng(seed)
    width = np.abs(rng.normal(scale=0.08, size=n))
    width[: n // 4] = 0.0                      # 25% exactly zero, as in ds2
    return np.column_stack([width, rng.normal(size=n)]), ["width_eta_3", "a"]


def test_sqrt_columns_round_trip_exactly_including_the_zeros():
    x, names = _widths_with_a_spike_at_zero()
    tf = FeatureTransform(x, names, sqrt_features=("width_eta_3",))
    back = tf.inverse(tf.forward(x, dequantize=False), quantize=False)
    np.testing.assert_allclose(back, x, rtol=0, atol=1e-12)
    assert (back[: len(x) // 4, 0] == 0.0).all()


def test_sqrt_columns_can_never_come_out_negative():
    """Whatever the flow emits, an impossible width is impossible to produce."""
    x, names = _widths_with_a_spike_at_zero()
    tf = FeatureTransform(x, names, sqrt_features=("width_eta_3",))
    wild = np.random.default_rng(6).normal(scale=5.0, size=(400, 2))
    out = tf.inverse(wild, quantize=False)
    assert (out[:, 0] >= 0.0).all()
    # without the transform the same draws do go negative, which is the bug
    plain = FeatureTransform(x, names)
    assert (plain.inverse(wild, quantize=False)[:, 0] < 0).any()


def test_official_sparsity_spacing_matches_the_ds2_geometry():
    geom = GEOMETRIES["ds2"]
    assert OFFICIAL_SPARSITY_SPACING == pytest.approx(1.0 / (geom.n_alpha * geom.n_r))


def test_floors_cover_every_feature_space():
    for d in (7, 187, 362):
        assert set(NULL_FLOORS[d]) == {"auc", "chi2_mean", "swd", "w1_mean", "sep_mean"}
        assert abs(NULL_FLOORS[d]["auc"] - 0.5) < 0.01


def test_energy_conditioning_uses_the_training_range():
    train = np.array([1e3, 1e4, 1e6])
    c_train, lo, hi = normalized_log_energy(train)
    assert c_train.min() == pytest.approx(0.0) and c_train.max() == pytest.approx(1.0)
    c_eval, _, _ = normalized_log_energy(np.array([1e5]), lo, hi)
    expected = (np.log(1e5) - np.log(1e3)) / (np.log(1e6) - np.log(1e3))
    assert c_eval.item() == pytest.approx(expected)


def test_energy_bins_are_quartiles_of_the_condition():
    labels = energy_bin_labels(np.array([0.1, 0.3, 0.6, 0.9]))
    assert list(labels) == list(BIN_NAMES)


def _official_like(n=500, seed=4):
    """Rows shaped like the official features, with consistent energies."""
    rng = np.random.default_rng(seed)
    names = official_feature_names()
    x = rng.normal(size=(n, len(names)))
    x[:, names.index("logE_inc")] = rng.uniform(3.0, 6.0, n)
    return x, names


def test_relative_energy_is_exactly_reversible():
    x, names = _official_like()
    inc = x[:, names.index("logE_inc")]
    y = to_relative_energy(x, names)
    assert y.shape[1] == len(names) - 1          # the condition is not modelled
    np.testing.assert_allclose(from_relative_energy(y, names, inc), x,
                               rtol=0, atol=1e-12)


def test_relative_energy_removes_the_incident_energy_from_energies_only():
    x, names = _official_like()
    inc = x[:, names.index("logE_inc")]
    y = to_relative_energy(x, names)
    kept = [n for n in names if n != "logE_inc"]
    np.testing.assert_allclose(y[:, kept.index("logE_tot")],
                               x[:, names.index("logE_tot")] - inc)
    np.testing.assert_allclose(y[:, kept.index("logE_layer_20")],
                               x[:, names.index("logE_layer_20")] - inc)
    np.testing.assert_array_equal(y[:, kept.index("width_eta_20")],
                                  x[:, names.index("width_eta_20")])


def test_total_mode_shifts_only_the_total_energy():
    x, names = _official_like()
    inc = x[:, names.index("logE_inc")]
    y = to_relative_energy(x, names, which="total")
    kept = [n for n in names if n != "logE_inc"]
    np.testing.assert_allclose(y[:, kept.index("logE_tot")],
                               x[:, names.index("logE_tot")] - inc)
    np.testing.assert_array_equal(y[:, kept.index("logE_layer_20")],
                                  x[:, names.index("logE_layer_20")])
    np.testing.assert_allclose(from_relative_energy(y, names, inc, which="total"),
                               x, rtol=0, atol=1e-12)


def test_unknown_relative_mode_is_rejected():
    x, names = _official_like(n=5)
    with pytest.raises(ValueError):
        to_relative_energy(x, names, which="layers")


def test_decoded_incident_energy_is_the_condition_not_a_guess():
    """Whatever the model emits, E_inc in the output is the value it was given."""
    x, names = _official_like()
    inc = x[:, names.index("logE_inc")]
    noisy = to_relative_energy(x, names) + 0.3        # a model's imperfect output
    decoded = from_relative_energy(noisy, names, inc)
    np.testing.assert_array_equal(decoded[:, names.index("logE_inc")], inc)


def _with_empty_layers(n=400, share=0.3, seed=7):
    """Official-shaped rows where some layers are exactly empty."""
    x, names = _official_like(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    cols = [names.index(f"logE_layer_{i}") for i in (40, 41, 42)]
    x[:, cols] = np.abs(x[:, cols]) + 1.0            # non-empty values sit above
    empty = rng.random((n, len(cols))) < share
    for k, c in enumerate(cols):
        x[empty[:, k], c] = -8.0                     # the official empty value
    return x, names, cols, empty


def test_find_atoms_finds_the_empty_layers_and_the_gap():
    x, names, cols, empty = _with_empty_layers()
    atoms = find_atoms(x)
    assert sorted(atoms) == sorted(cols)             # no other column has a mass
    for c in cols:
        value, hi = atoms[c]
        assert value == -8.0
        assert hi == x[x[:, c] > -8.0, c].min()      # the gap stops at real data


def test_find_atoms_ignores_a_mass_below_the_threshold():
    x, _, cols, _ = _with_empty_layers(share=0.02)
    assert find_atoms(x, min_share=0.05) == {}
    assert sorted(find_atoms(x, min_share=0.01)) == sorted(cols)


def test_a_unique_smallest_value_is_not_an_atom():
    """Otherwise any threshold below 1/n makes every column an atom."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4))
    assert find_atoms(x, min_share=0.0) == {}


def test_atom_survives_the_transform_round_trip_exactly():
    """Real showers must come back bit-exact, or the null check is a lie.

    Per-column behaviour only: these rows empty the energy column alone, which
    no real shower does, so the layer-consistency rule is off here. The
    physical version is ``test_real_showers_keep_their_empty_layers...``.
    """
    x, names, cols, empty = _with_empty_layers()
    tf = FeatureTransform(x, names, atoms=True, layer_consistent=False, seed=0)
    assert len(tf.atoms) == len(cols)
    back = tf.inverse(tf.forward(x))
    for k, c in enumerate(cols):
        np.testing.assert_array_equal(back[empty[:, k], c], -8.0)
    np.testing.assert_allclose(back, x, rtol=0, atol=1e-9)


def test_dequantized_atom_lands_in_the_gap_and_nowhere_else():
    x, names, cols, empty = _with_empty_layers()
    tf = FeatureTransform(x, names, atoms=True, seed=0)
    spread = tf._dequantize(x)
    for k, c in enumerate(cols):
        value, hi = tf.atoms[c]
        moved = spread[empty[:, k], c]
        assert np.all((moved > value) & (moved < hi))     # inside the empty gap
        # rows that were not empty are untouched
        np.testing.assert_array_equal(spread[~empty[:, k], c], x[~empty[:, k], c])


def test_snapping_turns_a_near_miss_into_an_exactly_empty_layer():
    """What the model actually emits: values just under, or just over, the atom."""
    x, names, cols, _ = _with_empty_layers()
    tf = FeatureTransform(x, names, atoms=True, seed=0)
    c = cols[0]
    value, hi = tf.atoms[c]
    y = x.copy()
    y[:4, c] = [-9.5, -8.0, value + 0.3 * (hi - value), hi + 0.5]
    snapped = tf._quantize(y)
    np.testing.assert_array_equal(snapped[:3, c], -8.0)   # all three read "empty"
    assert snapped[3, c] == hi + 0.5                      # a real value is kept


def _one_layer_official(n=400, share=0.3, layer=44, seed=11):
    """Rows with a full column block for one layer, sometimes empty."""
    rng = np.random.default_rng(seed)
    names = official_feature_names()
    x = rng.normal(size=(n, len(names)))
    x[:, names.index("logE_inc")] = rng.uniform(3.0, 6.0, n)
    tags = ["logE_layer", "EC_eta", "EC_phi", "width_eta", "width_phi",
            "EC_R", "width_R", "sparsity"]
    cols = {t: names.index(f"{t}_{layer}") for t in tags}
    x[:, cols["logE_layer"]] = rng.uniform(1.0, 4.0, n)        # above the atom
    for t in ("width_eta", "width_phi", "EC_R", "width_R"):
        x[:, cols[t]] = rng.uniform(0.5, 3.0, n)               # non-negative
    x[:, cols["sparsity"]] = rng.uniform(0.1, 0.9, n)
    empty = rng.random(n) < share
    x[empty, cols["logE_layer"]] = -8.0
    for t in ("EC_eta", "EC_phi", "width_eta", "width_phi", "EC_R", "width_R"):
        x[empty, cols[t]] = 0.0
    x[empty, cols["sparsity"]] = 1.0
    return x, names, cols, empty


def test_real_showers_keep_their_empty_layers_through_the_round_trip():
    x, names, cols, empty = _one_layer_official()
    tf = FeatureTransform(x, names, atoms=True, seed=0)
    back = tf.inverse(tf.forward(x))
    np.testing.assert_array_equal(back[empty, cols["logE_layer"]], -8.0)
    np.testing.assert_array_equal(back[empty, cols["EC_eta"]], 0.0)
    np.testing.assert_array_equal(back[empty, cols["sparsity"]], 1.0)
    # and a layer that was not empty is not emptied
    assert np.all(back[~empty, cols["logE_layer"]] > -8.0)


def test_an_empty_layer_is_emptied_in_every_column():
    """The failure this rule exists for: a generated layer with no energy but
    a non-zero centre, which no real shower contains."""
    x, names, cols, _ = _one_layer_official()
    tf = FeatureTransform(x, names, atoms=True, seed=0)
    y = x.copy()
    y[:3, cols["logE_layer"]] = [-8.0, -9.2, tf.atoms[cols["logE_layer"]][0]]
    y[:3, cols["EC_eta"]] = 0.7                     # inconsistent, as generated
    y[:3, cols["width_eta"]] = 1.3
    y[:3, cols["sparsity"]] = 0.4
    out = tf._quantize(y)
    np.testing.assert_array_equal(out[:3, cols["EC_eta"]], 0.0)
    np.testing.assert_array_equal(out[:3, cols["width_eta"]], 0.0)
    np.testing.assert_array_equal(out[:3, cols["sparsity"]], 1.0)
    np.testing.assert_array_equal(out[:3, cols["logE_layer"]], -8.0)


def test_layer_consistency_can_be_switched_off_for_the_ablation():
    x, names, cols, _ = _one_layer_official()
    tf = FeatureTransform(x, names, atoms=True, layer_consistent=False, seed=0)
    assert tf.blocks == []
    y = x.copy()
    y[:3, cols["logE_layer"]] = -9.0
    y[:3, cols["EC_eta"]] = 0.7
    out = tf._quantize(y)
    np.testing.assert_array_equal(out[:3, cols["logE_layer"]], -8.0)  # atom only
    np.testing.assert_array_equal(out[:3, cols["EC_eta"]], 0.7)       # untouched


def test_sqrt_fraction_is_reversible_and_touches_energies_only():
    x, names = _official_like()
    e_inc = 10.0 ** x[:, names.index("logE_inc")]
    y = to_sqrt_fraction(x, names, e_inc)
    back = from_sqrt_fraction(y, names, e_inc)
    # atol, not rtol: these are log10 values, and a log that happens to sit near
    # 0 has no meaningful relative scale. 1e-9 in log10 is 2 parts per billion
    # of energy.
    np.testing.assert_allclose(back, x, rtol=0, atol=1e-9)
    # the 45 layer energies changed; everything else is untouched
    changed = [n for n, same in zip(names, np.all(y == x, axis=0)) if not same]
    assert changed == [f"logE_layer_{i}" for i in range(45)]


def test_sqrt_fraction_keeps_an_empty_layer_exactly_empty():
    """log10(0 + 1e-8) = -8 is the official value for an empty layer."""
    x, names = _official_like(n=3)
    col = names.index("logE_layer_7")
    x[:, col] = -8.0
    e_inc = 10.0 ** x[:, names.index("logE_inc")]
    y = to_sqrt_fraction(x, names, e_inc)
    np.testing.assert_array_equal(y[:, col], np.zeros(3))
    np.testing.assert_array_equal(from_sqrt_fraction(y, names, e_inc)[:, col],
                                  np.full(3, -8.0))


def test_sqrt_fraction_makes_impossible_energies_unreachable():
    """The reason for the transform: a flow's negative output is still a
    physical energy here, whereas in log space it is a layer holding less than
    nothing (24% of generated deep layers landed below -8)."""
    x, names = _official_like(n=200)
    e_inc = 10.0 ** x[:, names.index("logE_inc")]
    cols = [names.index(f"logE_layer_{i}") for i in range(45)]
    y = to_sqrt_fraction(x, names, e_inc)
    y[:, cols] -= 5.0                        # push every layer far negative
    decoded = from_sqrt_fraction(y, names, e_inc)
    assert np.all(decoded[:, cols] >= -8.0)
    energy = 10.0 ** decoded[:, cols] - 1e-8
    assert np.all(energy >= -1e-12)


def test_total_from_layers_matches_real_data_and_fixes_a_wrong_total():
    from flow_matching.demo_calo import total_from_layers
    x, names, cols, empty = _one_layer_official(n=60)
    # make every total consistent first, the way the real features are
    e = [names.index(f"logE_layer_{i}") for i in range(45)]
    x[:, names.index("logE_tot")] = np.log10(
        np.sum(np.clip(10.0 ** x[:, e] - 1e-8, 0, None), axis=1) + 1e-8)
    np.testing.assert_allclose(total_from_layers(x, names), x, rtol=0, atol=1e-9)

    broken = x.copy()
    broken[:, names.index("logE_tot")] += 0.4       # 2.5x too much energy
    fixed = total_from_layers(broken, names)
    np.testing.assert_allclose(fixed[:, names.index("logE_tot")],
                               x[:, names.index("logE_tot")], rtol=0, atol=1e-9)
    # only that column is touched
    other = [i for i in range(len(names)) if i != names.index("logE_tot")]
    np.testing.assert_array_equal(fixed[:, other], broken[:, other])


def test_derive_total_and_relative_energy_are_rejected_together():
    with pytest.raises(SystemExit):
        check_parameterisation("official", "total", False, derive_total=True)


def test_derive_total_needs_the_official_space():
    with pytest.raises(SystemExit):
        check_parameterisation("core", None, False, derive_total=True)


def test_energy_sqrt_rejects_the_conflicting_relative_mode():
    with pytest.raises(SystemExit):
        main(features="official", energy_sqrt=True, relative_energy="all",
             n_train=10, n_eval=10, n_steps=1)


def test_relative_energy_path_runs_end_to_end():
    """A tiny training run through the whole relative-energy path.

    Checks shapes and plumbing only; the model is far too small to be good.
    """
    if not os.path.exists(DS2) or not os.path.exists(DS2.replace("_1", "_2")):
        pytest.skip("both ds2 files are needed")
    try:
        from pinnde_eval.calochallenge import ensure_official_code
        ensure_official_code()
    except Exception as exc:
        pytest.skip(f"CaloChallenge code unavailable: {exc}")
    _, res = main(features="official", relative_energy=True, n_train=2000,
                  n_eval=800, n_steps=30, hidden=32, depth=2, ode_steps=5,
                  data_dir=os.path.dirname(DS2))
    assert 0.0 <= res["auc"][0] <= 1.0
    assert len(res["sep_per_feature"]) == 362       # scored in the full space


def test_real_showers_through_the_harness_score_at_the_floor():
    """The whole scoring path, with a perfect generator in the model's place.

    If this lands far from AUC 0.5 the harness is manufacturing differences,
    and no model result from it can be trusted.
    """
    if not os.path.exists(DS2) or not os.path.exists(DS2.replace("_1", "_2")):
        pytest.skip("both ds2 files are needed")
    _, res = main(n_train=6000, n_eval=2000, null=True,
                  data_dir=os.path.dirname(DS2))
    assert abs(res["auc"][0] - 0.5) < 0.03
    assert res["ks_p_combined"] > 0.01


def _rank_gauss_data(n=4000, seed=5):
    """Physically shaped rows: widths and radii non-negative, sparsity in [0,1].

    The fixture has to obey the rules itself -- the transform reproduces the
    training marginal exactly, so a fixture with negative widths would produce
    negative widths and the test would be checking the fixture, not the code.
    """
    rng = np.random.default_rng(seed)
    names = official_feature_names()
    x = np.zeros((n, len(names)))
    x[:, names.index("logE_inc")] = rng.uniform(3.0, 6.0, n)
    x[:, names.index("logE_tot")] = rng.uniform(3.0, 5.5, n)
    empty = rng.random(n) < 0.4
    for layer in range(45):
        x[:, names.index(f"logE_layer_{layer}")] = rng.uniform(0.5, 3.5, n)
        for tag in ("EC_eta", "EC_phi"):
            x[:, names.index(f"{tag}_{layer}")] = rng.normal(0, 0.05, n)
        for tag in ("width_eta", "width_phi", "EC_R", "width_R"):
            x[:, names.index(f"{tag}_{layer}")] = np.abs(rng.normal(0, 0.08, n))
        x[:, names.index(f"sparsity_{layer}")] = 1.0 - rng.integers(0, 145, n) / 144.0
    for tag, value in (("logE_layer", -8.0), ("width_eta", 0.0), ("width_phi", 0.0),
                       ("EC_R", 0.0), ("width_R", 0.0), ("sparsity", 1.0)):
        x[empty, names.index(f"{tag}_44")] = value
    return x, names, empty


def test_rank_gauss_makes_every_marginal_standard_normal():
    from flow_matching.demo_calo import RankGaussTransform
    x, names, _ = _rank_gauss_data()
    z = RankGaussTransform(x, names, seed=0).forward(x)
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05
    # every column, not just the average of them
    assert np.all(np.abs(z.mean(axis=0)) < 0.15)
    assert np.all(np.abs(z.std(axis=0) - 1.0) < 0.15)


def test_rank_gauss_returns_the_training_marginal_from_pure_noise():
    """The point of it: the marginal is right whatever the model learned."""
    from flow_matching.demo_calo import RankGaussTransform
    x, names, _ = _rank_gauss_data()
    tf = RankGaussTransform(x, names, seed=0)
    gen = tf.inverse(np.random.default_rng(2).normal(size=(4000, x.shape[1])))
    for col in ("logE_layer_44", "logE_tot", "width_phi_10"):
        j = names.index(col)
        a = np.percentile(x[:, j], [10, 50, 90])
        b = np.percentile(gen[:, j], [10, 50, 90])
        # sampling noise on a percentile of a few thousand draws, not a claim
        # about the transform, sets this tolerance
        np.testing.assert_allclose(b, a, rtol=0.05, atol=0.1)


def test_rank_gauss_cannot_produce_an_impossible_value():
    from flow_matching.demo_calo import RankGaussTransform
    x, names, _ = _rank_gauss_data()
    tf = RankGaussTransform(x, names, seed=0)
    # a badly behaved model, far outside the prior
    gen = tf.inverse(np.random.default_rng(3).normal(size=(2000, x.shape[1])) * 6)
    assert np.isfinite(gen).all()
    for j, n in enumerate(names):
        if n.startswith(("width_", "EC_R_")):
            assert gen[:, j].min() >= 0.0, n
        if n.startswith("sparsity_"):
            assert gen[:, j].min() >= 0.0 and gen[:, j].max() <= 1.0, n


def test_rank_gauss_reproduces_the_atom():
    from flow_matching.demo_calo import RankGaussTransform
    x, names, empty = _rank_gauss_data()
    tf = RankGaussTransform(x, names, seed=0)
    gen = tf.inverse(np.random.default_rng(4).normal(size=(5000, x.shape[1])))
    share = np.mean(gen[:, names.index("logE_layer_44")] <= -8.0 + 1e-9)
    assert abs(share - empty.mean()) < 0.05, share
