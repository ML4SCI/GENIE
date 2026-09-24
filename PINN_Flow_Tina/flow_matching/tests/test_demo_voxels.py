"""Tests for generating raw voxels rather than summary features.

The end-to-end test is the important one: it checks that features computed from
generated voxels satisfy every physical rule automatically, which is the whole
argument for generating voxels in the first place.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from flow_matching.demo_calo import FeatureTransform
from flow_matching.demo_voxels import (EPS, N_VOXELS, from_log_voxels,
                                       load_voxels, main, to_log_voxels)

DS2 = os.path.join(os.path.dirname(__file__), "..", "..", "dataset_2_1.hdf5")


def test_log_round_trip_is_exact_including_the_zeros():
    rng = np.random.default_rng(0)
    x = np.abs(rng.normal(size=(50, 20))) * 100.0
    x[rng.random(x.shape) < 0.7] = 0.0           # the usual 70% empty
    back = from_log_voxels(to_log_voxels(x))
    np.testing.assert_allclose(back, x, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(back[x == 0.0], 0.0)   # exactly zero, still


def test_an_empty_voxel_is_exactly_minus_eight():
    assert to_log_voxels(np.zeros((2, 3)))[0, 0] == -8.0


def test_a_negative_log_decodes_to_no_energy_not_negative_energy():
    """The model can emit anything; a voxel can never hold less than nothing."""
    assert np.all(from_log_voxels(np.array([[-20.0, -9.0, -8.0]])) >= 0.0)


def test_atom_snapping_restores_exact_zeros_through_the_transform():
    rng = np.random.default_rng(1)
    x = np.abs(rng.normal(size=(400, 12))) * 10.0 + 0.5
    empty = rng.random(x.shape) < 0.6
    x[empty] = 0.0
    names = [f"voxel_{i}" for i in range(x.shape[1])]
    tf = FeatureTransform(to_log_voxels(x), names, spacings={}, atoms=True,
                          layer_consistent=False, seed=0)
    back = from_log_voxels(tf.inverse(tf.forward(to_log_voxels(x))))
    np.testing.assert_array_equal(back[empty], 0.0)
    np.testing.assert_allclose(back[~empty], x[~empty], rtol=1e-6, atol=1e-6)


def test_voxels_load_with_the_expected_shape():
    if not os.path.exists(DS2):
        pytest.skip("ds2 needed")
    x, e = load_voxels(DS2, 5)
    assert x.shape == (5, N_VOXELS) and e.shape == (5,)
    assert np.all(x >= 0.0) and np.all(e > 0.0)


def test_features_from_generated_voxels_are_internally_consistent(tmp_path):
    """Generating voxels makes the *consistency* rules hold for free.

    A model this small generates nonsense, and that is deliberate: consistency
    here comes from the features being *computed* from a voxel grid, not from
    the model being any good. Conservation is explicitly not in this set -- a
    voxel model can and does emit showers depositing far more energy than they
    received, which is why that check exists separately (DEVLOG section 26).
    """
    if not os.path.exists(DS2) or not os.path.exists(DS2.replace("_1", "_2")):
        pytest.skip("both ds2 files needed")
    try:
        from pinnde_eval.calochallenge import ensure_official_code
        ensure_official_code()
    except Exception as exc:
        pytest.skip(f"CaloChallenge code unavailable: {exc}")

    out = str(tmp_path / "voxels.npz")
    main(n_train=1500, n_eval=300, n_steps=20, hidden=32, depth=2, ode_steps=4,
         data_dir=os.path.dirname(os.path.abspath(DS2)), save_samples=out,
         plot_dir=None)

    from pinnde_eval.validate_physical import report
    rows = report(out, quiet=True)
    conservation = "deposits less energy than it got"
    for check, (real, gen) in rows.items():
        assert real == 0.0, f"{check} fires on real Geant4 showers"
        if check == conservation:
            continue                    # not free, and not expected from a toy
        assert gen == 0.0, f"{check} fires on features computed from voxels"
    assert conservation in rows, "the conservation check must still be run"


def test_generated_voxels_are_as_empty_as_real_ones(tmp_path):
    """Atom snapping has to carry over to 6480 columns, not just 362."""
    if not os.path.exists(DS2) or not os.path.exists(DS2.replace("_1", "_2")):
        pytest.skip("both ds2 files needed")
    rng = np.random.default_rng(0)
    x, _ = load_voxels(DS2, 800)
    names = [f"voxel_{i}" for i in range(x.shape[1])]
    tf = FeatureTransform(to_log_voxels(x), names, spacings={}, atoms=True,
                          layer_consistent=False, seed=0)
    # a "model" that emits standard normals in the transformed space
    z = rng.normal(size=(500, x.shape[1]))
    gen = from_log_voxels(tf.inverse(z))
    assert np.mean(gen == 0.0) > 0.2      # zeros are reachable at all
    assert np.all(gen >= 0.0)


def test_a_narrow_band_still_round_trips_real_showers_exactly():
    """Narrowing the band must not cost exactness, only noise."""
    rng = np.random.default_rng(3)
    x = np.abs(rng.normal(size=(300, 8))) * 10.0 + 0.5
    empty = rng.random(x.shape) < 0.7
    x[empty] = 0.0
    names = [f"voxel_{i}" for i in range(x.shape[1])]
    for band in (None, 1.0, 0.25):
        tf = FeatureTransform(to_log_voxels(x), names, spacings={}, atoms=True,
                              layer_consistent=False, seed=0, atom_band=band)
        back = from_log_voxels(tf.inverse(tf.forward(to_log_voxels(x))))
        np.testing.assert_array_equal(back[empty], 0.0)
        np.testing.assert_allclose(back[~empty], x[~empty], rtol=1e-6, atol=1e-6)


def test_a_narrow_band_injects_less_noise():
    """The reason for the option: less unpredictable variance in each column."""
    rng = np.random.default_rng(4)
    x = np.abs(rng.normal(size=(2000, 4))) * 10.0 + 0.5
    x[rng.random(x.shape) < 0.75] = 0.0
    names = [f"voxel_{i}" for i in range(x.shape[1])]
    lx = to_log_voxels(x)
    wide = FeatureTransform(lx, names, spacings={}, atoms=True,
                            layer_consistent=False, seed=0)
    narrow = FeatureTransform(lx, names, spacings={}, atoms=True,
                              layer_consistent=False, seed=0, atom_band=0.25)
    assert narrow._dequantize(lx).var(axis=0).mean() < \
        0.5 * wide._dequantize(lx).var(axis=0).mean()
