"""Tests for CaloChallenge observables and separation power.

The observable tests run on synthetic voxel grids whose answers are known
analytically -- a single lit voxel has an exact centre of gravity and zero
width -- so they assert correctness rather than merely reproducing whatever
the code currently returns. They need no downloaded data.

One integration test reads the real ds2 file if it happens to be present and
is skipped otherwise, so the suite stays runnable without a 1.4 GB download.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pinnde_eval as pe
from pinnde_eval.observables import Geometry, get_geometry

DS2 = pe.GEOMETRIES["ds2"]
DS2_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "dataset_2_1.hdf5")


def one_hot_shower(geom, layer, alpha, r, energy=1.0, n=1):
    """A batch of showers with a single lit voxel at (layer, alpha, r)."""
    x = np.zeros((n, geom.n_voxels))
    idx = (layer * geom.n_alpha + alpha) * geom.n_r + r
    x[:, idx] = energy
    return x


# --- geometry ---------------------------------------------------------------

def test_geometry_voxel_count():
    assert DS2.n_voxels == 45 * 16 * 9 == 6480
    assert pe.GEOMETRIES["ds3"].n_voxels == 45 * 50 * 18 == 40500


def test_geometry_reshape_axis_order():
    """reshape must be (N, layer, alpha, r) -- layer slowest, r fastest."""
    x = one_hot_shower(DS2, layer=3, alpha=5, r=7)
    v = DS2.reshape(x)
    assert v.shape == (1, 45, 16, 9)
    assert v[0, 3, 5, 7] == 1.0
    assert v.sum() == 1.0        # nothing landed anywhere else


def test_geometry_rejects_wrong_voxel_count():
    with pytest.raises(ValueError, match="expects 6480 voxels"):
        DS2.reshape(np.zeros((2, 100)))


def test_get_geometry_accepts_name_or_object():
    assert get_geometry("ds2") is DS2
    g = Geometry(2, 3, 4)
    assert get_geometry(g) is g
    with pytest.raises(ValueError, match="unknown geometry"):
        get_geometry("nope")


def test_geometry_rejects_bad_r_centers():
    with pytest.raises(ValueError, match="r_centers must have shape"):
        Geometry(2, 3, 4, r_centers=[1.0, 2.0])


# --- observables on analytically known showers ------------------------------

def test_single_voxel_has_exact_cog_and_zero_width():
    x = one_hot_shower(DS2, layer=11, alpha=0, r=4, energy=2.5)
    feats, names = pe.shower_observables(x, np.array([10.0]), geometry="ds2")
    f = dict(zip(names, feats[0]))

    assert f["E_tot"] == pytest.approx(2.5)
    assert f["f_samp"] == pytest.approx(0.25)
    assert f["z_mean"] == pytest.approx(11.0)     # exactly the lit layer
    assert f["z_width"] == pytest.approx(0.0)
    assert f["r_mean"] == pytest.approx(4.5)      # bin centre = index + 0.5
    assert f["r_width"] == pytest.approx(0.0)
    assert f["sparsity"] == pytest.approx(1 - 1 / DS2.n_voxels)


def test_two_layers_give_midpoint_and_half_separation():
    """Equal energy in layers 10 and 20: <z> = 15, sigma_z = 5 exactly."""
    x = one_hot_shower(DS2, layer=10, alpha=0, r=0)
    x += one_hot_shower(DS2, layer=20, alpha=0, r=0)
    feats, names = pe.shower_observables(x, np.array([1.0]))
    f = dict(zip(names, feats[0]))
    assert f["z_mean"] == pytest.approx(15.0)
    assert f["z_width"] == pytest.approx(5.0)


def test_radial_width_uses_radial_axis_not_angular():
    """Two radii, one angle -> r_width > 0. Two angles, one radius -> 0.

    This is the test that catches a transposed flatten order: under the wrong
    ordering these two cases swap.
    """
    spread_in_r = (one_hot_shower(DS2, layer=5, alpha=0, r=0)
                   + one_hot_shower(DS2, layer=5, alpha=0, r=8))
    spread_in_alpha = (one_hot_shower(DS2, layer=5, alpha=0, r=4)
                       + one_hot_shower(DS2, layer=5, alpha=15, r=4))

    feats_r, names = pe.shower_observables(spread_in_r, np.array([1.0]))
    feats_a, _ = pe.shower_observables(spread_in_alpha, np.array([1.0]))
    r_width_r = dict(zip(names, feats_r[0]))["r_width"]
    r_width_a = dict(zip(names, feats_a[0]))["r_width"]

    assert r_width_r == pytest.approx(4.0)   # radii 0.5 and 8.5, half-spread 4
    assert r_width_a == pytest.approx(0.0)   # same radius, different angle


def test_empty_shower_gives_zeros_not_nan():
    x = np.zeros((1, DS2.n_voxels))
    feats, names = pe.shower_observables(x, np.array([100.0]))
    assert np.all(np.isfinite(feats))
    f = dict(zip(names, feats[0]))
    assert f["E_tot"] == 0.0
    assert f["z_mean"] == 0.0 and f["r_width"] == 0.0
    assert f["sparsity"] == pytest.approx(1.0)


def test_zero_incident_energy_does_not_divide_by_zero():
    x = one_hot_shower(DS2, layer=1, alpha=1, r=1)
    feats, names = pe.shower_observables(x, np.array([0.0]))
    assert np.all(np.isfinite(feats))
    assert dict(zip(names, feats[0]))["f_samp"] == 0.0


def test_threshold_zeroes_small_voxels():
    x = one_hot_shower(DS2, layer=2, alpha=2, r=2, energy=0.001)
    feats, names = pe.shower_observables(
        x, np.array([1.0]), threshold=pe.READOUT_THRESHOLD_MEV)
    assert dict(zip(names, feats[0]))["E_tot"] == 0.0


def test_include_layers_appends_layer_energies():
    x = one_hot_shower(DS2, layer=7, alpha=0, r=0, energy=3.0)
    feats, names = pe.shower_observables(x, np.array([1.0]), include_layers=True)
    assert feats.shape == (1, len(pe.observables.CORE_FEATURES) + 45)
    assert names[-45:] == [f"E_layer_{i}" for i in range(45)]
    assert feats[0][len(pe.observables.CORE_FEATURES) + 7] == pytest.approx(3.0)


def test_layer_energies_and_radial_profile_shapes():
    x = one_hot_shower(DS2, layer=4, alpha=3, r=6, energy=2.0, n=5)
    le = pe.layer_energies(x)
    rp = pe.radial_profile(x)
    assert le.shape == (5, 45) and rp.shape == (5, 9)
    assert le[0, 4] == pytest.approx(2.0) and le[0].sum() == pytest.approx(2.0)
    assert rp[0, 6] == pytest.approx(2.0)


def test_observables_rejects_mismatched_energies():
    x = one_hot_shower(DS2, 0, 0, 0, n=3)
    with pytest.raises(ValueError, match="incident_energies has 2 rows"):
        pe.shower_observables(x, np.array([1.0, 2.0]))


def test_voxel_energy_spectrum_keeps_only_above_threshold():
    x = one_hot_shower(DS2, layer=0, alpha=0, r=0, energy=5.0, n=3)
    spec = pe.voxel_energy_spectrum(x)
    assert spec.shape == (3,) and np.all(spec == 5.0)


def test_features_fn_plugs_into_evaluate():
    rng = np.random.default_rng(0)
    e_inc = rng.uniform(1e3, 1e5, size=64)
    real = rng.random((64, DS2.n_voxels)) * (rng.random((64, DS2.n_voxels)) > 0.9)
    gen = rng.random((64, DS2.n_voxels)) * (rng.random((64, DS2.n_voxels)) > 0.9)

    fn = pe.shower_features_fn(e_inc, geometry="ds2")
    res = pe.evaluate(real, gen, tier="monitor", features_fn=fn)
    assert set(res) == {"mmd", "swd"} and np.isfinite(res["swd"])


# --- separation power -------------------------------------------------------

def test_separation_power_zero_for_identical_samples():
    x = np.random.default_rng(3).normal(size=(500, 2))
    assert np.allclose(pe.separation_power(x, x.copy()), 0.0)


def test_separation_power_one_for_disjoint_samples():
    real = np.zeros((200, 1))
    gen = np.full((200, 1), 100.0)
    assert pe.separation_power(real, gen)[0] == pytest.approx(1.0)


def test_separation_power_bounded_and_symmetric():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(400, 3))
    b = rng.normal(loc=1.0, size=(400, 3))
    s_ab = pe.separation_power(a, b)
    s_ba = pe.separation_power(b, a)
    assert np.all((s_ab >= 0) & (s_ab <= 1))
    assert np.allclose(s_ab, s_ba)


def test_separation_power_grows_with_shift():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(2000, 1))
    shifts = [pe.separation_power(a, rng.normal(loc=m, size=(2000, 1)))[0]
              for m in (0.0, 0.5, 1.0, 2.0)]
    assert shifts == sorted(shifts)


def test_separation_power_has_a_positive_null_floor():
    """Two independent draws from one distribution give S > 0, not S = 0.

    This is the finite-N binning floor: the reason an absolute separation
    power cannot be read as model error without measuring the floor at that N.
    """
    rng = np.random.default_rng(6)
    a, b = rng.normal(size=(1000, 1)), rng.normal(size=(1000, 1))
    s = pe.separation_power(a, b)[0]
    assert 0.0 < s < 0.1


def test_separation_power_in_evaluate_results():
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(300, 2)), rng.normal(size=(300, 2))
    res = pe.evaluate(a, b, tier="full")
    assert "sep_per_feature" in res and "sep_mean" in res
    assert res["sep_per_feature"].shape == (2,)


# --- per-layer observables (official CaloChallenge granularity) -------------

def test_per_layer_shapes_and_names():
    x = one_hot_shower(DS2, layer=0, alpha=0, r=0, n=4)
    feats, names = pe.per_layer_observables(x)
    assert feats.shape == (4, 4 * 45) and len(names) == 4 * 45
    assert names[0] == "E_layer_0" and names[45] == "sparsity_layer_0"
    assert names[90] == "r_mean_layer_0" and names[135] == "r_width_layer_0"


def test_per_layer_isolates_the_lit_layer():
    """One lit voxel: only its layer has energy, and only it is non-sparse."""
    x = one_hot_shower(DS2, layer=20, alpha=3, r=6, energy=4.0)
    feats, names = pe.per_layer_observables(x)
    f = dict(zip(names, feats[0]))

    assert f["E_layer_20"] == pytest.approx(4.0)
    assert f["E_layer_19"] == 0.0 and f["E_layer_21"] == 0.0

    # sparsity is per layer: 143 of 144 voxels empty in the lit layer, all
    # 144 empty everywhere else.
    assert f["sparsity_layer_20"] == pytest.approx(1 - 1 / (16 * 9))
    assert f["sparsity_layer_19"] == pytest.approx(1.0)

    # radial centre resolves within the lit layer; dead layers give 0, not NaN
    assert f["r_mean_layer_20"] == pytest.approx(6.5)
    assert f["r_width_layer_20"] == pytest.approx(0.0)
    assert f["r_mean_layer_19"] == 0.0


def test_per_layer_radial_width_within_one_layer():
    x = (one_hot_shower(DS2, layer=7, alpha=0, r=1)
         + one_hot_shower(DS2, layer=7, alpha=0, r=7))
    feats, names = pe.per_layer_observables(x)
    f = dict(zip(names, feats[0]))
    assert f["r_mean_layer_7"] == pytest.approx(4.5)    # (1.5 + 7.5) / 2
    assert f["r_width_layer_7"] == pytest.approx(3.0)   # half-separation


def test_per_layer_energies_agree_with_layer_energies():
    rng = np.random.default_rng(20)
    x = rng.random((6, DS2.n_voxels))
    feats, names = pe.per_layer_observables(x)
    assert np.allclose(feats[:, :45], pe.layer_energies(x))


def test_per_layer_all_finite_on_empty_showers():
    feats, _ = pe.per_layer_observables(np.zeros((2, DS2.n_voxels)))
    assert np.all(np.isfinite(feats))


def test_features_fn_per_layer_widens_the_vector():
    rng = np.random.default_rng(21)
    x = rng.random((8, DS2.n_voxels))
    e_inc = rng.uniform(1e3, 1e5, size=8)
    core = pe.shower_features_fn(e_inc)(x)
    wide = pe.shower_features_fn(e_inc, per_layer=True)(x)
    assert core.shape == (8, 7)
    assert wide.shape == (8, 7 + 4 * 45)
    assert np.allclose(wide[:, :7], core)


# --- standardization (needed once features carry different units) -----------

def test_standardize_makes_swd_invariant_to_feature_units():
    """Rescaling one feature must not change a standardized result.

    Shower observables mix MeV energies with dimensionless sparsity; without
    standardization swd measures only the largest-scale feature.
    """
    rng = np.random.default_rng(10)
    real = rng.normal(size=(600, 3))
    gen = rng.normal(loc=0.3, size=(600, 3))

    blown = real.copy(), gen.copy()
    blown[0][:, 0] *= 1e5
    blown[1][:, 0] *= 1e5

    plain = pe.evaluate(real, gen, tier="monitor", standardize=True)
    scaled = pe.evaluate(blown[0], blown[1], tier="monitor", standardize=True)
    assert scaled["swd"] == pytest.approx(plain["swd"], rel=1e-6)

    # ...whereas without it the rescaled version is wildly different.
    raw = pe.evaluate(blown[0], blown[1], tier="monitor", standardize=False)
    assert raw["swd"] > 100 * plain["swd"]


def test_standardize_preserves_real_differences():
    """A genuine shift must survive standardization, not be normalized away."""
    rng = np.random.default_rng(11)
    real = rng.normal(size=(600, 2))
    same = rng.normal(size=(600, 2))
    shifted = rng.normal(loc=1.5, size=(600, 2))

    null = pe.evaluate(real, same, tier="monitor", standardize=True)
    signal = pe.evaluate(real, shifted, tier="monitor", standardize=True)
    assert signal["swd"] > 10 * null["swd"]


def test_heterogeneous_scale_warning(capsys):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(200, 2))
    x[:, 0] *= 1e4
    pe.evaluate(x, x.copy(), tier="monitor", standardize=False)
    assert "scale-dependent" in capsys.readouterr().out

    pe.evaluate(x, x.copy(), tier="monitor", standardize=True)
    assert "scale-dependent" not in capsys.readouterr().out


# --- report() readability at shower dimensionality --------------------------

def test_report_summarizes_long_per_feature_arrays(capsys):
    """187 observables must not print as a 187-entry row."""
    rng = np.random.default_rng(30)
    a, b = rng.normal(size=(200, 40)), rng.normal(size=(200, 40))
    pe.report(pe.evaluate(a, b, tier="full", standardize=True))
    out = capsys.readouterr().out
    assert "d=40" in out and "mean" in out and "max" in out
    assert max(len(line) for line in out.splitlines()) < 200


def test_report_prints_short_arrays_in_full(capsys):
    rng = np.random.default_rng(31)
    a, b = rng.normal(size=(200, 3)), rng.normal(size=(200, 3))
    pe.report(pe.evaluate(a, b, tier="full"))
    out = capsys.readouterr().out
    assert "d=3" not in out and out.count(",") >= 2


# --- integration with the real file (skipped when absent) -------------------

@pytest.mark.skipif(not os.path.exists(DS2_PATH), reason="ds2 file not present")
def test_real_ds2_flatten_order_and_observables():
    showers, e_inc = pe.load_calochallenge(DS2_PATH, n=200)
    assert showers.shape == (200, 6480) and e_inc.shape == (200,)

    # Radial profile must fall off monotonically from the core; the angular
    # profile must be flat to a few percent (azimuthal symmetry). This pins
    # the (layer, alpha, r) convention against the real data.
    v = DS2.reshape(showers).mean(axis=0)
    radial = v.sum(axis=(0, 1))
    angular = v.sum(axis=(0, 2))
    assert np.all(np.diff(radial) < 0)
    assert angular.std() / angular.mean() < 0.05

    feats, names = pe.shower_observables(showers, e_inc, geometry="ds2")
    f = dict(zip(names, feats.T))
    assert feats.shape == (200, 7)
    assert np.all(np.isfinite(feats))
    assert np.all((f["z_mean"] >= 0) & (f["z_mean"] < 45))
    assert np.all((f["r_mean"] >= 0) & (f["r_mean"] <= 9))
    assert np.all((f["sparsity"] >= 0) & (f["sparsity"] < 1.0))
    assert np.all(f["E_tot"] > 0)

    # Physics the extractor must reproduce, and a transposed or otherwise
    # broken geometry would not: higher incident energy lights up more voxels
    # (sparsity falls), and the shower maximum moves deeper (<z> grows like
    # log E). Both are strong, so loose thresholds still catch a real break.
    log_e = np.log(e_inc)
    assert np.corrcoef(log_e, f["sparsity"])[0, 1] < -0.8
    assert np.corrcoef(log_e, f["z_mean"])[0, 1] > 0.7
