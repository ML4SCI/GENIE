"""Do our 362 columns match the CaloChallenge's own classifier input?

The point of ``calochallenge.py`` is parity with the challenge, so the tests
compare against their code rather than against our expectations:

1. the columns are assembled exactly as their ``prepare_high_data_for_classifier``
   assembles them, checked on real showers;
2. the two columns we can compute independently (layer energy, layer sparsity)
   agree with ``observables.per_layer_observables``.

Both need ds2 and their code, so both skip when either is unavailable.
"""

import ast
import os

import numpy as np
import pytest

from pinnde_eval.calochallenge import (ensure_official_code, official_features,
                                       official_feature_names,
                                       official_features_from_file)
from pinnde_eval.observables import load_calochallenge, per_layer_observables

DS2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "dataset_2_1.hdf5")
N_SHOWERS = 200


def _official_code_or_skip():
    try:
        return ensure_official_code()
    except Exception as exc:                       # offline, or upstream moved
        pytest.skip(f"CaloChallenge code unavailable: {exc}")


def _showers_or_skip():
    if not os.path.exists(DS2):
        pytest.skip("dataset_2_1.hdf5 not present")
    return load_calochallenge(DS2, n=N_SHOWERS)


def test_columns_are_362_and_named():
    _official_code_or_skip()
    showers, e_inc = _showers_or_skip()
    feats, names = official_features(showers, e_inc)
    assert feats.shape == (N_SHOWERS, 362)
    assert names == official_feature_names()
    assert len(set(names)) == len(names)
    assert np.isfinite(feats).all()


def test_matches_their_own_classifier_input(tmp_path):
    """Our assembly against ``prepare_high_data_for_classifier`` itself.

    Their ``evaluate.py`` cannot simply be imported (it pulls in plotting and
    ResNet modules), so the two functions we need are lifted out of it with
    ``ast`` and executed on their own.
    """
    import h5py

    code_dir = _official_code_or_skip()
    showers, e_inc = _showers_or_skip()

    source = open(os.path.join(code_dir, "evaluate.py")).read()
    wanted = ("extract_shower_and_energy", "prepare_high_data_for_classifier")
    picked = [node for node in ast.parse(source).body
              if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert len(picked) == len(wanted)
    namespace = {"np": np}
    exec(compile(ast.Module(picked, []), "evaluate.py", "exec"), namespace)

    path = tmp_path / "sample.hdf5"
    with h5py.File(path, "w") as handle:
        handle["showers"] = showers
        handle["incident_energies"] = e_inc.reshape(-1, 1)

    import sys
    sys.path.insert(0, code_dir)
    import HighLevelFeatures as HLF

    hlf = HLF.HighLevelFeatures(
        "electron", filename=os.path.join(code_dir, "binning_dataset_2.xml"))
    hlf.CalculateFeatures(np.asarray(showers, dtype=np.float64))
    with h5py.File(path, "r") as handle:
        theirs = namespace["prepare_high_data_for_classifier"](handle, hlf, 0.)

    ours, _ = official_features(showers, e_inc)
    assert theirs.shape == (N_SHOWERS, 363)        # their last column is the label
    np.testing.assert_allclose(ours, theirs[:, :-1], rtol=1e-12, atol=0)


def test_layer_energy_and_sparsity_match_our_observables():
    """The two columns we compute ourselves must agree with theirs.

    Their sparsity is ``1 - (voxel > 0).mean()`` per layer, which for a zero
    threshold is our "fraction of empty voxels"; their layer energy is our sum
    over the layer. The eta and phi columns have no counterpart on our side --
    they are the reason this module exists.
    """
    _official_code_or_skip()
    showers, e_inc = _showers_or_skip()
    feats, names = official_features(showers, e_inc)
    mine, my_names = per_layer_observables(showers)

    column = {name: i for i, name in enumerate(names)}
    my_column = {name: i for i, name in enumerate(my_names)}
    for layer in range(45):
        theirs_e = 10 ** feats[:, column[f"logE_layer_{layer}"]] - 1e-8
        mine_e = mine[:, my_column[f"E_layer_{layer}"]]
        np.testing.assert_allclose(theirs_e, mine_e, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(feats[:, column[f"sparsity_{layer}"]],
                                   mine[:, my_column[f"sparsity_layer_{layer}"]],
                                   rtol=0, atol=1e-12)


def test_cache_round_trip(tmp_path):
    """A cached extraction returns the same numbers without recomputing."""
    _official_code_or_skip()
    if not os.path.exists(DS2):
        pytest.skip("dataset_2_1.hdf5 not present")
    cache = tmp_path / "feats.npz"
    first, names, e_inc = official_features_from_file(DS2, 100, cache=str(cache))
    assert cache.exists()
    again, names_again, e_again = official_features_from_file(DS2, 100, cache=str(cache))
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(e_inc, e_again)
    assert names == names_again
