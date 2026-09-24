"""The CaloChallenge's own high-level features, computed with their own code.

The 2026-08-31 meeting settled this: to compare against published submissions we
must use the challenge's features and binning *exactly* as they compute them.
So this module downloads their code and calls it, instead of reimplementing the
definitions and hoping they agree.

For dataset 2 their classifier reads **362 numbers per shower**, in this order:

===========================  =======  ==========================================
column                       count    what it is
===========================  =======  ==========================================
``logE_inc``                       1  log10 of the incident energy
``logE_layer_i``                  45  log10 of the energy in layer i
``EC_eta_i`` / ``EC_phi_i``    45+45  centre of energy in eta / phi, per layer
``width_eta_i``/``width_phi_i``45+45  width in eta / phi, per layer
``logE_tot``                       1  log10 of the total deposited energy
``sparsity_i``                    45  fraction of dead voxels in layer i
``EC_R_i`` / ``width_R_i``     45+45  radial centre and width, per layer
===========================  =======  ==========================================

The eta and phi columns are the 180 we could not compute before: they need the
detector maps in ``binning_dataset_2.xml``.

Their code carries no licence, so it is **not** copied into this repository.
``ensure_official_code()`` downloads it on demand into ``calochallenge_code/``
(gitignored), pinned to one commit and checked by MD5, so the features cannot
change under us. ``test_calochallenge.py`` checks our column assembly against
their own ``prepare_high_data_for_classifier`` on real showers.

    from pinnde_eval.calochallenge import official_features_from_file
    feats, names, e_inc = official_features_from_file("dataset_2_1.hdf5", 10000)
"""

import hashlib
import importlib
import os
import sys
import urllib.request

import numpy as np

from .observables import load_calochallenge

# Pinned so the features are reproducible; from github.com/CaloChallenge/homepage
COMMIT = "3073d13897f7ccc7a30ff7b419a27a9020626d56"
RAW_URL = "https://raw.githubusercontent.com/CaloChallenge/homepage/{commit}/code/{name}"
OFFICIAL_FILES = {
    "HighLevelFeatures.py": "5db8a9bd52f5140a1e739a2b66a06a41",
    "XMLHandler.py": "690da2eee8d58f534b04b0d865149861",
    "binning_dataset_2.xml": "8507505f3a262ca17db6c78e425669b4",
    # only used by the parity test, which reads one function out of it
    "evaluate.py": "4530bc71d83490972242cf1c8e12aa39",
}

CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "calochallenge_code")


def ensure_official_code(code_dir=None, download=True):
    """Return the folder holding the official code, downloading it if needed.

    Every file is checked against the MD5 recorded above, so a truncated
    download or a changed upstream file is caught rather than quietly used.
    """
    code_dir = os.path.abspath(code_dir or CODE_DIR)
    os.makedirs(code_dir, exist_ok=True)
    for name, md5 in OFFICIAL_FILES.items():
        path = os.path.join(code_dir, name)
        if not os.path.exists(path):
            if not download:
                raise FileNotFoundError(
                    f"{path} missing; call ensure_official_code() with network access")
            url = RAW_URL.format(commit=COMMIT, name=name)
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            with open(path, "wb") as handle:
                handle.write(data)
        got = hashlib.md5(open(path, "rb").read()).hexdigest()
        if got != md5:
            raise RuntimeError(
                f"{path} has MD5 {got}, expected {md5}. Delete it and retry.")
    return code_dir


def _high_level_features(geometry="ds2", code_dir=None):
    """Instantiate their ``HighLevelFeatures`` with the ds2 detector binning."""
    if geometry != "ds2":
        raise ValueError(f"only ds2 is wired up, got {geometry}")
    code_dir = ensure_official_code(code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)          # their modules import each other
    hlf_module = importlib.import_module("HighLevelFeatures")
    return hlf_module.HighLevelFeatures(
        "electron", filename=os.path.join(code_dir, "binning_dataset_2.xml"))


def official_feature_names(n_layers=45):
    """The 362 column names, in the order their classifier receives them."""
    per_layer = ("logE_layer", "EC_eta", "EC_phi", "width_eta", "width_phi")
    names = ["logE_inc"]
    for prefix in per_layer:
        names += [f"{prefix}_{i}" for i in range(n_layers)]
    names.append("logE_tot")
    for prefix in ("sparsity", "EC_R", "width_R"):
        names += [f"{prefix}_{i}" for i in range(n_layers)]
    return names


def official_features(showers, e_inc, geometry="ds2", code_dir=None, threshold=0.0):
    """Their high-level features for ``showers`` (N, n_voxels), in MeV.

    Returns ``(features, names)`` with features of shape (N, 362) for ds2.

    The assembly below mirrors ``prepare_high_data_for_classifier`` in their
    ``evaluate.py``: the same columns, the same order, the same ``1e-8`` inside
    the logs and the same division of the eta/phi/radial columns by 100. The
    parity test compares the two on real showers.
    """
    hlf = _high_level_features(geometry, code_dir)
    showers = np.asarray(showers, dtype=np.float64)
    hlf.CalculateFeatures(showers)
    e_inc = np.asarray(e_inc, dtype=np.float64).reshape(-1, 1)

    def stack(per_layer_dict, layers):
        return np.concatenate([np.asarray(per_layer_dict[i]).reshape(-1, 1)
                               for i in layers], axis=1)

    layers = list(hlf.GetElayers())
    alpha_layers = list(hlf.layersBinnedInAlpha)
    columns = [
        np.log10(e_inc),
        np.log10(stack(hlf.GetElayers(), layers) + 1e-8),
        stack(hlf.GetECEtas(), alpha_layers) / 1e2,
        stack(hlf.GetECPhis(), alpha_layers) / 1e2,
        stack(hlf.GetWidthEtas(), alpha_layers) / 1e2,
        stack(hlf.GetWidthPhis(), alpha_layers) / 1e2,
        np.log10(np.asarray(hlf.GetEtot()).reshape(-1, 1) + 1e-8),
        stack(hlf.GetSparsity(), layers),
        stack(hlf.GetECR(), layers) / 1e2,
        stack(hlf.GetWidthR(), layers) / 1e2,
    ]
    features = np.concatenate(columns, axis=1)
    names = official_feature_names(len(layers))
    if features.shape[1] != len(names):
        raise RuntimeError(
            f"built {features.shape[1]} columns but named {len(names)}")
    return features, names


def official_features_from_file(path, n, start=0, chunk=5000, geometry="ds2",
                                code_dir=None, cache=None):
    """Features for ``n`` showers of a ds2 file, read in chunks.

    ``cache`` is an optional ``.npz`` path. A cache whose stored ``path``,
    ``start`` and ``n`` match is reused; otherwise the features are computed and
    written there. 100,000 showers are 362 columns of float64, i.e. ~290 MB, so
    caching turns a minutes-long extraction into a load of a second or two.
    """
    if cache and os.path.exists(cache):
        stored = np.load(cache, allow_pickle=False)
        same = (str(stored["source"]) == os.path.abspath(path)
                and int(stored["start"]) == start and int(stored["n"]) == n)
        if same:
            return (stored["features"], official_feature_names(),
                    stored["e_inc"])

    blocks, energies, names = [], [], None
    for begin in range(start, start + n, chunk):
        showers, e_inc = load_calochallenge(
            path, n=min(chunk, start + n - begin), start=begin)
        feats, names = official_features(showers, e_inc, geometry=geometry,
                                         code_dir=code_dir)
        blocks.append(feats)
        energies.append(e_inc)
        del showers
    features = np.vstack(blocks)
    e_inc = np.concatenate(energies)

    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        np.savez(cache, features=features, e_inc=e_inc,
                 source=os.path.abspath(path), start=start, n=n)
    return features, names, e_inc
