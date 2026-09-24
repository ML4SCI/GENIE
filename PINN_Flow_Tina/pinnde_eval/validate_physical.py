"""Are the generated showers physically possible?

Every metric in this repo compares *distributions*: chi2, the classifier, the
Wasserstein distances. None of them asks whether a single generated shower could
exist at all. A model can sit near the floor on all of them and still emit a
layer holding negative energy, a width below zero, or a total energy that is not
the sum of its parts.

These checks ask that question directly, one shower at a time. The real Geant4
showers from the same file go through the identical code as a control: a check
that real data fails is a broken check, not a finding, and the report prints
both columns side by side so that is obvious.

Run it on anything saved with ``demo_calo --save-samples``:

    python -m pinnde_eval.validate_physical runs/atom.npz

Every check is stated as "fraction of rows that violate it", so 0.00% is a pass
and the real column should read 0.00% throughout.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# The official empty-layer values: an empty layer has no energy (log10 of
# 0 + 1e-8), no extent and no centre, and none of its 144 voxels is lit.
EMPTY_LOG_ENERGY = -8.0
EMPTY_VALUES = {"EC_eta": 0.0, "EC_phi": 0.0, "width_eta": 0.0,
                "width_phi": 0.0, "EC_R": 0.0, "width_R": 0.0,
                "sparsity": 1.0}
SPARSITY_SPACING = 1.0 / 144
N_LAYERS = 45


def _cols(names, prefix):
    return [i for i, n in enumerate(names) if n.startswith(prefix)]


def _frac(mask):
    """Fraction of rows with at least one violation."""
    mask = np.asarray(mask)
    return float(np.mean(mask.any(axis=1) if mask.ndim > 1 else mask))


def check_finite(x, names):
    return _frac(~np.isfinite(x))


def check_non_negative(x, names):
    """Widths and radial centres are distances: they cannot be negative."""
    cols = _cols(names, "width_") + _cols(names, "EC_R_")
    return _frac(x[:, cols] < 0)


def check_energy_floor(x, names):
    """Nothing can hold less energy than an empty layer does."""
    cols = _cols(names, "logE_layer_")
    return _frac(x[:, cols] < EMPTY_LOG_ENERGY - 1e-9)


def check_sparsity_range(x, names):
    cols = _cols(names, "sparsity_")
    v = x[:, cols]
    return _frac((v > 1.0 + 1e-9) | (v < 0.0))


def check_sparsity_lattice(x, names):
    """A layer has 144 voxels, so sparsity can only be a multiple of 1/144."""
    cols = _cols(names, "sparsity_")
    v = x[:, cols] / SPARSITY_SPACING
    return _frac(np.abs(v - np.round(v)) > 1e-6)


def check_empty_layer_consistency(x, names):
    """A layer with no energy has no centre, no width and no lit voxel.

    True in 100% of real ds2 showers. A generated layer with no energy but a
    non-zero centre is the most detectable defect measured so far: it
    took the classifier from 0.892 to 0.968 (DEVLOG section 24).
    """
    index = {n: i for i, n in enumerate(names)}
    bad = np.zeros(len(x), dtype=bool)
    for layer in range(N_LAYERS):
        e = index.get(f"logE_layer_{layer}")
        if e is None:
            continue
        empty = x[:, e] <= EMPTY_LOG_ENERGY + 1e-9
        if not empty.any():
            continue
        for tag, value in EMPTY_VALUES.items():
            i = index.get(f"{tag}_{layer}")
            if i is not None:
                bad |= empty & (np.abs(x[:, i] - value) > 1e-9)
    return float(np.mean(bad))


def check_energy_has_somewhere_to_be(x, names):
    """A layer holding energy must have at least one lit voxel.

    Sparsity is the fraction of the layer's 144 voxels that are dark, so
    sparsity exactly 1 means every voxel is dark. Together with a non-empty
    energy that is a layer holding energy in no voxel at all. True in 0% of
    real showers; before the rule that forbids it, 58.6% of generated showers
    had at least one such layer.

    The mirror image of ``check_empty_layer_consistency``: emptiness has to
    agree in both directions, and checking only one direction is how this got
    through unnoticed.
    """
    index = {n: i for i, n in enumerate(names)}
    bad = np.zeros(len(x), dtype=bool)
    for layer in range(N_LAYERS):
        e, s = index.get(f"logE_layer_{layer}"), index.get(f"sparsity_{layer}")
        if e is None or s is None:
            continue
        bad |= (x[:, e] > EMPTY_LOG_ENERGY + 1e-9) & (x[:, s] >= 1.0 - 1e-9)
    return float(np.mean(bad))


def check_energy_is_conserved(x, names, max_ratio=3.0):
    """The shower cannot deposit far more energy than the particle brought in.

    Not a consistency rule like the others -- a conservation one, and the
    validator went without it for a day too long. A voxel model generated
    showers depositing a median of 10^10.5 MeV from a beam of at most 10^6, and
    every one of the other checks passed, because the features were perfectly
    consistent with each other and with a physically absurd shower.

    Real ds2 tops out at E_tot / E_inc = 1.6 (the sampling fraction is about
    0.78, with fluctuations), so 3 is a generous ceiling rather than a tight
    one.
    """
    if "logE_tot" not in names or "logE_inc" not in names:
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = 10.0 ** (x[:, names.index("logE_tot")]
                         - x[:, names.index("logE_inc")])
    return float(np.mean(~(ratio <= max_ratio)))


def check_energy_adds_up(x, names, rtol=1e-6):
    """E_tot is the sum of the 45 layer energies -- exactly, by construction.

    Real ds2 satisfies this to 2e-15 relative, so it is a true invariant and
    not a tolerance question. The model generates the total and the layers as
    separate columns and nothing ties them together.
    """
    cols = [names.index(f"logE_layer_{i}") for i in range(N_LAYERS)
            if f"logE_layer_{i}" in names]
    if "logE_tot" not in names or not cols:
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        layers = np.sum(np.clip(10.0 ** x[:, cols] - 1e-8, 0.0, None), axis=1)
        total = np.clip(10.0 ** x[:, names.index("logE_tot")] - 1e-8, 0.0, None)
        rel = np.abs(total - layers) / np.maximum(layers, 1e-12)
    return float(np.mean(~(rel <= rtol)))       # NaN counts as a violation


CHECKS = (
    ("all values finite", check_finite),
    ("widths and radial centres >= 0", check_non_negative),
    ("layer energy >= empty", check_energy_floor),
    ("sparsity within [0, 1]", check_sparsity_range),
    ("sparsity on the 1/144 lattice", check_sparsity_lattice),
    ("empty layer empty in every column", check_empty_layer_consistency),
    ("lit layer has a lit voxel", check_energy_has_somewhere_to_be),
    ("E_tot = sum of layer energies", check_energy_adds_up),
    ("deposits less energy than it got", check_energy_is_conserved),
)


def run_checks(x, names):
    return {label: fn(np.asarray(x, dtype=np.float64), list(names))
            for label, fn in CHECKS}


def report(path, quiet=False):
    """Check one ``--save-samples`` file. Returns {check: (real, generated)}."""
    d = np.load(path, allow_pickle=True)
    names = [str(n) for n in d["names"]]
    real, gen = run_checks(d["real"], names), run_checks(d["generated"], names)
    rows = {k: (real[k], gen[k]) for k, _ in CHECKS}
    if not quiet:
        print(f"\nphysical validity of {os.path.basename(path)} "
              f"({len(d['generated'])} showers each)")
        print("rows violating each check; the real column is the control\n")
        print(f"  {'check':<36} {'Geant4':>9} {'generated':>11}")
        for label, _ in CHECKS:
            r, g = rows[label]
            flag = "" if g <= max(r, 1e-12) else "   <-- fails"
            print(f"  {label:<36} {100 * r:8.2f}% {100 * g:10.2f}%{flag}")
    return rows


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: python -m pinnde_eval.validate_physical "
                         "<file.npz> [more.npz ...]\n"
                         "files come from demo_calo --save-samples")
    for p in paths:
        report(p)
    print()
