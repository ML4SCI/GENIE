"""Conditional flow matching on real CaloChallenge showers.

The toy demo (``demo_conditional``) learns p(3 toy observables | c) on a
synthetic calo-flavoured distribution. This does the same job on real Geant4
data: it learns **p(shower observables | E_inc)** from CaloChallenge ds2 and
scores the result against the measured Geant4-vs-Geant4 null floor.

Train and evaluation data are kept strictly apart: the model trains on
``dataset_2_1.hdf5`` and is scored against ``dataset_2_2.hdf5``, which it never
sees. The floor row it is compared to comes from ds2_1-vs-ds2_2 (DEVLOG
section 11) and is what a *perfect* generator scores -- the point of the
stability study was that comparing to zero is meaningless.

Feature space is the 7 core observables. ``E_tot`` spans three decades
(1 GeV to 1 TeV incident), so it is modelled in log space; everything is then
z-scored with the training statistics and inverted after sampling.

Run from the ``Tina`` folder with both ds2 files present:

    python -m flow_matching.demo_calo
    python -m flow_matching.demo_calo --device cuda    # train and sample on the GPU
    python -m flow_matching.demo_calo --features official   # the challenge's 362
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from pinnde_eval import (binned_residual_map, classifier_discrepancy,
                         discrete_observables, evaluate, evaluate_by_condition,
                         report, separation_power)
from pinnde_eval.calochallenge import official_features_from_file
from pinnde_eval.observables import observables_from_file

from .core import sample
from .train import train_flow_matching

# Measured Geant4-vs-Geant4 floors at N=8000, standardized (DEVLOG sections
# 11 and 15). Any model number is only meaningful next to the row for its own
# feature space -- the floor moves with dimension, and FPD moves violently
# (2.0e-04 at d=7, 4.1e-02 at d=187).
# Measured Geant4-vs-Geant4 floors at N=8000, standardized, over 10 disjoint
# pairs (`python -m pinnde_eval.validate_matched`, 2026-09-16).
#
# These are the **energy-matched** floors: the two Geant4 samples in each pair
# share incident energies shower by shower, which is the position a conditional
# model is in, since it generates at the evaluation set's own energies. The
# older unmatched floors sit 3 to 5 standard errors higher on chi2, swd, w1 and
# sep (DEVLOG section 19), and scoring a model against one is what let the model
# of section 13 look better than Geant4. AUC and Sinkhorn are unaffected.
#
# Spread across the 10 repeats:
#   d=7    auc +/-0.0049  chi2 +/-0.073  swd +/-0.0024  w1 +/-0.0020  sep +/-0.00022
#   d=362  auc +/-0.0033  chi2 +/-0.019  swd +/-0.0006  w1 +/-0.0005  sep +/-0.00005
NULL_FLOORS = {
    7: {"auc": 0.4985, "chi2_mean": 0.8846, "swd": 0.0177,
        "w1_mean": 0.0157, "sep_mean": 0.00239},
    # No matched floor was measured at d=187; this is the unmatched one over 10
    # disjoint pairs, so it stays generous to a model by roughly those same
    # 3 to 5 standard errors on chi2, swd, w1 and sep.
    187: {"auc": 0.4994, "chi2_mean": 1.0183, "swd": 0.0219,
          "w1_mean": 0.0203, "sep_mean": 0.00288},
    362: {"auc": 0.5003, "chi2_mean": 0.9532, "swd": 0.0179,
          "w1_mean": 0.0152, "sep_mean": 0.00263},
}

BIN_EDGES = (0.25, 0.5, 0.75)
BIN_NAMES = ("E 0-25%", "E 25-50%", "E 50-75%", "E 75-100%")
LOG_FEATURES = ("E_tot",)

# ds2 has 16 x 9 = 144 voxels per layer, so a layer's sparsity can only take the
# values k/144. The official features are already in log10 where they need to
# be, so only the sparsity columns need dequantizing.
OFFICIAL_SPARSITY_SPACING = 1.0 / 144


def energy_bin_labels(c):
    return np.array(BIN_NAMES)[np.digitize(np.asarray(c).ravel(), BIN_EDGES)]


def apply_layer_rules(x, blocks, atoms, lattice):
    """The energy decides whether a layer is empty, and it decides both ways.

    A layer the energy calls empty is empty in every column; a layer it calls
    lit must have somewhere to put that energy, so sparsity 1 (not one lit voxel
    in 144) is reserved for the empty case. Before this rule existed, 58.6% of
    generated showers had a layer holding energy in zero voxels; with it, 5.9%.
    It also removed six sevenths of the sparsity-out-of-range values, which had
    been logged as a separate problem.
    """
    for e_col, cols, s_col in blocks:
        empty = x[:, e_col] <= atoms[e_col][0]
        if empty.any():
            for i, v in cols.items():
                x[empty, i] = v
        if s_col is not None and s_col in lattice:
            below_one = lattice[s_col][lattice[s_col] < 1.0]
            if below_one.size:
                dark = ~empty & (x[:, s_col] >= 1.0)
                x[dark, s_col] = below_one.max()      # one lit voxel, the least
    return x


class RankGaussTransform:
    """Map each feature onto a standard normal by its rank, and back.

    The flow starts from Gaussian noise, so the simplest thing to hand it is
    data that is already Gaussian. Pushing each column through its own empirical
    CDF and then the normal quantile function does that: every marginal becomes
    N(0, 1) by construction, and the inverse maps back through the training
    quantiles, so whatever the model does, the generated marginal is the
    training marginal.

    The motivation is the measurement in DEVLOG section 29: a sample carrying
    this model's marginals with Geant4's exact correlations is still detectable
    at 0.916, so per-feature accuracy rather than correlation is what the
    classifier reads.

    Three things follow, each of which needed its own machinery before:

    * **Atoms.** Values tied at the empty value occupy a contiguous block of
      ranks, so an atom becomes an interval in Gaussian space whose width is its
      probability mass -- the band of section 24, sized by how common the value
      is rather than by an accident of where the next real value happens to sit.
    * **Impossible values.** The inverse only ever emits values the training set
      contains, so a negative width or a sparsity above 1 is unreachable.
    * **Lattices.** Sparsity comes back on its comb for the same reason.

    The cost is that generated values are drawn from the training set's own
    values, which for 100,000 rows is a fine grid for a continuous feature but
    is worth stating: the marginal cannot be more accurate than the training
    sample, and it cannot be less.
    """

    def __init__(self, feats, names, geometry="ds2", seed=0, n_quantiles=20000,
                 layer_consistent=True, atom_min_share=0.001, **_ignored):
        raw = np.asarray(feats, dtype=np.float64)
        self.names = list(names)
        self.rng = np.random.default_rng(seed)
        # The quantile grid: the sorted training values, thinned if there are
        # more rows than we need to resolve a marginal.
        step = max(1, len(raw) // n_quantiles)
        self.grid = np.sort(raw, axis=0)[::step]
        self.n = len(self.grid)
        # Reused from the z-score transform so both obey the same physics rule.
        self.discrete = {}
        self.lattice = {i: np.unique(raw[:, i]) for i in range(raw.shape[1])
                        if f"sparsity_" in names[i]}
        self.atoms = find_atoms(raw, min_share=atom_min_share)
        self.blocks = [(e, cols, sp) for e, cols, sp in layer_blocks(self.names)
                       if e in self.atoms] if layer_consistent else []

    def forward(self, feats, dequantize=True):
        x = np.asarray(feats, dtype=np.float64)
        u = np.empty(x.shape)
        for j in range(x.shape[1]):
            lo = np.searchsorted(self.grid[:, j], x[:, j], side="left")
            hi = np.searchsorted(self.grid[:, j], x[:, j], side="right")
            # Ties share a block of ranks; spreading within it is what turns an
            # atom into an interval instead of an unreachable point.
            spread = self.rng.random(len(x)) if dequantize else 0.5
            u[:, j] = (lo + spread * np.maximum(hi - lo, 1)) / (self.n + 1.0)
        return _normal_quantile(np.clip(u, 1e-7, 1 - 1e-7))

    def inverse(self, z, quantize=True):
        u = _normal_cdf(np.asarray(z, dtype=np.float64))
        idx = np.clip((u * self.n).astype(int), 0, self.n - 1)
        x = np.take_along_axis(self.grid, idx, axis=0)
        return apply_layer_rules(x, self.blocks, self.atoms, self.lattice) \
            if quantize else x


def _normal_quantile(u):
    """Phi^-1 without pulling in scipy."""
    from math import sqrt
    try:
        from scipy.special import erfinv
    except ImportError:                     # pragma: no cover
        raise SystemExit("--rank-gauss needs scipy")
    return sqrt(2.0) * erfinv(2.0 * np.asarray(u) - 1.0)


def _normal_cdf(z):
    from math import sqrt
    from scipy.special import erf
    return 0.5 * (1.0 + erf(np.asarray(z) / sqrt(2.0)))


class FeatureTransform:
    """Dequantize discrete columns, log the wide ones, z-score. Invertible.

    Fit on the training set only, so the evaluation set gets no say in the
    normalization -- the same discipline as ``evaluate(standardize=True)``.

    **Dequantization.** ``sparsity`` is a voxel *count*: it only takes values
    ``1 - k/6480``. A continuous flow puts smooth density everywhere and can
    never reproduce that comb, which is trivially detectable -- and worst at
    low incident energy, where a shower lights as few as 6 voxels and sparsity
    spans only ~350 distinct values (DEVLOG section 14). The standard fix for
    discrete data under a continuous model is to spread each atom over its
    lattice cell during training (add U(0,1) to the integer count) and floor
    back to the lattice when sampling. The model then learns a smooth density
    whose quantized marginal matches the data exactly.
    """

    def __init__(self, feats, names, geometry="ds2", seed=0, spacings=None,
                 sqrt_features=(), atoms=False, atom_min_share=0.001,
                 layer_consistent=True, atom_band=None):
        self.names = list(names)
        self.log_idx = [i for i, n in enumerate(names) if n in LOG_FEATURES]
        # Columns modelled as sqrt(x) and returned as y**2: widths and radial
        # centres are non-negative with a spike at exactly 0 (23% of ds2 width
        # values), and an unbounded flow put 30% of deep-layer widths below
        # zero, which is impossible. Squaring cannot produce a negative, keeps
        # exact 0 reachable, and folds mass near y=0 onto the spike.
        self.sqrt_idx = [i for i, n in enumerate(names) if n in set(sqrt_features)]
        spacings = discrete_observables(geometry) if spacings is None else spacings
        self.discrete = {names.index(n): s for n, s in spacings.items()
                         if n in names}
        # The lattice values exactly as the data stores them, for snapping.
        raw = np.asarray(feats, dtype=np.float64)
        self.lattice = {i: np.unique(raw[:, i]) for i in self.discrete}
        self.atoms = find_atoms(raw, skip=self.discrete,
                                min_share=atom_min_share) if atoms else {}
        # How much of the empty gap the point mass is spread over. The whole gap
        # (the default) is simplest, but the gap is ~6 log units wide and the
        # spread is noise the network cannot predict: for a voxel lit 10% of the
        # time it is 51% of that column's variance, against a real signal of
        # 11%. A narrow band just under the first real value keeps the two
        # regions disjoint and injects far less noise.
        self.atom_band = atom_band
        # An empty layer is emptied as a whole, not column by column.
        self.blocks = [(e, cols, s) for e, cols, s in layer_blocks(self.names)
                       if e in self.atoms] if atoms and layer_consistent else []
        self.rng = np.random.default_rng(seed)
        x = self._to_log(self._dequantize(np.asarray(feats, dtype=np.float64)))
        self.mean = x.mean(axis=0)
        self.std = np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)

    def _dequantize(self, x):
        """Spread each lattice atom uniformly over its own cell."""
        x = x.copy()
        for i, spacing in self.discrete.items():
            x[:, i] = x[:, i] + self.rng.uniform(0.0, spacing, size=len(x))
        for i, (value, hi) in self.atoms.items():
            at = x[:, i] <= value
            lo = value if not self.atom_band else max(value, hi - self.atom_band)
            x[at, i] = self.rng.uniform(lo, hi, size=int(at.sum()))
        return x

    def _quantize(self, x):
        """Snap back onto the lattice: the exact inverse of _dequantize.

        A grid point is replaced by the value the data itself stores there,
        not by ``k * spacing``. The two differ in the last bit (1 - 71/144 is
        not 73 * (1/144) in floating point), and one bit is enough for an
        exact-match test such as KS to reject a perfect generator: measured on
        ds2 sparsity, p = 5e-25 where real against real gives 0.12.

        Points off the observed lattice (sparsity above 1, say) are left where
        they are, so impossible values stay visible instead of being pulled
        back into range.
        """
        x = x.copy()
        for i, spacing in self.discrete.items():
            grid = np.floor(x[:, i] / spacing) * spacing
            values = self.lattice[i]
            idx = np.clip(np.searchsorted(values, grid), 0, len(values) - 1)
            below = values[np.clip(idx - 1, 0, len(values) - 1)]
            nearest = np.where(np.abs(grid - below) < np.abs(grid - values[idx]),
                               below, values[idx])
            x[:, i] = np.where(np.abs(grid - nearest) < 0.5 * spacing, nearest, grid)
        for i, (value, hi) in self.atoms.items():
            # Strictly inside the band. A real value sitting exactly at the top
            # edge comes back from the z-score round trip a bit-width either
            # side of it, and half of those would otherwise be snapped to the
            # atom -- destroying the smallest genuine value in the column.
            x[x[:, i] < hi - max(abs(hi), 1.0) * 1e-9, i] = value
        return apply_layer_rules(x, self.blocks, self.atoms, self.lattice)

    def _to_log(self, x):
        x = x.copy()
        for i in self.log_idx:
            x[:, i] = np.log(np.clip(x[:, i], 1e-8, None))
        for i in self.sqrt_idx:
            x[:, i] = np.sqrt(np.clip(x[:, i], 0.0, None))
        return x

    def forward(self, feats, dequantize=True):
        x = np.asarray(feats, dtype=np.float64)
        if dequantize:
            x = self._dequantize(x)
        return (self._to_log(x) - self.mean) / self.std

    def inverse(self, z, quantize=True):
        x = np.asarray(z, dtype=np.float64) * self.std + self.mean
        for i in self.log_idx:
            x[:, i] = np.exp(np.clip(x[:, i], -50, 50))
        for i in self.sqrt_idx:
            x[:, i] = x[:, i] ** 2               # never negative, by construction
        return self._quantize(x) if quantize else x


RELATIVE_MODES = ("total", "all")


def _relative_energy_columns(names, which="all"):
    """(incident column, energy columns, modelled columns) for official features.

    ``which="all"`` makes every energy relative; ``which="total"`` only log10
    E_tot, leaving the 45 layer energies absolute. The split exists because the
    two behave differently: relative E_tot fixed the energy response, while
    relative layer energies made that family much worse (DEVLOG section 21).
    """
    if which not in RELATIVE_MODES:
        raise ValueError(f"which must be one of {RELATIVE_MODES}, got {which!r}")
    inc = names.index("logE_inc")
    energy = [i for i, n in enumerate(names)
              if n == "logE_tot" or (which == "all" and n.startswith("logE_layer_"))]
    modelled = [i for i in range(len(names)) if i != inc]
    return inc, energy, modelled


def to_relative_energy(x, names, which="all"):
    """Energies as log10(E / E_inc), with the incident energy itself removed.

    The official features store log10 of every energy, so a column spans three
    decades set by the incident energy alone, and the physics -- a sampling
    fraction with a 2% spread at high energy -- is a sliver of that range. A
    model trained on the absolute column smeared that spread to 4.4% (DEVLOG
    section 21). Dividing by E_inc leaves the part that is actually shower
    physics. log10 E_inc goes too: it is the condition, so the model should
    not be asked to regenerate it.
    """
    inc, energy, modelled = _relative_energy_columns(names, which)
    y = np.array(x, dtype=np.float64, copy=True)
    y[:, energy] -= y[:, [inc]]
    return y[:, modelled]


def from_relative_energy(y, names, log10_e_inc, which="all"):
    """Exact inverse of ``to_relative_energy``, given each shower's true E_inc."""
    inc, energy, modelled = _relative_energy_columns(names, which)
    log10_e_inc = np.asarray(log10_e_inc, dtype=np.float64).ravel()
    x = np.empty((len(y), len(names)))
    x[:, modelled] = y
    x[:, inc] = log10_e_inc
    x[:, energy] += log10_e_inc[:, None]
    return x


def drop_non_finite(real, gen, cond, e_inc):
    """Remove generated showers containing NaN or inf, with their real partners.

    A sampler trajectory can diverge, and a single NaN shower stalls or crashes
    the scoring (the relative-energy run of 2026-09-17 lost its whole report to
    2 of 8000). Each generated shower was produced at its real partner's energy,
    so both are dropped together to keep the per-energy comparison aligned.
    Returns ``(real, gen, cond, e_inc, n_dropped)``.
    """
    keep = np.isfinite(np.asarray(gen, dtype=np.float64)).all(axis=1)
    e_inc = np.asarray(e_inc).ravel()
    return real[keep], gen[keep], cond[keep], e_inc[keep], int((~keep).sum())


def find_atoms(x, skip=(), min_share=0.01):
    """Columns whose smallest value is a point mass, and the gap above it.

    An empty layer is not a small number, it is an exact one: log10 E_layer is
    exactly -8, every width and radial centre exactly 0. Over ds2 that is 17.8%
    of all (layer, shower) pairs and 51% of the last layer. A continuous flow
    puts zero probability on any exact value, so it can never reproduce one:
    measured on a 100k-step run, real showers are empty in 17.8% of pairs and
    generated showers in 0.0%.

    Returns ``{column: (atom, next_value)}``. The empty interval between the
    two is real estate the data never uses, so training can spread the atom
    across it and sampling can snap all of it back -- the same trick as the
    sparsity lattice, applied to a point mass instead of a comb.
    """
    atoms = {}
    x = np.asarray(x, dtype=np.float64)
    for i in range(x.shape[1]):
        if i in skip:
            continue
        col = x[:, i]
        value = col.min()
        count = int(np.sum(col == value))
        # A single smallest value is just the smallest value; a point mass has
        # to be repeated. Without this, any threshold below 1/n calls every
        # column an atom.
        if count < 2 or count < min_share * len(col):
            continue
        above = col[col > value]
        if above.size:
            atoms[i] = (float(value), float(above.min()))
    return atoms


LAYER_EMPTY_VALUES = {"EC_eta": 0.0, "EC_phi": 0.0, "width_eta": 0.0,
                      "width_phi": 0.0, "EC_R": 0.0, "width_R": 0.0,
                      "sparsity": 1.0}


def layer_blocks(names, n_layers=45):
    """For each layer: its energy column, and what its other columns hold when
    the layer is empty.

    Emptiness belongs to the layer, not to the column. Measured on ds2: when a
    layer has no energy, its centres and widths are exactly 0 and its sparsity
    exactly 1, in 100% of real showers. Snapping columns one at a time breaks
    that -- a generated layer came out with no energy but a non-zero centre,
    which no real shower contains, and a classifier reads it immediately (AUC
    0.892 -> 0.968 with per-column snapping alone).

    Returns ``[(energy_column, {column: empty_value})]``.
    """
    index = {n: i for i, n in enumerate(names)}
    blocks = []
    for layer in range(n_layers):
        e = index.get(f"logE_layer_{layer}")
        if e is None:
            continue
        cols = {index[f"{tag}_{layer}"]: v
                for tag, v in LAYER_EMPTY_VALUES.items()
                if f"{tag}_{layer}" in index}
        if cols:
            blocks.append((e, cols, index.get(f"sparsity_{layer}")))
    return blocks


def check_parameterisation(features, relative_energy, energy_sqrt,
                           positive_sqrt=False, derive_total=False):
    """Validate the parameterisation flags. Returns the relative-energy mode.

    Checked before any data is loaded, so a bad combination fails in a second
    rather than after a ten-minute extraction.
    """
    if relative_energy is True:
        relative_energy = "all"
    if relative_energy and relative_energy not in RELATIVE_MODES:
        raise SystemExit(f"--relative-energy must be one of {RELATIVE_MODES}")
    if (relative_energy or energy_sqrt or positive_sqrt
            or derive_total) and features != "official":
        raise SystemExit("--relative-energy, --energy-sqrt, --positive-sqrt and "
                         "--derive-total need --features official")
    if derive_total and relative_energy:
        raise SystemExit("--derive-total computes E_tot from the layers, so it "
                         "cannot also be modelled relative to E_inc")
    if energy_sqrt and relative_energy == "all":
        raise SystemExit("--energy-sqrt already divides the layer energies by "
                         "E_inc; use --relative-energy total with it, not all")
    return relative_energy


def total_from_layers(x, names):
    """Replace log10 E_tot with the total the 45 layer energies actually add to.

    E_tot is not an independent quantity: it *is* the sum of the layers, exact
    to 2e-15 in real ds2. The model had no way to know that -- it generated the
    total as its own column -- and 100% of generated showers came out
    inconsistent, a defect no distribution metric noticed because each column's
    marginal was fine on its own.
    """
    cols = _layer_energy_columns(names)
    y = np.array(x, dtype=np.float64, copy=True)
    energy = np.clip(10.0 ** y[:, cols] - 1e-8, 0.0, None)
    y[:, names.index("logE_tot")] = np.log10(energy.sum(axis=1) + 1e-8)
    return y


def _layer_energy_columns(names):
    return [i for i, n in enumerate(names) if n.startswith("logE_layer_")]


def to_sqrt_fraction(x, names, e_inc):
    """Layer energies as sqrt(E_layer / E_inc) instead of log10(E_layer + 1e-8).

    The official column is an unbounded log, so a flow puts 24% of deep-layer
    values below -8 -- under "no energy at all", which is impossible. A layer's
    energy is non-negative with a spike at exactly 0, the same shape as the
    widths that sqrt fixed: squaring on the way out cannot go negative, an empty
    layer sits at exactly 0, and dividing by E_inc removes the three decades of
    scale that the condition already carries.
    """
    cols = _layer_energy_columns(names)
    y = np.array(x, dtype=np.float64, copy=True)
    energy = np.clip(10.0 ** y[:, cols] - 1e-8, 0.0, None)
    y[:, cols] = np.sqrt(energy / np.asarray(e_inc, dtype=np.float64).reshape(-1, 1))
    return y


def from_sqrt_fraction(y, names, e_inc):
    """Exact inverse of ``to_sqrt_fraction``, given each shower's true E_inc."""
    cols = _layer_energy_columns(names)
    x = np.array(y, dtype=np.float64, copy=True)
    energy = x[:, cols] ** 2 * np.asarray(e_inc, dtype=np.float64).reshape(-1, 1)
    x[:, cols] = np.log10(energy + 1e-8)
    return x


def out_of_range_fraction(real, gen, rel_tol=1e-6):
    """Fraction of generated values outside the real range, per column.

    A value counts only when it is outside by more than ``rel_tol`` of the
    column's range (at least ``rel_tol`` in absolute terms). Without that
    margin, floating-point noise at a hard boundary -- an exact 0 coming back
    as -7e-18 -- is reported as an impossible value: the null check found 59% of
    *real* showers flagged that way in one column. A genuine overshoot, such as
    a negative width of -0.01, is millions of times larger and still counts.
    """
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    lo, hi = real.min(axis=0), real.max(axis=0)
    margin = rel_tol * np.maximum(hi - lo, 1.0)
    return ((gen < lo - margin) | (gen > hi + margin)).mean(axis=0)


def normalized_log_energy(e_inc, lo=None, hi=None):
    """Map incident energy to c in [0, 1] via log, the ds2 sampling variable."""
    log_e = np.log(np.asarray(e_inc, dtype=np.float64).ravel())
    lo = np.min(log_e) if lo is None else lo
    hi = np.max(log_e) if hi is None else hi
    return ((log_e - lo) / (hi - lo)).reshape(-1, 1), lo, hi


def compare_to_floor(res, title, floor):
    """Print model numbers beside the measured perfect-generator floor.

    For d=7 and d=362 the floor comes from Geant4 pairs whose incident energies
    are matched shower by shower, which is the position this model is in: it
    generates at the evaluation set's own energies and so never pays the
    energy-draw variance two independent Geant4 samples pay. An unmatched floor
    sits 3 to 5 standard errors higher on chi2, swd, w1 and sep (DEVLOG section
    19) -- enough to make a model look better than Geant4 without being it.
    Landing inside the floor's spread means "indistinguishable at this sample
    size", never "better".
    """
    print(f"\n{title}")
    print(f"{'metric':>10} | {'model':>10} | {'null floor':>10} | {'vs floor':>9}")
    got = {"auc": res["auc"][0], "chi2_mean": res["chi2_mean"],
           "swd": res["swd"], "w1_mean": res["w1_mean"],
           "sep_mean": res["sep_mean"]}
    for key, ref in floor.items():
        val = got[key]
        note = f"{val - 0.5:+.4f}" if key == "auc" else f"{val / ref:.1f}x"
        print(f"{key:>10} | {val:10.4f} | {ref:10.4f} | {note:>9}")


def main(n_train=100000, n_eval=8000, n_steps=30000, hidden=384, depth=5,
         ode_steps=200, seed=0, plot_path=None, data_dir=None,
         dequantize=True, features="core", device="cpu", null=False,
         save_samples=None, relative_energy=False, lr=1e-3, clip_grad=None,
         positive_sqrt=False, energy_sqrt=False, atom_snap=False,
         atom_min_share=0.001, derive_total=False, rank_gauss=False):
    """Train and score p(observables | E_inc) on real ds2. ~20 min on CPU.

    The defaults are the configuration that actually fixes the low-energy
    failure (DEVLOG section 14). The original ``hidden=128, depth=3,
    n_steps=12000`` reaches AUC 0.79 in the lowest energy quartile; it is kept
    documented because the *diagnosis* is the useful part, not the number.

    ``features`` chooses the space the model works in:

    * ``"core"`` -- the 7 whole-shower observables (the default).
    * ``"per-layer"`` -- 187 columns, the 7 core plus 4 per layer. A much
      harder target: at low incident energy 98 of the 187 columns put over half
      their mass on a single value (an empty layer has sparsity exactly 1 and
      radial centre exactly 0), so the effective dimension collapses where the
      model already struggled.
    * ``"official"`` -- the 362 features the CaloChallenge's own classifier
      reads, computed with their code (``pinnde_eval.calochallenge``). This is
      the only space in which our numbers can be compared with published
      submissions, and it makes the empty-layer problem worse still: an empty
      layer gives log10 E_layer exactly -8, sparsity exactly 1, and centres and
      widths exactly 0.

    ``device="cuda"`` trains and samples on the GPU; the metrics run on CPU
    either way. The GPU draws its own random numbers, so a GPU run is a
    different random realisation from a CPU run with the same seed.

    ``save_samples`` is an optional ``.npz`` path. The evaluation showers, the
    generated ones, their energies and the column names are written there, so a
    diagnosis (per column family, per energy slice, per layer) can be run on a
    finished model without training it again.

    ``null=True`` skips training and feeds **real Geant4 showers, matched to the
    evaluation set's incident energies**, through everything a model's samples
    go through: the same transform, the same metrics, the same tables. A perfect
    generator is exactly that, so every number must land on the floor. It is the
    check that the harness is not inventing the failure it reports.
    """
    relative_energy = check_parameterisation(features, relative_energy,
                                             energy_sqrt, positive_sqrt,
                                             derive_total)
    data_dir = data_dir or os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(data_dir, "dataset_2_1.hdf5")
    eval_path = os.path.join(data_dir, "dataset_2_2.hdf5")
    for p in (train_path, eval_path):
        if not os.path.exists(p):
            raise SystemExit(
                f"missing {p}\nDownload ds2 from "
                f"https://zenodo.org/records/6366271 into the Tina/ folder.")

    print(f"extracting {features} features: {n_train} train / {n_eval} eval showers")
    if features == "official":
        cache_dir = os.path.join(data_dir, "calochallenge_cache")
        x_train, names, e_train = official_features_from_file(
            train_path, n_train,
            cache=os.path.join(cache_dir, f"ds2_1_official_{n_train}.npz"))
        x_eval, _, e_eval = official_features_from_file(
            eval_path, n_eval,
            cache=os.path.join(cache_dir, f"ds2_2_official_{n_eval}.npz"))
    else:
        per_layer = features == "per-layer"
        x_train, names, e_train = observables_from_file(train_path, n_train,
                                                        per_layer=per_layer)
        x_eval, _, e_eval = observables_from_file(eval_path, n_eval,
                                                  per_layer=per_layer)
    print(f"  {len(names)} observables"
          + (f": {', '.join(names)}" if len(names) <= 12 else
             f" (7 core + 4 x 45 per-layer)"))
    floor = NULL_FLOORS.get(len(names))
    if floor is None:
        raise SystemExit(f"no measured null floor for d={len(names)}; run "
                         f"pinnde_eval.validate_calo at this dimension first")

    # What the flow actually models. Both energy transforms need E_inc per
    # shower, so log10 E_inc stops being modelled and comes from the condition.
    drop_inc = bool(relative_energy) or energy_sqrt
    # E_tot is the sum of the layers, so with --derive-total it is not modelled
    # either: it is computed from the generated layer energies on the way out.
    dropped = ([n for n in names if n == "logE_inc"] if drop_inc else []) \
        + (["logE_tot"] if derive_total else [])
    model_names = [n for n in names if n not in set(dropped)]
    inc_col = names.index("logE_inc") if drop_inc else None
    tot_col = names.index("logE_tot") if derive_total else None

    def encode(x, e):
        y = np.asarray(x, dtype=np.float64)
        if energy_sqrt:
            y = to_sqrt_fraction(y, names, e)
        if relative_energy:
            y = to_relative_energy(y, names, which=relative_energy)
        elif drop_inc:
            y = np.delete(y, inc_col, axis=1)
        if derive_total:                     # after any inc column was removed
            kept = [n for n in names if not (drop_inc and n == "logE_inc")]
            y = np.delete(y, kept.index("logE_tot"), axis=1)
        return y

    def decode(y):
        e = np.asarray(e_eval, dtype=np.float64).ravel()
        x = y
        if derive_total:                     # put the column back, any value
            kept = [n for n in names if not (drop_inc and n == "logE_inc")]
            x = np.insert(x, kept.index("logE_tot"), 0.0, axis=1)
        if relative_energy:
            x = from_relative_energy(x, names, np.log10(e), which=relative_energy)
        elif drop_inc:
            x = np.insert(x, inc_col, np.log10(e), axis=1)
        if energy_sqrt:
            x = from_sqrt_fraction(x, names, e)
        return total_from_layers(x, names) if derive_total else x

    if relative_energy:
        which = "E_tot only" if relative_energy == "total" else "E_tot and every layer"
        print(f"  energies modelled as log10(E / E_inc) for {which}; log10 E_inc "
              f"comes from the condition")

    if energy_sqrt:
        print("  layer energies modelled as sqrt(E_layer / E_inc) and returned "
              "squared, so a generated layer can be empty but never negative")

    spacings = None
    if features == "official":
        spacings = {n: OFFICIAL_SPARSITY_SPACING for n in model_names
                    if n.startswith("sparsity_")}
    # Non-negative columns with a spike at 0: the widths and the radial centre.
    sqrt_features = ()
    if positive_sqrt:
        sqrt_features = tuple(n for n in model_names
                              if n.startswith(("width_", "EC_R_")))
        print(f"  {len(sqrt_features)} non-negative columns modelled as sqrt(x) "
              f"and returned squared")
    train_enc = encode(x_train, e_train)
    cls = RankGaussTransform if rank_gauss else FeatureTransform
    if rank_gauss:
        print("  every feature mapped to a standard normal by its rank; the "
              "generated marginal is the training marginal by construction")
    tf = cls(train_enc, model_names, seed=seed, spacings=spacings,
             sqrt_features=sqrt_features, atoms=atom_snap,
             atom_min_share=atom_min_share)
    if atom_snap:
        share = np.mean([np.mean(train_enc[:, i] == v)
                         for i, (v, _) in tf.atoms.items()]) if tf.atoms else 0.0
        print(f"  {len(tf.atoms)} columns have an empty-layer atom "
              f"(mean {100 * share:.1f}% of rows); it is spread over the gap "
              f"above it in training and snapped back when sampling")
        print(f"  {len(tf.blocks)} layers are emptied as a whole: if the energy "
              f"says empty, that layer's centres and widths are set to 0 and "
              f"its sparsity to 1, as in every real shower")
    if not dequantize:                      # ablation: treat counts as continuous
        tf.discrete = {}
    print(f"  dequantized columns: "
          f"{[model_names[i] for i in tf.discrete] or 'none (ablation)'}")
    z_train = tf.forward(train_enc)
    c_train, lo, hi = normalized_log_energy(e_train)
    c_eval, _, _ = normalized_log_energy(e_eval, lo, hi)

    if null:
        # What a perfect generator produces: real Geant4 showers from the
        # training file, matched to the evaluation set's energies shower by
        # shower, pushed through the same transform the model's samples take.
        # Everything below must then land on the floor; if it does not, the
        # harness is inventing the failure rather than measuring it.
        from pinnde_eval.floors import match_by_energy
        picks, worst = match_by_energy(np.log(e_eval), np.log(e_train))
        model = None
        # Bit-exact, deliberately not round-tripped through the transform: the
        # round trip moves exact zeros by ~1e-17, which is invisible to the
        # classifier and the histograms but makes KS and the range check call
        # real showers impossible. The transform has its own lossless test.
        gen = x_train[picks]
        print(f"\nnull check: no training. Real Geant4 showers stand in for "
              f"the model,\nmatched to the evaluation energies to within "
              f"{worst:.2e} in log E_inc.")
    else:
        print(f"\ntraining conditional flow matching: d={len(model_names)}, "
              f"cond=1, n={n_train}, steps={n_steps}, hidden={hidden}, "
              f"depth={depth}, lr={lr}, clip_grad={clip_grad}")
        model, history = train_flow_matching(
            torch.tensor(z_train, dtype=torch.float32), dim=len(model_names),
            cond=torch.tensor(c_train, dtype=torch.float32),
            n_steps=n_steps, hidden=hidden, depth=depth, device=device,
            lr=lr, clip_grad=clip_grad,
            seed=seed, monitor_every=max(1, n_steps // 8),
            monitor_real=torch.tensor(tf.forward(encode(x_eval, e_eval)),
                                      dtype=torch.float32),
            monitor_cond=torch.tensor(c_eval, dtype=torch.float32),
        )
        print("\nstep |      loss |        mmd |        swd")
        for step, loss, m, s in history:
            print(f"{step:5d} | {loss:9.4f} | {m:10.4e} | {s:10.4f}")

        # Generate at the evaluation set's own conditions, then invert to
        # physical units so every number below is in the observables' real scale.
        z_gen = sample(model, n_eval, len(model_names),
                       cond=torch.tensor(c_eval, dtype=torch.float32),
                       steps=ode_steps, device=device, seed=seed)
        gen = decode(tf.inverse(z_gen.detach().cpu().numpy()))

    if save_samples:
        os.makedirs(os.path.dirname(os.path.abspath(save_samples)), exist_ok=True)
        np.savez(save_samples, real=x_eval, generated=gen,
                 e_inc=np.asarray(e_eval).ravel(), names=np.array(names))
        print(f"\nsaved {len(gen)} generated and {len(x_eval)} real showers to "
              f"{save_samples}")

    n_before = len(gen)
    x_eval, gen, c_eval, e_eval, n_dropped = drop_non_finite(x_eval, gen, c_eval, e_eval)
    if n_dropped:
        print(f"\nWARNING: {n_dropped} of {n_before} generated showers contain NaN "
              f"or inf -- the sampler diverged for them. That is a model failure, "
              f"not noise.\nScoring continues on the {len(gen)} intact showers and "
              f"their real partners.")
    else:
        print(f"\nall {n_before} generated showers are finite")

    print()
    res = evaluate(x_eval, gen, tier="full", seed=seed, standardize=True)
    report(res, title="conditional FM vs Geant4 (ds2_2, pooled over energy)")
    compare_to_floor(res, "against the measured perfect-generator floor", floor)

    # At d=187 the per-observable table is unreadable in full; show the worst.
    order = np.argsort(res["sep_per_feature"])[::-1]
    shown = order if len(names) <= 12 else order[:10]
    print(f"\nper observable"
          + ("" if len(names) <= 12 else f" (worst 10 of {len(names)})") + ":")
    print(f"{'observable':>18} | {'chi2':>7} | {'sep':>8} | {'floor sep':>9}")
    for i in shown:
        print(f"{names[i]:>18} | {res['chi2_per_feature'][i]:7.2f} | "
              f"{res['sep_per_feature'][i]:8.5f} | {floor['sep_mean']:9.4f}")

    # Per energy bin -- the pooled number hides a bad slice.
    labels = energy_bin_labels(c_eval)
    by_bin = evaluate_by_condition(x_eval, gen, labels, tier="full", seed=seed,
                                   standardize=True)
    print(f"\n{'energy bin':>12} | {'n':>5} | {'swd':>7} | {'auc':>15} | "
          f"{'sep':>8}")
    for nm in BIN_NAMES:
        if nm not in by_bin:
            continue
        r = by_bin[nm]
        print(f"{nm:>12} | {r['n_real']:5d} | {r['swd']:7.4f} | "
              f"{r['auc'][0]:5.3f} +/- {r['auc'][1]:5.3f} | {r['sep_mean']:8.5f}")

    # Does the generator respect the physical range of each observable?
    # sparsity and f_samp are bounded; an unconstrained flow can leave the
    # support, which is a concrete defect rather than a distributional one.
    out_of_range = out_of_range_fraction(x_eval, gen)
    worst = np.argsort(out_of_range)[::-1]
    rows = worst[:10] if len(names) > 12 else range(len(names))
    print(f"\n{'observable':>12} | {'Geant4 range':>22} | {'gen out of range':>16}"
          + ("" if len(names) <= 12 else f"   (worst 10 of {len(names)})"))
    for i in rows:
        lo, hi = x_eval[:, i].min(), x_eval[:, i].max()
        print(f"{names[i]:>12} | {lo:10.4g} .. {hi:9.4g} | {out_of_range[i]:15.2%}")

    if features == "official":
        # The official feature set contains log10 E_inc, a deterministic
        # function of the condition the model was handed. A continuous flow
        # cannot emit a point mass, so this column is a defect we build in;
        # this measures how big it is.
        col = names.index("logE_inc")
        asked = np.log10(np.asarray(e_eval, dtype=np.float64).ravel())
        error = np.abs(gen[:, col] - asked)
        print(f"\ngenerated log10 E_inc against the energy it was conditioned "
              f"on:\n  median error {np.median(error):.4f}, 90th percentile "
              f"{np.quantile(error, 0.9):.4f}, over a range of "
              f"{asked.max() - asked.min():.2f} decades")

    # Where does it fail? Run the local maps on the *worst energy bin*, not
    # pooled. The failure here is conditional: pooled AUC sits at the floor
    # while one quartile is far from it, so pooled maps see nothing.
    worst_bin = max((nm for nm in BIN_NAMES if nm in by_bin),
                    key=lambda nm: by_bin[nm]["auc"][0])
    mask = labels == worst_bin
    print(f"\nlocal maps on the worst slice ({worst_bin}, "
          f"auc {by_bin[worst_bin]['auc'][0]:.3f}, n={int(mask.sum())}):")

    for tag, r, g in (("pooled", x_eval, gen),
                      (worst_bin, x_eval[mask], gen[mask])):
        r_std, g_std = _standardized_pair(r, g)
        _, p_gen, oof_auc = classifier_discrepancy(r_std, g_std, seed=seed)
        worst_feats = tuple(np.argsort(separation_power(r, g))[-2:][::-1])
        resid, _ = binned_residual_map(r, g, features=worst_feats, bins=15)
        print(f"  {tag:>12}: oof AUC {oof_auc:.4f} | "
              f"confidently-fake gen {(p_gen < 0.1).mean():5.1%} | "
              f"max |r| {np.abs(resid).max():4.1f} on "
              f"({names[worst_feats[0]]}, {names[worst_feats[1]]})")

    if plot_path:
        # With more than a handful of columns, plot the ones the model gets
        # most wrong rather than a fixed list.
        show = list(order[:6]) if len(names) > 12 else [0, 2, 3, 4, 5, 6]
        _plot(x_eval, c_eval, gen, names, plot_path, show)
    return model, res


def _standardized_pair(real, gen):
    mean, std = real.mean(axis=0), real.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return (real - mean) / std, (gen - mean) / std


def _plot(x_eval, c_eval, gen, names, path, show=(0, 2, 3, 4, 5, 6)):
    """Conditional means vs energy (top) and marginals (bottom).

    ``show`` picks the columns to draw, one per figure column.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.asarray(c_eval).ravel()
    edges = np.linspace(0, 1, 11)
    mids = 0.5 * (edges[:-1] + edges[1:])
    show = list(show)

    fig, axes = plt.subplots(2, len(show), figsize=(3.1 * len(show), 6.6))
    for col, j in enumerate(show):
        ax = axes[0, col]
        for arr, label, color in ((x_eval, "Geant4", "tab:blue"),
                                  (gen, "flow matching", "tab:orange")):
            m = np.array([arr[(c >= a) & (c < b), j].mean()
                          for a, b in zip(edges, edges[1:])])
            s = np.array([arr[(c >= a) & (c < b), j].std()
                          for a, b in zip(edges, edges[1:])])
            ax.plot(mids, m, marker="o", ms=3, color=color, label=label)
            ax.fill_between(mids, m - s, m + s, color=color, alpha=0.2)
        ax.set_title(names[j], fontsize=10)
        ax.set_xlabel("normalized log $E_{inc}$", fontsize=8)
        if col == 0:
            ax.set_ylabel("conditional mean ± std")
            ax.legend(fontsize=8)
        if names[j] == "E_tot":
            ax.set_yscale("log")

        ax = axes[1, col]
        lo = min(x_eval[:, j].min(), gen[:, j].min())
        hi = max(x_eval[:, j].max(), gen[:, j].max())
        if names[j] == "E_tot":
            # E_inc is log-uniform, so E_tot spans three decades. Linear bins
            # on a log axis put almost everything in the first bin and render
            # as one meaningless block; bin in log space to match the axis.
            b = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 60)
        else:
            b = np.linspace(lo, hi, 60)
        ax.hist(x_eval[:, j], bins=b, density=True, histtype="step",
                color="tab:blue", label="Geant4")
        ax.hist(gen[:, j], bins=b, density=True, histtype="step",
                color="tab:orange", label="flow matching")
        ax.set_xlabel(names[j], fontsize=8)
        if names[j] == "E_tot":
            ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel("density")
    fig.suptitle("Conditional flow matching on real CaloChallenge ds2 showers")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=("core", "per-layer", "official"),
                        default="core",
                        help="7 core observables, 187 per-layer columns, or the "
                             "CaloChallenge's own 362 features (default: core)")
    parser.add_argument("--per-layer", action="store_true",
                        help="same as --features per-layer")
    parser.add_argument("--device", default="cpu",
                        help="where to train and sample, e.g. cuda (default: cpu)")
    parser.add_argument("--hidden", type=int, default=384,
                        help="width of the velocity field (default: 384)")
    parser.add_argument("--depth", type=int, default=5,
                        help="number of hidden layers (default: 5)")
    parser.add_argument("--steps", type=int, default=30000,
                        help="training steps (default: 30000)")
    parser.add_argument("--null", action="store_true",
                        help="skip training and feed real Geant4 showers at "
                             "matched energies through the same scoring: every "
                             "number must land on the floor")
    parser.add_argument("--ode-steps", type=int, default=200,
                        help="sampler steps from noise to data (default: 200)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for training, sampling and scoring (default: 0)")
    parser.add_argument("--save-samples", default=None, metavar="PATH",
                        help="write real and generated showers to this .npz "
                             "for diagnosis without retraining")
    parser.add_argument("--relative-energy", nargs="?", const="all", default=None,
                        choices=RELATIVE_MODES,
                        help="with --features official: model energies as "
                             "log10(E / E_inc) and take E_inc from the condition; "
                             "'total' for E_tot only, 'all' (the default when "
                             "given without a value) for E_tot and every layer")
    parser.add_argument("--positive-sqrt", action="store_true",
                        help="with --features official: model the widths and "
                             "radial centres as sqrt(x), so generated values "
                             "can never be negative")
    parser.add_argument("--energy-sqrt", action="store_true",
                        help="with --features official: model the 45 layer "
                             "energies as sqrt(E_layer / E_inc) instead of "
                             "log10 E_layer, so a generated layer can be empty "
                             "but never holds negative energy")
    parser.add_argument("--atom-snap", action="store_true",
                        help="treat an empty layer as the exact value it is: "
                             "spread the point mass over the empty gap above it "
                             "during training and snap it back when sampling")
    parser.add_argument("--derive-total", action="store_true",
                        help="with --features official: do not model log10 "
                             "E_tot, compute it from the generated layer "
                             "energies. Real showers satisfy E_tot = sum of "
                             "layers exactly; 100%% of generated ones did not")
    parser.add_argument("--rank-gauss", action="store_true",
                        help="map every feature to a standard normal by its "
                             "rank instead of z-scoring it. The generated "
                             "marginal is then the training marginal whatever "
                             "the model does, and impossible values are "
                             "unreachable")
    parser.add_argument("--atom-min-share", type=float, default=0.001,
                        help="how much of a column has to sit on its smallest "
                             "value before --atom-snap treats it as a point "
                             "mass (default: 0.001). At 0.01 the front-layer "
                             "widths fall below the line and 8%% of them come "
                             "out negative; 0.001 covers them and scores better "
                             "on every metric")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 1e-3)")
    parser.add_argument("--clip-grad", type=float, default=None,
                        help="cap on the gradient norm per step, e.g. 1.0 "
                             "(default: no cap)")
    args = parser.parse_args()
    features = "per-layer" if args.per_layer else args.features
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    name = {"core": "fm_calo.png", "per-layer": "fm_calo_per_layer.png",
            "official": "fm_calo_official.png"}[features]
    if args.null:
        name = "null_" + name            # never overwrite a model's figure
    main(features=features, device=args.device, hidden=args.hidden,
         depth=args.depth, n_steps=args.steps, null=args.null,
         ode_steps=args.ode_steps, seed=args.seed,
         save_samples=args.save_samples, relative_energy=args.relative_energy,
         lr=args.lr, clip_grad=args.clip_grad, positive_sqrt=args.positive_sqrt,
         energy_sqrt=args.energy_sqrt, atom_snap=args.atom_snap,
         atom_min_share=args.atom_min_share, derive_total=args.derive_total,
         rank_gauss=args.rank_gauss,
         plot_path=os.path.join(out, name) if os.path.isdir(out) else None)
