"""CaloChallenge shower observables -- the ``features_fn`` for real data.

``evaluate(real, gen, features_fn=...)`` maps raw samples to physics features
before any metric runs. This module supplies that function for CaloChallenge
voxelised showers, turning an ``(N, n_voxels)`` array of energy deposits into
the high-level observables physicists actually use to judge shower quality.

Definitions follow the CaloChallenge 2022 paper (arXiv:2410.21611), section on
high-level features:

* ``E_tot``     total deposited energy, sum over all voxels
* ``f_samp``    sampling fraction, E_tot / E_inc
* ``<z>``       longitudinal centre of gravity, energy-weighted mean layer
* ``sigma_z``   longitudinal width, energy-weighted RMS in layer
* ``<r>``       transverse centre of gravity, energy-weighted mean radius
* ``sigma_r``   transverse width, energy-weighted RMS in radius
* ``sparsity``  fraction of voxels with zero energy

**Voxel ordering.** The flattened voxel axis is ``(layer, alpha, r)`` -- layer
slowest, radius fastest. This was determined empirically from ds2 rather than
assumed: under this ordering the radial profile falls off monotonically by a
factor ~50 from core to edge and the angular profile is flat to 0.3%
(azimuthal symmetry, as it must be). Under the transposed ordering the radial
profile is a physically impossible sawtooth. Getting this backwards leaves
``<z>`` looking plausible while silently corrupting ``sigma_r``, so
``test_observables.py`` asserts the convention on synthetic showers.

**Radial coordinate.** By default radius is the bin *index* (0.5, 1.5, ...),
a monotonic proxy that is correct up to the radial binning. The real detector
bins are not equal width; for exact CaloChallenge parity pass the official bin
centres via ``Geometry(..., r_centers=...)``.
"""

import numpy as np

from ._utils import to_numpy

# CaloChallenge readout threshold: 15.15 keV, expressed in MeV (data is in MeV).
READOUT_THRESHOLD_MEV = 0.01515

CORE_FEATURES = ("E_tot", "f_samp", "z_mean", "z_width", "r_mean", "r_width",
                 "sparsity")


class Geometry:
    """Voxel grid of a CaloChallenge dataset.

    ``n_voxels == n_layers * n_alpha * n_r``, flattened as (layer, alpha, r).
    ``r_centers`` defaults to bin indices + 0.5; pass the official bin centres
    for exact parity with the CaloChallenge helper scripts.
    """

    def __init__(self, n_layers, n_alpha, n_r, r_centers=None, name=""):
        self.n_layers = int(n_layers)
        self.n_alpha = int(n_alpha)
        self.n_r = int(n_r)
        self.name = name
        if r_centers is None:
            r_centers = np.arange(self.n_r) + 0.5
        r_centers = np.asarray(r_centers, dtype=np.float64)
        if r_centers.shape != (self.n_r,):
            raise ValueError(f"r_centers must have shape ({self.n_r},), "
                             f"got {r_centers.shape}")
        self.r_centers = r_centers

    @property
    def n_voxels(self):
        return self.n_layers * self.n_alpha * self.n_r

    def reshape(self, showers):
        """(N, n_voxels) -> (N, n_layers, n_alpha, n_r)."""
        showers = np.asarray(showers, dtype=np.float64)
        if showers.ndim != 2:
            raise ValueError(f"showers must be 2-D (N, n_voxels), got shape "
                             f"{showers.shape}")
        if showers.shape[1] != self.n_voxels:
            raise ValueError(
                f"geometry {self.name or ''} expects {self.n_voxels} voxels "
                f"({self.n_layers}x{self.n_alpha}x{self.n_r}), got "
                f"{showers.shape[1]}")
        return showers.reshape(-1, self.n_layers, self.n_alpha, self.n_r)

    def __repr__(self):
        return (f"Geometry(n_layers={self.n_layers}, n_alpha={self.n_alpha}, "
                f"n_r={self.n_r}, name={self.name!r})")


# The two regular-grid CaloChallenge datasets. ds1 (photons/pions) has an
# irregular geometry with varying bins per layer and needs its own handling.
GEOMETRIES = {
    "ds2": Geometry(45, 16, 9, name="ds2"),
    "ds3": Geometry(45, 50, 18, name="ds3"),
}


def get_geometry(geometry):
    """Accept a Geometry, or a name from ``GEOMETRIES`` ("ds2", "ds3")."""
    if isinstance(geometry, Geometry):
        return geometry
    try:
        return GEOMETRIES[geometry]
    except KeyError:
        raise ValueError(f"unknown geometry {geometry!r}; pass a Geometry or "
                         f"one of {sorted(GEOMETRIES)}") from None


def _weighted_mean_and_width(weights, coords, axis):
    """Energy-weighted mean and RMS of ``coords`` along ``axis``.

    Showers with zero total weight give 0 for both rather than NaN, so a dead
    event cannot poison a whole batch of metrics.
    """
    total = weights.sum(axis=axis)
    safe = np.where(total > 0, total, 1.0)
    mean = (weights * coords).sum(axis=axis) / safe
    mean_sq = (weights * coords ** 2).sum(axis=axis) / safe
    var = np.clip(mean_sq - mean ** 2, 0.0, None)   # clip float cancellation
    return np.where(total > 0, mean, 0.0), np.where(total > 0, np.sqrt(var), 0.0)


def layer_energies(showers, geometry="ds2"):
    """Per-layer deposited energy. Returns ``(N, n_layers)``."""
    geom = get_geometry(geometry)
    return geom.reshape(to_numpy(showers)).sum(axis=(2, 3))


def radial_profile(showers, geometry="ds2"):
    """Energy summed over layers and angle, per radial bin. ``(N, n_r)``."""
    geom = get_geometry(geometry)
    return geom.reshape(to_numpy(showers)).sum(axis=(1, 2))


def voxel_energy_spectrum(showers, geometry="ds2", threshold=0.0):
    """Pooled energies of all voxels above ``threshold``. Returns 1-D.

    This is a distribution over voxels rather than a per-shower scalar, so it
    is not part of the feature vector; compare it between real and generated
    with ``histogram_chi2`` or ``separation_power`` directly.
    """
    x = to_numpy(showers).ravel()
    return x[x > threshold]


def discrete_observables(geometry="ds2"):
    """Observables that live on a lattice, and their spacing.

    ``sparsity`` is a *count*: the fraction of empty voxels, so it takes only
    the values ``1 - k/n_voxels`` for integer k. That matters for generative
    models -- a continuous density cannot reproduce an atomic comb, and the
    coarseness is energy dependent. In ds2 a low-energy shower lights 6-395 of
    6480 voxels, giving ~350 distinct sparsity values, while a high-energy one
    lights thousands and looks continuous. Models should dequantize it during
    training and quantize back when sampling.

    Returns ``{name: spacing}``.
    """
    geom = get_geometry(geometry)
    return {"sparsity": 1.0 / geom.n_voxels}


def per_layer_observables(showers, geometry="ds2", threshold=0.0):
    """Per-layer observables, matching the official CaloChallenge definitions.

    Returns ``(features, names)`` with four quantities for each layer, in
    blocks: energy, sparsity, radial centre of energy, radial width. For ds2
    that is 4 x 45 = 180 columns.

    The official ``HighLevelFeatures`` class computes exactly these per layer
    (``E_layers``, ``sparsity``, ``EC_r``, ``width_r``) using the same
    energy-weighted RMS, so results here are comparable with published
    CaloChallenge numbers. Its remaining per-layer features -- the centres and
    widths in eta and phi -- need the detector eta/phi maps from ``binning.xml``
    and are not reproduced here; ``sigma_r`` carries the transverse
    information instead.

    ``shower_observables`` gives the whole-shower versions of the same
    quantities: coarser, but 7 numbers instead of 180.
    """
    geom = get_geometry(geometry)
    x = to_numpy(showers).astype(np.float64)
    if threshold > 0:
        x = np.where(x > threshold, x, 0.0)
    v = geom.reshape(x)                       # (N, layer, alpha, r)

    layer_e = v.sum(axis=(2, 3))              # (N, n_layers)
    sparsity = (v == 0).mean(axis=(2, 3))     # (N, n_layers)
    radial_e = v.sum(axis=2)                  # (N, n_layers, n_r)
    r_mean, r_width = _weighted_mean_and_width(radial_e, geom.r_centers, axis=2)

    names = ([f"E_layer_{i}" for i in range(geom.n_layers)]
             + [f"sparsity_layer_{i}" for i in range(geom.n_layers)]
             + [f"r_mean_layer_{i}" for i in range(geom.n_layers)]
             + [f"r_width_layer_{i}" for i in range(geom.n_layers)])
    return np.hstack([layer_e, sparsity, r_mean, r_width]), names


def shower_observables(showers, incident_energies, geometry="ds2",
                       include_layers=False, threshold=0.0):
    """High-level observables per shower. Returns ``(features, names)``.

    ``showers`` is ``(N, n_voxels)`` of energy deposits (MeV),
    ``incident_energies`` is ``(N,)`` or ``(N, 1)`` (MeV). Features are the
    seven in ``CORE_FEATURES``, optionally followed by the per-layer energies.
    Voxels at or below ``threshold`` are zeroed first; the CaloChallenge
    readout threshold is ``READOUT_THRESHOLD_MEV``.
    """
    geom = get_geometry(geometry)
    x = to_numpy(showers).astype(np.float64)
    e_inc = np.asarray(to_numpy(incident_energies), dtype=np.float64).ravel()
    if len(e_inc) != len(x):
        raise ValueError(f"incident_energies has {len(e_inc)} rows, showers "
                         f"has {len(x)}")
    if threshold > 0:
        x = np.where(x > threshold, x, 0.0)

    v = geom.reshape(x)                       # (N, layer, alpha, r)
    e_tot = v.sum(axis=(1, 2, 3))

    layer_e = v.sum(axis=(2, 3))              # (N, n_layers)
    z_coord = np.arange(geom.n_layers, dtype=np.float64)
    z_mean, z_width = _weighted_mean_and_width(layer_e, z_coord, axis=1)

    radial_e = v.sum(axis=(1, 2))             # (N, n_r)
    r_mean, r_width = _weighted_mean_and_width(radial_e, geom.r_centers, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_samp = np.where(e_inc > 0, e_tot / np.where(e_inc > 0, e_inc, 1.0), 0.0)
    sparsity = (v == 0).mean(axis=(1, 2, 3))

    names = list(CORE_FEATURES)
    cols = [e_tot, f_samp, z_mean, z_width, r_mean, r_width, sparsity]
    if include_layers:
        names += [f"E_layer_{i}" for i in range(geom.n_layers)]
        cols += [layer_e[:, i] for i in range(geom.n_layers)]
    return np.column_stack(cols), names


def shower_features_fn(incident_energies, geometry="ds2", include_layers=False,
                       per_layer=False, threshold=0.0):
    """Build a ``features_fn`` for ``evaluate`` bound to fixed conditions.

    ``evaluate`` applies one ``features_fn`` to both real and generated, so
    this is for the matched-condition setting where the two sets share
    incident energies (the usual conditional-generation comparison)::

        fn = shower_features_fn(e_inc, geometry="ds2")
        results = evaluate(real_showers, gen_showers, features_fn=fn,
                           standardize=True)

    ``per_layer=True`` appends the 4-per-layer block from
    ``per_layer_observables``, giving the resolution published CaloChallenge
    numbers are quoted at. Always pass ``standardize=True`` alongside it --
    layer energies span orders of magnitude across the shower.
    """
    def features_fn(showers):
        feats, _ = shower_observables(showers, incident_energies,
                                      geometry=geometry,
                                      include_layers=include_layers,
                                      threshold=threshold)
        if per_layer:
            layer_feats, _ = per_layer_observables(showers, geometry=geometry,
                                                   threshold=threshold)
            feats = np.hstack([feats, layer_feats])
        return feats
    return features_fn


def load_calochallenge(path, n=None, start=0, dtype=np.float64):
    """Read ``(showers, incident_energies)`` from a CaloChallenge HDF5 file.

    Reads a slice rather than the whole file: ds2 is 100k x 6480 float64, i.e.
    ~5 GB per file in memory. The stability study (DEVLOG section 8) shows
    N of a few thousand is enough for trustworthy metrics, so a slice is
    usually the right thing to work with.

    ``n=None`` reads to the end from ``start``.
    """
    import h5py

    with h5py.File(path, "r") as f:
        stop = None if n is None else start + n
        showers = f["showers"][start:stop].astype(dtype)
        e_inc = f["incident_energies"][start:stop].astype(dtype)
    return showers, e_inc.ravel()


def observables_from_file(path, n, start=0, geometry="ds2", chunk=5000,
                          per_layer=False, threshold=0.0):
    """Observables for ``n`` showers, reading the file in chunks.

    Returns ``(features, names, incident_energies)``. Raw ds2 is float64, so
    100k showers is ~5 GB; the observables are a handful of numbers each.
    Reading in chunks keeps peak memory at one chunk regardless of ``n``.
    """
    feat_blocks, e_blocks, names = [], [], None
    for s in range(start, start + n, chunk):
        showers, e_inc = load_calochallenge(path, n=min(chunk, start + n - s),
                                            start=s)
        feats, names = shower_observables(showers, e_inc, geometry=geometry,
                                          threshold=threshold)
        if per_layer:
            layer_feats, layer_names = per_layer_observables(
                showers, geometry=geometry, threshold=threshold)
            feats = np.hstack([feats, layer_feats])
            names = names + layer_names
        feat_blocks.append(feats)
        e_blocks.append(e_inc)
        del showers
    return np.vstack(feat_blocks), names, np.concatenate(e_blocks)
