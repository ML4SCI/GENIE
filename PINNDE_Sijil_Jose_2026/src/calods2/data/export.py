"""
export.py -- shared, failure-tolerant conversion of generated samples into a
CaloChallenge HDF5, with diagnostics for sampler blow-up.

Why this exists: a 100k-step run produced a few voxels so large that
expm1() overflowed float32 (E_dep/E_inc printed `inf`), and the old inline
export then died on `assert np.isfinite(flat).all()` -- AFTER the whole
training and sampling had finished. Two separate things were wrong:

  * a blow-up is a model/sampler result worth MEASURING, not an assertion
    failure. physical_showers_mev() clamps to what is physically possible
    and reports exactly how much it had to clamp.
  * the clamp itself is physics, not a fudge: a single voxel cannot hold
    more than the incident energy, so x_log = log1p(E_voxel/E_inc) > log(2)
    is unphysical by construction.

A large `clipped` count means the sampler is diverging for some noise draws
and the numbers downstream are not trustworthy -- fix the model, don't lean
on the clamp.
"""

import numpy as np
import torch

from . import dataset as gendata

# log1p(E_voxel / E_inc) = log(2) means a single voxel holds the entire
# incident energy. Anything above that is unphysical.
MAX_X_LOG = float(np.log(2.0))


def diverged_mask(x0_std, stats, max_x_log=None):
    """Per-shower: did the ODE run away? True where any voxel exceeds the
    incident energy (or is non-finite), which is unphysical by construction.

    Measured at 100k training steps: 5 showers in 2048 (0.24%), all with
    E_inc in 1-10 GeV -- the widest part of the shower distribution, where
    the learned field is sharpest and the reverse ODE (|lambda| ~ 1/sigma0
    = 100 near t=0) is stiffest. Everything else was clean.
    """
    # Read the module global at call time, so this and physical_showers_mev
    # can never disagree about what counts as diverged.
    if max_x_log is None:
        max_x_log = MAX_X_LOG
    x_log = gendata.unstandardize_x0(x0_std.double(), stats)
    return (~torch.isfinite(x_log) | (x_log > max_x_log)).any(dim=1)


def physical_showers_mev(x0_std, energy_gev, stats, report=True):
    """
    Standardized model output -> (showers in MeV, diagnostics dict).

    Clamps each voxel to at most the incident energy and reports what that
    cost. Works in float64 for the energy sums so a blow-up shows up as a
    large number rather than `inf`.
    """
    x_log = gendata.unstandardize_x0(x0_std.double(), stats)
    n_voxels = x_log.numel()

    nonfinite = int((~torch.isfinite(x_log)).sum())
    over = int((x_log > MAX_X_LOG).sum())
    showers_over = int((x_log > MAX_X_LOG).any(dim=1).sum())
    x_log = torch.nan_to_num(x_log, nan=0.0, posinf=MAX_X_LOG, neginf=0.0)
    x_log = x_log.clamp(max=MAX_X_LOG)

    showers_gev = gendata.x0_to_physical_showers(x_log, energy_gev.double(), stats)
    showers_mev = (showers_gev * 1000.0).float()

    ratio = (showers_gev.reshape(showers_gev.shape[0], -1).sum(1)
             / energy_gev.double().reshape(-1))
    diag = {
        "clipped_voxels": over,
        "clipped_frac": over / max(1, n_voxels),
        "showers_affected": showers_over,
        "showers_frac": showers_over / max(1, x_log.shape[0]),
        "nonfinite_voxels": nonfinite,
        "ratio_median": ratio.median().item(),
        "ratio_mean": ratio.mean().item(),
        "max_x_log": None,
    }
    if report:
        print(f"  physical conversion: clamped {over:,} voxels "
              f"({diag['clipped_frac']:.3e} of all) in {showers_over:,} showers "
              f"({diag['showers_frac']:.4f}); non-finite {nonfinite:,}")
        if showers_over:
            print("  WARNING: voxels above the incident energy mean the sampler is "
                  "diverging for some noise draws -- treat the metrics as suspect.")
    return showers_mev, diag


def write_hdf5(path, showers_mev, einc_mev, stats, threshold_kev=15.15):
    """Write the CaloChallenge layout. Returns the thresholded array."""
    import h5py

    sh = showers_mev.numpy().astype(np.float32) if torch.is_tensor(showers_mev) else showers_mev
    sh = sh.reshape(sh.shape[0], -1)
    if threshold_kev > 0:
        thr = threshold_kev / 1000.0
        before = int((sh > 0).sum())
        sh = np.where(sh < thr, 0.0, sh)
        print(f"  applied {threshold_kev} keV threshold: {before:,} -> "
              f"{int((sh > 0).sum()):,} non-zero voxels")

    assert sh.shape[1] == stats["Nz"] * stats["Nphi"] * stats["Nr"]
    if not np.isfinite(sh).all() or (sh < 0).any():
        # Should be impossible after physical_showers_mev; keep the file rather
        # than losing the run, and say so loudly.
        bad = int((~np.isfinite(sh)).sum() + (sh < 0).sum())
        print(f"  WARNING: {bad:,} non-finite/negative voxels remained; zeroing them")
        sh = np.nan_to_num(sh, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)

    e = einc_mev.numpy().astype(np.float32) if torch.is_tensor(einc_mev) else einc_mev
    with h5py.File(path, "w") as f:
        f.create_dataset("incident_energies", data=e.reshape(sh.shape[0], -1),
                         compression="gzip")
        f.create_dataset("showers", data=sh, compression="gzip")
    print(f"  wrote {path}")
    return sh
