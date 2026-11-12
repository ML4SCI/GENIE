"""
gendata.py -- loads Dataset 2 and builds the (x0, energy_cond) pairs that
flow_de.qVectorField/FlowDE and networks_calo.PhysicsNN operate on.

Analogous to the reference repo's gendata.load_or_generate_gmm_{1,2,3}d,
which build the (N, d) target tensor a PINNDE model is trained against.
Here the "target distribution" is real Dataset 2 calorimeter showers
instead of a synthetic GMM, and there is a second, conditioning variable
(incident energy) that the toy examples don't have.

Preprocessing matches train_auto_encoder_improved.py exactly (same
log1p(shower_GeV / energy_GeV) representation, same log10(E) standardization)
so a PINNDE-generated shower and an AE-reconstructed shower are directly
comparable, and so generate_and_export.py can reuse the AE pipeline's exact
inverse transform and HDF5 writer conventions.
"""

import torch

EPS = 1e-8


THRESHOLD_MEV = 0.01515      # CaloChallenge Dataset 2 readout threshold (15.15 keV)


def load_dataset2(data_path="data/Dataset2.pt", n=None, device="cpu",
                   standardize=True, representation="fraction"):
    """
    representation:
        "fraction"  (original, matches the AE pipeline): log1p(E_voxel / E_inc).
            Voxel fractions are <= 0.28, so log1p is almost the identity and
            an empty voxel and a 15.15 keV hit end up a median 5.5e-4
            standardized units apart -- far below any regression error.
        "threshold": log1p(E_voxel / 15.15 keV). Empty = 0, threshold = log 2,
            1 GeV deposit = 11.1; after standardization the empty-to-hit gap
            is 0.22 units (400x larger), so a model can actually resolve it.

    Returns:
        x0:      (N, 6480) float32 -- STANDARDIZED transform of the showers
        e_cond:  (N, 1)    float32 -- standardized log10(incident energy in GeV)
        energy_gev: (N, 1) float32 -- raw incident energy in GeV (for bookkeeping/plots)
        stats:   dict with the constants needed to invert both transforms

    Why x0 is standardized (measured, not assumed)
    ----------------------------------------------
    The reverse-time interpolant is x(t) = alpha_t*x0 + sigma_t*eps with
    eps ~ N(0, I). In raw log1p space the data occupies a ball of radius
    ||x0|| ~ 0.0555 while the noise lives on a shell of radius
    ||eps|| ~ sqrt(6480) = 80.5 -- the noise is 1450x larger than the signal.
    The vector field G is then ~99.9% noise direction (measured: ||G|| =
    79.61 at every t, essentially sqrt(d)), so the physics residual can be
    driven to ~1.5 and still carry error 30x larger than the entire data
    signal. That is what made MSE_f and MSE_u irreconcilable: they were
    describing different things.

    The toy 1-3D GMM cases this formalism was validated on had the opposite
    regime (signal radius ~4.5 vs noise radius 1.0-1.7). Standardizing x0 to
    unit per-dimension variance restores that regime -- it is the same
    normalization every practical diffusion model applies before adding
    unit-variance noise, and it is an exactly invertible change of variables,
    so the formalism itself is untouched.

    A single global scalar mean/std is used rather than per-dimension stats:
    ~76% of voxels are exactly zero and many are near-always-zero, so
    per-dimension division would blow up dead voxels into pure amplified
    noise. Global scaling preserves the relative structure across voxels.
    """
    data = torch.load(data_path, map_location="cpu")
    showers_mev = data["showers"].float()          # (N, 45, 16, 9), MeV
    energies_mev = data["energies"].float()         # (N, 1), MeV

    if n is not None and n < showers_mev.shape[0]:
        idx = torch.randperm(showers_mev.shape[0])[:n]
        showers_mev, energies_mev = showers_mev[idx], energies_mev[idx]

    showers_gev = showers_mev / 1000.0
    energy_gev = energies_mev / 1000.0

    if representation == "fraction":
        x_log = torch.log1p(showers_gev / (energy_gev.reshape(-1, 1, 1, 1) + EPS))
    elif representation == "threshold":
        x_log = torch.log1p(showers_mev / THRESHOLD_MEV)
    else:
        raise ValueError(f"unknown representation {representation!r}")
    x0 = x_log.reshape(x_log.shape[0], -1)          # (N, 6480)

    if standardize:
        x0_mean = x0.mean().item()
        x0_std = x0.std().item()
        x0 = (x0 - x0_mean) / (x0_std + EPS)
    else:
        x0_mean, x0_std = 0.0, 1.0

    log_e = torch.log10(energy_gev + EPS)
    log_e_mean = log_e.mean().item()
    log_e_std = log_e.std().item()
    e_cond = (log_e - log_e_mean) / log_e_std

    stats = {"log_e_mean": log_e_mean, "log_e_std": log_e_std,
             "x0_mean": x0_mean, "x0_std": x0_std,
             "representation": representation,
             "Nz": showers_mev.shape[1], "Nphi": showers_mev.shape[2],
             "Nr": showers_mev.shape[3]}

    return x0.to(device), e_cond.to(device), energy_gev.to(device), stats


def unstandardize_x0(x0_std, stats):
    """Invert the x0 standardization applied in load_dataset2 -- maps the
    model's output back into raw log1p(shower/energy) space before
    x0_to_physical_showers() exponentiates it."""
    return x0_std * (stats.get("x0_std", 1.0) + EPS) + stats.get("x0_mean", 0.0)


def empty_value(stats):
    """The standardized value an EMPTY voxel takes. Both representations map
    zero energy to log1p(0) = 0, so this is just where 0 lands after
    standardization. The hit-gated read-out writes exactly this value, which
    inverts to exactly 0 MeV."""
    return (0.0 - stats.get("x0_mean", 0.0)) / (stats.get("x0_std", 1.0) + EPS)


def energy_gev_to_cond(energy_gev, stats):
    """Standardize a raw energy (GeV) using stats saved at training time --
    needed at sampling time when the caller specifies a target energy."""
    log_e = torch.log10(energy_gev + EPS)
    return (log_e - stats["log_e_mean"]) / stats["log_e_std"]


def x0_to_physical_showers(x0_flat, energy_gev, stats):
    """Inverse of the transform above: (N, 6480) UNstandardized log-space ->
    (N, Nz, Nphi, Nr) physical showers in GeV. For the "fraction"
    representation this matches model_to_physical() in
    train_auto_encoder_improved.py exactly."""
    Nz, Nphi, Nr = stats["Nz"], stats["Nphi"], stats["Nr"]
    x_log = x0_flat.reshape(-1, Nz, Nphi, Nr)
    if stats.get("representation", "fraction") == "threshold":
        return torch.clamp(torch.expm1(x_log), min=0.0) * (THRESHOLD_MEV / 1000.0)
    energy = energy_gev.reshape(-1, 1, 1, 1)
    return torch.clamp(torch.expm1(x_log), min=0.0) * energy
