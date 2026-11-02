"""
calo_dataset.py -- Dataset 2 loading and the autoencoder's log transform.

Extracted from train_auto_encoder_improved.py, which did all of this at IMPORT
time: `data = torch.load('data/Dataset2.pt')` ran on `import`, so merely
importing the module pulled 2.5 GB off disk and built DataLoaders. Any other
module that wanted the autoencoder class had to work around it --
train_latent.py set sys.argv and CALO_BATCH_SIZE before importing, purely to
survive the side effects.

Everything here is now an explicit call.

The standardization constants for log10(E) are measured over the full training
set. They are defaults only: load_calo_data() recomputes them and every model
stores its own in its checkpoint, because a model must be evaluated with the
constants it was trained with.
"""

import os

import torch
from torch.utils.data import DataLoader, Dataset, random_split

# measured over all 100k showers of dataset_2_1 (E in GeV)
DS2_LOG_E_MEAN = 1.4973
DS2_LOG_E_STD = 0.8660

THRESHOLD_MEV = 0.01515          # Dataset-2 readout threshold, 15.15 keV
THRESHOLD_LOG = float(torch.log1p(torch.tensor(THRESHOLD_MEV / 1000.0)))


class CaloDataset(Dataset):
    """RAW (shower, energy) pairs in GeV. The log transform lives in the
    training step, not here -- an earlier version applied it in both places,
    giving a double-log bug."""

    def __init__(self, showers, energies):
        assert showers.shape[0] == energies.shape[0]
        self.showers = showers
        self.energies = energies

    def __len__(self):
        return self.showers.shape[0]

    def __getitem__(self, idx):
        return self.showers[idx], self.energies[idx]


def load_calo_data(path="data/Dataset2.pt"):
    """-> (showers_gev, energies_gev, stats).

    Dataset2.pt stores BOTH arrays in MeV; both are converted to GeV here so
    that log1p(shower / energy) is a true energy fraction. Converting only the
    energies -- the original bug -- inflates the ratio ~1000x, stretches the
    log1p target range ~15x, and is the likeliest cause of a long, stubborn
    training plateau. Checkpoints trained that way are marked with
    `shower_units_gev`.
    """
    data = torch.load(path, map_location="cpu")
    energies = data["energies"] / 1000.0
    showers = data["showers"] / 1000.0
    log_e = torch.log10(energies + 1e-8)
    stats = {
        "log_e_mean": log_e.mean().item(),
        "log_e_std": log_e.std().item(),
        "n_showers": showers.shape[0],
        "geometry": tuple(showers.shape[1:]),
    }
    return showers, energies, stats


def make_loaders(showers, energies, batch_size=None, train_frac=0.9, seed=42,
                 num_workers=0):
    """-> (train_loader, val_loader, train_indices).

    The split seed is fixed at 42 so that every run and every sweep arm sees
    the same validation showers. train_indices is returned because x_scale is
    computed from a subset of the TRAIN split only.

    Data is already a tensor in RAM, so workers add overhead and no value.
    """
    if batch_size is None:
        batch_size = int(os.environ.get("CALO_BATCH_SIZE", 4096))
    ds = CaloDataset(showers, energies)
    n_train = int(train_frac * len(ds))
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=min(256, batch_size), shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.indices


def compute_x_scale(showers, energies, train_indices, n=20000):
    """Global std of the log-space target over a fixed subset of the TRAIN
    split, ~8e-4. The model works on x_log / x_scale so its inputs and value
    head are O(1) instead of ~1e-4. Saved as a buffer in the checkpoint."""
    idx = torch.tensor(train_indices[:n])
    return physical_to_model(showers[idx].float(), energies[idx].float()).std().item()


def physical_to_model(x, energy):
    """
    x:      [B, Nz, Nphi, Nr]   raw shower in GeV per voxel
    energy: any shape that flattens to [B]
    returns x_log: log1p(x / E_incident)
    """
    assert x.dim() == 4, (
        f"physical_to_model: expected x of shape [B, Nz, Nphi, Nr], "
        f"got {tuple(x.shape)}"
    )
    B = x.shape[0]
    assert energy.numel() == B, (
        f"physical_to_model: energy has {energy.numel()} elements, "
        f"expected {B} (one per shower)"
    )

    E = energy.reshape(-1, 1, 1, 1)
    x_norm = x / (E + 1e-8)
    x_log = torch.log1p(x_norm)

    assert x_log.shape == x.shape, (
        f"physical_to_model: output shape {tuple(x_log.shape)} "
        f"!= input shape {tuple(x.shape)}"
    )
    return x_log


def model_to_physical(x_log, energy):
    """
    Inverse of physical_to_model. Clamps to >=0 because the model uses linear
    output and may emit slightly negative values for empty voxels.
    """
    assert x_log.dim() == 4, (
        f"model_to_physical: expected x_log of shape [B, Nz, Nphi, Nr], "
        f"got {tuple(x_log.shape)}"
    )
    B = x_log.shape[0]
    assert energy.numel() == B, (
        f"model_to_physical: energy has {energy.numel()} elements, "
        f"expected {B}"
    )

    E = energy.reshape(-1, 1, 1, 1)
    x_norm = torch.clamp(torch.expm1(x_log), min=0.0)
    x = x_norm * E
    return x
