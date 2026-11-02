"""
generate_caloeval_hdf5.py
=========================
Runs the trained FactorizedCaloAE and writes a CaloChallenge-format HDF5.

UNITS -- read carefully before changing anything here
------------------------------------------------------
Dataset2.pt contains:
    data['energies'] : MeV  (range 1,000 -- 1,000,000 MeV = 1 GeV -- 1 TeV)
    data['showers']  : MeV  (voxel deposits, max ~6000 MeV)

The training script does:
    energies = data['energies'] / 1000.0   -> GeV
    showers  = data['showers']             -> MeV  (NO conversion)

So the model was trained on:
    x_net = log1p( shower_MeV / E_GeV )
           = log1p( shower_MeV * 1000 / E_MeV )
           = log1p( 1000 * true_energy_fraction )

model_to_physical correctly inverts this:
    shower_hat = expm1(x_hat) * E_GeV
               = values in MeV  (numerically)

So after inference the reconstructed showers are already in MeV.
The incident energies (which were divided by 1000) must be multiplied
back by 1000 to restore MeV for the CaloChallenge evaluator.

SUMMARY of what this script does:
    energies_model = data['energies'] / 1000          [GeV, matches training]
    showers_model  = data['showers']                   [MeV, matches training]
    x_net = log1p(showers_model / energies_model)      [matches training]
    x_hat = model(x_net)
    recon_mev = expm1(x_hat) * energies_model          [MeV output]
    einc_mev  = energies_model * 1000                  [MeV, restored]
    -> write recon_mev and einc_mev to HDF5             [both MeV, correct]

Run:
    python models/1_latent_diffusion/generate_caloeval_hdf5.py \
        --weights best_calo_ae_cond_improved.pt \
        --data    data/Dataset2.pt \
        --out     ae_recon_ds2.hdf5 \
        --split   val          # or 'all' for 100k events (needed for cls-* modes)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import DataLoader, TensorDataset, random_split


# ============================================================
# Model: imported, not copied
#
# This file used to carry its own 304-line copy of FactorizedCaloAE and its
# blocks. That copy used a decoder intermediate transverse resolution of
# (8, 4) while the training script had been corrected to the encoder's true
# mirror (8, 5) -- and because conv weights do not encode spatial size, each
# loaded into the other with zero missing or unexpected keys and silently
# reconstructed ~10% worse. Every checkpoint in this project was in fact
# trained at (8, 4), so the copy was right and the main script had drifted.
#
# The library now takes `mid_transverse` explicitly, records it in new
# checkpoints, and from_checkpoint() defaults a checkpoint without that field
# to the legacy (8, 4). Importing is therefore both correct and safe here.
# ============================================================

import os as _os, sys as _sys
# Walk up to the repo root rather than assuming a fixed depth: these scripts
# have been moved once already (out of the flat CaloChallenge/ directory) and
# a hardcoded "../src" silently breaks on the next move.
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d):
    if _os.path.isdir(_os.path.join(_d, "src", "calods2")):
        _sys.path.insert(0, _os.path.join(_d, "src"))
        break
    _d = _os.path.dirname(_d)
from calods2.models.autoencoder import FactorizedCaloAE, LEGACY_MID_TRANSVERSE

# ============================================================
# Preprocessing -- MUST match training exactly
#
# Training used:
#   energies_GeV = energies_MeV / 1000
#   x_net = log1p(shower_MeV / energies_GeV)
#
# Inverse:
#   shower_MeV_hat = expm1(x_hat) * energies_GeV
#
# Both functions below replicate this mixed-unit convention so that
# the loaded weights produce correct numerical outputs.
# ============================================================

def physical_to_model(shower_mev, energy_gev):
    """shower [MeV] + energy [GeV] -> log1p(shower_MeV / E_GeV). Matches training."""
    E = energy_gev.reshape(-1, 1, 1, 1)
    return torch.log1p(shower_mev / (E + 1e-8))


def model_to_physical_mev(x_hat_log, energy_gev):
    """model output -> MeV. Clamps >=0 (linear output may go slightly negative)."""
    E = energy_gev.reshape(-1, 1, 1, 1)
    return torch.clamp(torch.expm1(x_hat_log), min=0.0) * E


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="Generate CaloChallenge HDF5 from the trained AE.")
    p.add_argument('--weights',    default='best_calo_ae_cond_improved.pt')
    p.add_argument('--data',       default='data/Dataset2.pt')
    p.add_argument('--out',        default='ae_recon_ds2.hdf5')
    p.add_argument('--split',      choices=['val', 'all'], default='val',
                   help="'val' = held-out 10%% (seed 42); "
                        "'all' = full dataset (needed for cls-* modes which "
                        "require 100k events to fill the 60k/20k/20k split)")
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--threshold_kev', type=float, default=15.15,
                   help='Readout threshold in keV (Dataset 2/3 = 15.15 keV). '
                        'Set 0 to disable.')
    p.add_argument('--no_cuda',    action='store_true')
    args = p.parse_args()

    device = torch.device(
        'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    print(f'Device: {device}')

    # ---- load checkpoint ----
    ckpt = torch.load(args.weights, map_location=device)
    if not isinstance(ckpt, dict) or 'model_state' not in ckpt:
        raise ValueError(
            "Expected the improved-trainer checkpoint dict "
            "(keys: model_state, log_e_mean, log_e_std, Nz, Nphi, Nr, ...).")

    LOG_E_MEAN = ckpt['log_e_mean']
    LOG_E_STD  = ckpt['log_e_std']
    Nz  = ckpt.get('Nz',  45)
    Nphi = ckpt.get('Nphi', 16)
    Nr  = ckpt.get('Nr',   9)
    latent_channels = ckpt.get('latent_channels', 8)
    latent_length   = ckpt.get('latent_length',  12)

    print(f'Checkpoint: epoch={ckpt.get("epoch")}, '
          f'val_loss={ckpt.get("val_loss", float("nan")):.5f}')
    print(f'Energy conditioning: log10(E_GeV) mean={LOG_E_MEAN:.4f}, '
          f'std={LOG_E_STD:.4f}')

    # ---- load data exactly as training did ----
    data = torch.load(args.data)
    energies_gev = data['energies'].float() / 1000.0   # MeV -> GeV (matches training)
    showers_mev  = data['showers'].float()             # MeV, kept raw (matches training)
    N_total = showers_mev.shape[0]
    print(f'Loaded {N_total} showers  |  '
          f'E range: {energies_gev.min()*1000:.0f}–{energies_gev.max()*1000:.0f} MeV  |  '
          f'shower max voxel: {showers_mev.max():.1f} MeV')

    # quick sanity on the raw file: E_dep/E_inc (both MeV) is ~0.78 for Dataset 2
    ratio = (showers_mev.sum(dim=(1, 2, 3)) / (energies_gev.squeeze() * 1000.0 + 1e-8)).mean()
    print(f'Raw data E_dep/E_inc = {ratio:.4f}  (Dataset 2: ~0.78)')

    # ---- select split ----
    if args.split == 'val':
        full = TensorDataset(showers_mev, energies_gev)
        train_size = int(0.9 * N_total)
        gen = torch.Generator().manual_seed(42)
        _, val_sub = random_split(full, [train_size, N_total - train_size],
                                  generator=gen)
        idx = val_sub.indices
        showers_mev  = showers_mev[idx]
        energies_gev = energies_gev[idx]
        print(f'Held-out val split: {len(idx)} showers')
    else:
        print(f'Using all {N_total} showers')

    energies_gev = energies_gev.reshape(-1, 1)

    # ---- build model ----
    two_head = ckpt.get('two_head', False)
    model = FactorizedCaloAE(
        Nz=Nz, Nphi=Nphi, Nr=Nr,
        latent_channels=latent_channels, latent_length=latent_length,
        energy_embed_dim=ckpt.get('energy_embed_dim', 64),
        two_head=two_head, x_scale=ckpt.get('x_scale', 1.0),
        log_e_mean=LOG_E_MEAN, log_e_std=LOG_E_STD,
        mid_transverse=ckpt.get('mid_transverse', LEGACY_MID_TRANSVERSE),
    ).to(device)
    # strict=False only to tolerate hit_logit_scale being absent from very
    # old checkpoints; any other missing/unexpected key is still an error.
    missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
    if unexpected or [k for k in missing if k != 'hit_logit_scale']:
        raise RuntimeError(f'checkpoint mismatch: missing={missing} unexpected={unexpected}')
    print(f'Architecture: {"v4 two-head" if two_head else "single-output (pre-v4)"}')
    # Checkpoints saved after the units fix carry shower_units_gev=True and
    # were trained on log1p(shower_GeV / E_GeV); feeding them the old
    # log1p(shower_MeV / E_GeV) inputs would be 1000x off.
    unit_div = 1000.0 if ckpt.get('shower_units_gev', False) else 1.0
    print(f'Input units: {"shower GeV / E GeV" if unit_div == 1000.0 else "shower MeV / E GeV (legacy)"}')
    model.eval()
    print(f'Model loaded from {args.weights}')

    # ---- run inference ----
    loader = DataLoader(TensorDataset(showers_mev, energies_gev),
                        batch_size=args.batch_size, shuffle=False)

    recon_chunks = []
    einc_chunks  = []

    with torch.no_grad():
        for shower_b, e_b in loader:
            shower_b = shower_b.to(device)   # MeV
            e_b      = e_b.to(device)        # GeV

            # encode exactly as the checkpoint was trained:
            #   units-fixed (shower_units_gev): log1p(shower_GeV / E_GeV)
            #   older checkpoints:              log1p(shower_MeV / E_GeV)
            x_log = physical_to_model(shower_b / unit_div, e_b)

            # forward pass
            x_hat_log, _ = model(x_log, e_b)

            # decode back to MeV
            recon_mev_b = model_to_physical_mev(x_hat_log, e_b) * unit_div

            recon_chunks.append(recon_mev_b.cpu())
            einc_chunks.append(e_b.cpu())

    recon_mev = torch.cat(recon_chunks, dim=0).numpy().astype(np.float32)
    einc_gev  = torch.cat(einc_chunks,  dim=0).numpy().astype(np.float32)

    # ---- convert incident energies back to MeV for the evaluator ----
    # recon_mev is already in MeV -- DO NOT multiply by 1000
    # einc_gev was divided by 1000 at load time -- restore to MeV
    einc_mev = einc_gev * 1000.0

    # ---- verify the ratio is now physically sane ----
    ratio_out = (recon_mev.sum(axis=(1, 2, 3)) / einc_mev.squeeze()).mean()
    ratio_in = (showers_mev.sum(dim=(1, 2, 3)) / (energies_gev.squeeze() * 1000.0)).mean().item()
    print(f'Output E_dep/E_inc = {ratio_out:.4f}   (input showers: {ratio_in:.4f}; '
          f'Dataset 2 is ~0.78)')
    if not (0.5 * ratio_in < ratio_out < 2.0 * ratio_in):
        print('WARNING: output E_dep/E_inc is off by more than 2x from the input '
              '-- possible units error or a badly trained model!')
    else:
        print('Units check PASSED.')

    # ---- apply readout threshold ----
    if args.threshold_kev > 0:
        thr_mev = args.threshold_kev / 1000.0
        n_before = int((recon_mev > 0).sum())
        recon_mev = np.where(recon_mev < thr_mev, 0.0, recon_mev)
        n_after = int((recon_mev > 0).sum())
        print(f'Applied {args.threshold_kev} keV threshold: '
              f'{n_before:,} -> {n_after:,} non-zero voxels')

    # ---- flatten to CaloChallenge layout (N, Nz*Nphi*Nr) ----
    N = recon_mev.shape[0]
    recon_flat = recon_mev.reshape(N, -1)
    einc_flat  = einc_mev.reshape(N, -1)

    assert recon_flat.shape[1] == Nz * Nphi * Nr, \
        f'Voxel count mismatch: got {recon_flat.shape[1]}, expected {Nz*Nphi*Nr}'
    assert (recon_flat >= 0).all(), 'Negative voxel energies in output!'
    assert np.isfinite(recon_flat).all(), 'Non-finite values in output!'

    print(f'Output shapes: incident_energies {einc_flat.shape}, '
          f'showers {recon_flat.shape}')

    # ---- write HDF5 ----
    with h5py.File(args.out, 'w') as f:
        f.create_dataset('incident_energies', data=einc_flat, compression='gzip')
        f.create_dataset('showers',           data=recon_flat, compression='gzip')

    print(f'\nWrote {args.out}')
    print('Run: bash run_calochallenge_eval.sh '
          f'{args.out} <reference_geant4.hdf5>')


if __name__ == '__main__':
    main()
