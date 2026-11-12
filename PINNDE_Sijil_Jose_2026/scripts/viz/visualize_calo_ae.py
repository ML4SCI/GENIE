"""
================================================================================
visualize_calo_ae.py  --  STANDALONE evaluation & visualization
================================================================================

Runs completely separately from the training script. It does NOT import the
trainer module (importing it would re-load the dataset and re-run setup code,
since that logic is at module level in the trainer).

Instead this file is self-contained:
  * It redefines the model architecture and the preprocessing transforms,
    kept BYTE-FOR-BYTE consistent with the improved trainer.
  * It restores the energy-normalization constants (LOG_E_MEAN / LOG_E_STD)
    FROM THE CHECKPOINT. The improved trainer saves them under the keys
    "log_e_mean" / "log_e_std", so the standardized energy conditioning is
    guaranteed identical to training -- no need to recompute from data.

WHY THIS MATTERS
  energy_to_cond standardizes log10(E) using (mean, std). If those constants
  differ even slightly from training, every conditioned forward pass is wrong.
  Pulling them from the checkpoint removes that risk entirely.

USAGE
  python visualize_calo_ae.py
      --checkpoint best_calo_ae_cond_improved.pt
      --data       data/Dataset2.pt
      --outdir     viz_out
      [--relative]            # use relative residual |x_hat-x|/(x+eps)
      [--n-batches N]         # limit batches for a quick look
      [--device cuda|cpu]

If run with no arguments it uses the defaults above.

KEEPING THIS IN SYNC WITH THE TRAINER
  The model class and the three transform functions below are copied from the
  improved trainer. If you change the architecture or the transform there,
  copy the change here too. The checkpoint stores geometry + latent dims, so
  shape mismatches are caught loudly at load time rather than failing silently.
================================================================================
"""

import os
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")          # script-safe: never opens a window, never hangs
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

EPS = 1e-8

# These are restored from the checkpoint in main(). They are module-level
# because energy_to_cond() reads them -- exactly as in the trainer.
LOG_E_MEAN = None
LOG_E_STD = None


# ==============================================================================
# PREPROCESSING  --  copied verbatim from the improved trainer
# ==============================================================================

def physical_to_model(x, energy):
    """raw shower (GeV/voxel) -> model input  =  log1p(x / E_incident)."""
    assert x.dim() == 4, (
        f"physical_to_model: expected [B, Nz, Nphi, Nr], got {tuple(x.shape)}"
    )
    B = x.shape[0]
    assert energy.numel() == B, (
        f"physical_to_model: {energy.numel()} energies for {B} showers"
    )
    E = energy.reshape(-1, 1, 1, 1)
    x_log = torch.log1p(x / (E + 1e-8))
    assert x_log.shape == x.shape
    return x_log


def model_to_physical(x_log, energy):
    """model output -> raw shower. Clamps >= 0 (linear output may go slightly
    negative for empty voxels)."""
    assert x_log.dim() == 4, (
        f"model_to_physical: expected [B, Nz, Nphi, Nr], got {tuple(x_log.shape)}"
    )
    B = x_log.shape[0]
    assert energy.numel() == B
    E = energy.reshape(-1, 1, 1, 1)
    x_norm = torch.clamp(torch.expm1(x_log), min=0.0)
    return x_norm * E


def energy_to_cond(energy):
    """energy -> standardized log10(E), shape [B, 1].
    Uses module-level LOG_E_MEAN / LOG_E_STD restored from the checkpoint."""
    assert LOG_E_MEAN is not None and LOG_E_STD is not None, (
        "energy_to_cond: LOG_E_MEAN / LOG_E_STD not set. "
        "They must be restored from the checkpoint before any forward pass."
    )
    energy = energy.reshape(-1)
    log_e = torch.log10(energy + 1e-8)
    out = ((log_e - LOG_E_MEAN) / LOG_E_STD).unsqueeze(1)
    assert out.dim() == 2 and out.shape[1] == 1
    return out


# ==============================================================================
# MODEL  --  copied verbatim from the improved trainer
# ==============================================================================

class ResConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(h + self.shortcut(x))


class StridedDown2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


class UpBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super().__init__()
        self.size = size
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode='bilinear',
                          align_corners=False)
        return F.gelu(self.norm(self.conv(x)))


class EnergyEmbed(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
        )

    def forward(self, e_cond):
        return self.mlp(e_cond)


class FactorizedCaloAE(nn.Module):
    def __init__(self, Nz=45, Nphi=16, Nr=9,
                 latent_channels=8, latent_length=12, energy_embed_dim=64):
        super().__init__()
        self.Nz, self.Nphi, self.Nr = Nz, Nphi, Nr
        self.latent_channels = latent_channels
        self.latent_length = latent_length
        self.latent_dim = latent_channels * latent_length
        self.energy_embed_dim = energy_embed_dim
        self.energy_embed = EnergyEmbed(energy_embed_dim)

        self.enc2d = nn.Sequential(
            ResConvBlock2D(1, 16),
            StridedDown2D(16, 32),
            ResConvBlock2D(32, 32),
            StridedDown2D(32, 64),
            ResConvBlock2D(64, 64),
        )

        self.Henc = self.Wenc = self.transverse_features = None
        self.enc1d = self.fc_enc = self.fc_dec = None
        self.dec1d = self.dec2d = None
        self._build_lazy_modules()

    def _build_lazy_modules(self):
        with torch.no_grad():
            out = self.enc2d(torch.zeros(1, 1, self.Nphi, self.Nr))
        _, C, H, W = out.shape
        self.Henc, self.Wenc = H, W
        self.transverse_features = C * H * W
        print(f"[AE] transverse encoder output: C={C}, H={H}, W={W} "
              f"-> features={self.transverse_features}")

        self.enc1d = nn.Sequential(
            nn.Conv1d(self.transverse_features, 256, 3, padding=1),
            nn.GroupNorm(8, 256), nn.GELU(),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(),
        )
        self.fc_enc = nn.Linear(128 * self.Nz + self.energy_embed_dim,
                                self.latent_dim)
        self.fc_dec = nn.Linear(self.latent_dim + self.energy_embed_dim,
                                128 * self.Nz)
        self.dec1d = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.GroupNorm(8, 256), nn.GELU(),
            nn.Conv1d(256, self.transverse_features, 3, padding=1),
            nn.GroupNorm(8, self.transverse_features), nn.GELU(),
        )
        self.dec2d = nn.Sequential(
            ResConvBlock2D(64, 64),
            UpBlock2D(64, 32, size=(8, 4)),
            ResConvBlock2D(32, 32),
            UpBlock2D(32, 16, size=(self.Nphi, self.Nr)),
            ResConvBlock2D(16, 16),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),   # LINEAR output
        )

    def encode(self, x_log, energy):
        B = x_log.shape[0]
        x = x_log.reshape(B * self.Nz, 1, self.Nphi, self.Nr)
        x = self.enc2d(x)
        x = x.reshape(B, self.Nz, self.transverse_features)
        x = x.permute(0, 2, 1)
        x = self.enc1d(x)
        x = x.reshape(B, -1)
        e_emb = self.energy_embed(energy_to_cond(energy))
        x = torch.cat([x, e_emb], dim=1)
        z = self.fc_enc(x)
        return z.reshape(B, self.latent_channels, self.latent_length)

    def decode(self, z, energy):
        B = z.shape[0]
        z_flat = z.reshape(B, self.latent_dim)
        e_emb = self.energy_embed(energy_to_cond(energy))
        zc = torch.cat([z_flat, e_emb], dim=1)
        x = self.fc_dec(zc).reshape(B, 128, self.Nz)
        x = self.dec1d(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(B * self.Nz, 64, self.Henc, self.Wenc)
        x = self.dec2d(x)
        return x.reshape(B, self.Nz, self.Nphi, self.Nr)

    def forward(self, x_log, energy):
        z = self.encode(x_log, energy)
        x_hat_log = self.decode(z, energy)
        return x_hat_log, z


# ==============================================================================
# DATASET  --  copied verbatim from the improved trainer
# ==============================================================================

class CaloDataset(Dataset):
    def __init__(self, showers, energies):
        self.showers = showers.float()
        self.energies = energies.float()

    def __len__(self):
        return self.showers.shape[0]

    def __getitem__(self, idx):
        return self.showers[idx], self.energies[idx]


def build_val_loader(data_path, batch_size=256):
    """Rebuild the SAME validation split as training.

    The trainer uses random_split with a Generator seeded at 42 and a 90/10
    split. Reproducing that seed + ratio here gives the identical val set,
    so the evaluation is on data the model never trained on.
    """
    data = torch.load(data_path)
    energies = data['energies'] / 1000.0          # GeV
    showers = data['showers']

    dataset = CaloDataset(showers, energies)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)   # MUST match trainer
    _, val_dataset = random_split(dataset, [train_size, val_size],
                                  generator=generator)

    return DataLoader(val_dataset, batch_size=batch_size,
                      shuffle=False, num_workers=0, pin_memory=True)


# ==============================================================================
# RESIDUAL COMPUTATION
# ==============================================================================

@torch.no_grad()
def compute_mean_residual(model, dataloader, device,
                          relative=True, n_batches=None):
    """Mean true / reco / residual showers in PHYSICAL units (GeV/voxel)."""
    model.eval()
    true_sum = reco_sum = res_sum = None
    n_samples = 0

    for i, (x_raw, energy) in enumerate(dataloader):
        if n_batches is not None and i >= n_batches:
            break
        x_raw = x_raw.to(device)
        energy = energy.to(device)

        x_log = physical_to_model(x_raw, energy)
        x_hat_log, _ = model(x_log, energy)        # model takes RAW energy
        x_hat = model_to_physical(x_hat_log, energy)

        if relative:
            residual = torch.abs(x_hat - x_raw) / (x_raw + EPS)
        else:
            residual = torch.abs(x_hat - x_raw)

        if true_sum is None:
            true_sum = x_raw.sum(0)
            reco_sum = x_hat.sum(0)
            res_sum = residual.sum(0)
        else:
            true_sum += x_raw.sum(0)
            reco_sum += x_hat.sum(0)
            res_sum += residual.sum(0)
        n_samples += x_raw.shape[0]

    if n_samples == 0:
        raise RuntimeError("compute_mean_residual: no samples processed")

    return (
        (true_sum / n_samples).cpu().numpy(),
        (reco_sum / n_samples).cpu().numpy(),
        (res_sum / n_samples).cpu().numpy(),
    )


# ==============================================================================
# PLOTTING
# ==============================================================================

def _layer_grid(L, ncols=5, polar=False, figscale=2.0):
    nrows = int(np.ceil(L / ncols))
    subplot_kw = {'projection': 'polar'} if polar else {}
    fig, axes = plt.subplots(nrows, ncols, subplot_kw=subplot_kw,
                             figsize=(figscale * ncols, figscale * nrows),
                             constrained_layout=True)
    return fig, np.atleast_1d(axes).ravel()


def plot_shower_polar(data, n=0, P=0.3, ftsize=11,
                      cmap='viridis', title=None, filename=None):
    """Polar plot of every layer, SHARED color scale + colorbar."""
    shower = data[n]
    L, A, R = shower.shape
    vmax = max(float(shower.max()), EPS)
    norm = PowerNorm(gamma=P, vmin=0.0, vmax=vmax)

    theta_edges = np.linspace(0, 2 * np.pi, A + 1)
    r_edges = np.linspace(0.0, 1.0, R + 1)
    theta, r = np.meshgrid(theta_edges, r_edges, indexing='ij')

    fig, axes = _layer_grid(L, ncols=5, polar=True)
    pcm = None
    for ax, idx in zip(axes, range(L)):
        pcm = ax.pcolormesh(theta, r, shower[idx], shading='auto',
                            norm=norm, cmap=cmap)
        ax.set_title(f'Layer {idx}', pad=6, fontsize=ftsize)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, color='white', linewidth=0.4, alpha=0.5)
    for ax in axes[L:]:
        ax.set_visible(False)
    if pcm is not None:
        fig.colorbar(pcm, ax=axes.tolist(), shrink=0.6,
                     label='energy (GeV / voxel)')
    if title:
        fig.suptitle(title, fontsize=ftsize + 3)
    if filename:
        fig.savefig(filename, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return filename


def plot_layer_comparison(mean_true, mean_reco, mean_residual,
                          layers=(0, 5, 10, 15, 20, 25, 30, 40),
                          P=0.3, filename='layer_comparison_relative.png'):
    """rows = true / reco / residual; columns = selected layers."""
    layers = [l for l in layers if l < mean_true.shape[0]]
    ncols = len(layers)
    tv_max = max(float(mean_true.max()), float(mean_reco.max()), EPS)
    res_max = max(float(mean_residual.max()), EPS)
    sig_norm = PowerNorm(gamma=P, vmin=0.0, vmax=tv_max)
    res_norm = Normalize(vmin=0.0, vmax=res_max)

    fig, axes = plt.subplots(3, ncols, figsize=(2.0 * ncols, 6.2),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    rows = [('True', mean_true, sig_norm, 'viridis'),
            ('Reco', mean_reco, sig_norm, 'viridis'),
            ('|Residual|', mean_residual, res_norm, 'magma')]
    for r, (name, arr, norm, cmap) in enumerate(rows):
        last = None
        for c, l in enumerate(layers):
            ax = axes[r, c]
            last = ax.pcolormesh(arr[l].T, norm=norm, cmap=cmap)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect('equal')
            if r == 0:
                ax.set_title(f'Layer {l}', fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
        fig.colorbar(last, ax=axes[r, :].tolist(), shrink=0.8)
    fig.suptitle('Autoencoder reconstruction by layer', fontsize=13)
    fig.savefig(filename, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return filename


def plot_longitudinal_profile(mean_true, mean_reco,
                              filename='longitudinal_profile_relative.png'):
    """Energy per layer (summed over A, R): true vs reco, with ratio panel."""
    prof_true = mean_true.sum(axis=(1, 2))
    prof_reco = mean_reco.sum(axis=(1, 2))
    layers = np.arange(len(prof_true))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   constrained_layout=True)
    ax1.plot(layers, prof_true, 'o-', label='true', markersize=3)
    ax1.plot(layers, prof_reco, 's-', label='reco', markersize=3)
    ax1.set_ylabel('mean energy / layer (GeV)')
    ax1.legend(); ax1.grid(alpha=0.3)

    ratio = prof_reco / (prof_true + EPS)
    ax2.plot(layers, ratio, 'o-', color='crimson', markersize=3)
    ax2.axhline(1.0, color='k', lw=0.8, ls='--')
    ax2.set_ylabel('reco / true')
    ax2.set_xlabel('layer index (depth)')
    ax2.set_ylim(0.5, 1.5)
    ax2.grid(alpha=0.3)
    fig.suptitle('Longitudinal shower profile', fontsize=13)
    fig.savefig(filename, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return filename


# ==============================================================================
# CHECKPOINT LOADING
# ==============================================================================

def load_model_from_checkpoint(checkpoint_path, device):
    """Rebuild the model from the trainer's DICT checkpoint and restore the
    energy-normalization globals. Returns the model in eval mode."""
    global LOG_E_MEAN, LOG_E_STD

    ckpt = torch.load(checkpoint_path, map_location=device)

    # The improved trainer saves a dict. A bare state_dict would lack these
    # keys -- fail loudly with a clear message rather than crash obscurely.
    required = ["model_state", "log_e_mean", "log_e_std",
                "Nz", "Nphi", "Nr", "latent_channels", "latent_length"]
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' is missing keys {missing}. "
            f"This script expects the DICT checkpoint saved by the improved "
            f"trainer, not a bare state_dict."
        )

    # Restore energy standardization BEFORE building/using the model.
    LOG_E_MEAN = ckpt["log_e_mean"]
    LOG_E_STD = ckpt["log_e_std"]
    print(f"Restored energy stats: LOG_E_MEAN={LOG_E_MEAN:.4f}, "
          f"LOG_E_STD={LOG_E_STD:.4f}")

    model = FactorizedCaloAE(
        Nz=ckpt["Nz"], Nphi=ckpt["Nphi"], Nr=ckpt["Nr"],
        latent_channels=ckpt["latent_channels"],
        latent_length=ckpt["latent_length"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded model from epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f}")
    return model


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone visualization for the improved CaloAE.")
    parser.add_argument("--checkpoint", default="best_calo_ae_cond_improved.pt")
    parser.add_argument("--data", default="data/Dataset2.pt")
    parser.add_argument("--outdir", default="viz_out")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-batches", type=int, default=None,
                        help="limit batches for a quick look (default: all)")
    parser.add_argument("--relative", action="store_true",
                        help="use relative residual |x_hat-x|/(x+eps)")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    os.makedirs(args.outdir, exist_ok=True)

    model = load_model_from_checkpoint(args.checkpoint, device)
    val_loader = build_val_loader(args.data, batch_size=args.batch_size)

    mean_true, mean_reco, mean_res = compute_mean_residual(
        model, val_loader, device,
        relative=args.relative, n_batches=args.n_batches)

    res_label = "relative" if args.relative else "absolute"
    op = os.path.join
    plot_shower_polar(mean_true[None], title='Mean true shower',
                      filename=op(args.outdir, 'mean_true.png'))
    plot_shower_polar(mean_reco[None], title='Mean reconstructed shower',
                      filename=op(args.outdir, 'mean_reco.png'))
    plot_shower_polar(mean_res[None], P=1.0, cmap='magma',
                      title=f'Mean {res_label} residual',
                      filename=op(args.outdir, 'mean_residual_relative.png'))
    plot_layer_comparison(mean_true, mean_reco, mean_res,
                          filename=op(args.outdir, 'layer_comparison_relative.png'))
    plot_longitudinal_profile(mean_true, mean_reco,
                              filename=op(args.outdir,
                                          'longitudinal_profile_relative.png'))

    e_true = float(mean_true.sum())
    e_reco = float(mean_reco.sum())
    print("\n" + "=" * 60)
    print(f"Reconstruction diagnostics ({res_label} residual)")
    print("-" * 60)
    print(f"  residual min / max     : {mean_res.min():.4g} / {mean_res.max():.4g}")
    print(f"  residual mean          : {mean_res.mean():.4g}")
    print(f"  total energy true/reco : {e_true:.4g} / {e_reco:.4g}")
    print(f"  energy bias (reco/true): {e_reco / (e_true + EPS):.4f}")
    print(f"  worst layer (residual) : {mean_res.sum(axis=(1,2)).argmax()}")
    print(f"  figures written to     : {args.outdir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
