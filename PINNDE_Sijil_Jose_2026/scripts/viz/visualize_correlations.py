"""
visualize_correlations.py  --  STANDALONE voxel-correlation fidelity analysis
================================================================================

Mean-shower plots can look perfect while the autoencoder has destroyed the
voxel-to-voxel CORRELATION structure. A model that learns "mean shower + small
independent per-voxel noise" reproduces every mean exactly but is physically
wrong: real showers have strong spatial correlations (a hot voxel implies hot
neighbors; deep layers correlate with shallow ones).

This is especially important for the diffusion downstream: if the AE
reconstruction loses correlations, the decoder cannot express them either, and
no diffusion prior over the latent will put them back.

This script compares the correlation structure of TRUE vs RECONSTRUCTED showers.

Like the other viz scripts it is fully standalone: no trainer import, model and
transforms copied verbatim, energy-norm constants restored from the checkpoint.

FIGURES PRODUCED (written to --outdir, the SAME directory as the other scripts):
  1. corr_matrix.png          Full voxel correlation matrix: true / reco / diff.
  2. corr_scatter.png         True vs reco correlation, pair by pair.
  3. corr_layer_layer.png     Layer-to-layer (longitudinal) correlation.
  4. corr_radial_angular.png  Radial and angular profile correlations.
  5. corr_neighbor.png        Nearest-neighbor (adjacent-voxel) correlation.
  6. corr_by_ebin.png         Correlation fidelity per incident-energy bin.

All correlations are computed in PHYSICAL space (GeV/voxel), after the inverse
transform, because that is where correlations are physically meaningful.

USAGE
  python visualize_correlations.py
      --checkpoint best_calo_ae_cond_improved.pt
      --data       data/Dataset2.pt
      --outdir     latent_out
      [--max-samples N]   # showers used for correlation estimates (default 3000)
      [--max-voxels M]    # voxel subset for the full matrix (default 400)
      [--n-ebins K]       # energy bins for the fidelity-by-energy plot (default 5)
      [--device cuda|cpu]

WHY A VOXEL SUBSET FOR THE FULL MATRIX
  The detector has Nz*Nphi*Nr = 45*16*9 = 6480 voxels. A full 6480x6480
  correlation matrix is 42M entries and visually unreadable. For corr_matrix and
  corr_scatter we use the M highest-variance voxels (the ones that actually
  carry shower energy and where correlations are meaningful). The structured
  analyses (layer/radial/angular/neighbor) use ALL voxels via aggregation.

KEEPING THIS IN SYNC WITH THE TRAINER
  The model class and the three transforms are copied from the improved trainer.
  If the architecture or transform changes there, copy the change here.
================================================================================
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

EPS = 1e-8

LOG_E_MEAN = None
LOG_E_STD = None


# ==============================================================================
# PREPROCESSING  --  copied verbatim from the improved trainer
# ==============================================================================

def physical_to_model(x, energy):
    """raw shower (GeV/voxel) -> model input = log1p(x / E_incident)."""
    assert x.dim() == 4
    B = x.shape[0]
    assert energy.numel() == B
    E = energy.reshape(-1, 1, 1, 1)
    return torch.log1p(x / (E + 1e-8))


def model_to_physical(x_log, energy):
    """model output -> raw shower. Clamps >= 0 (linear output may go negative)."""
    assert x_log.dim() == 4
    B = x_log.shape[0]
    assert energy.numel() == B
    E = energy.reshape(-1, 1, 1, 1)
    return torch.clamp(torch.expm1(x_log), min=0.0) * E


def energy_to_cond(energy):
    """energy -> standardized log10(E), shape [B, 1]."""
    assert LOG_E_MEAN is not None and LOG_E_STD is not None, (
        "energy_to_cond: LOG_E_MEAN / LOG_E_STD not set "
        "(must be restored from the checkpoint first)."
    )
    energy = energy.reshape(-1)
    log_e = torch.log10(energy + 1e-8)
    return ((log_e - LOG_E_MEAN) / LOG_E_STD).unsqueeze(1)


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
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
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
    """Rebuild the SAME validation split as training (seed 42, 90/10)."""
    data = torch.load(data_path)
    energies = data['energies'] / 1000.0
    showers = data['showers']

    dataset = CaloDataset(showers, energies)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(dataset, [train_size, val_size],
                                  generator=generator)
    return DataLoader(val_dataset, batch_size=batch_size,
                      shuffle=False, num_workers=0, pin_memory=True)


# ==============================================================================
# CHECKPOINT LOADING
# ==============================================================================

def load_model_from_checkpoint(checkpoint_path, device):
    """Rebuild the model from the trainer's DICT checkpoint and restore the
    energy-normalization globals."""
    global LOG_E_MEAN, LOG_E_STD
    ckpt = torch.load(checkpoint_path, map_location=device)

    required = ["model_state", "log_e_mean", "log_e_std",
                "Nz", "Nphi", "Nr", "latent_channels", "latent_length"]
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' is missing keys {missing}. "
            f"Expected the DICT checkpoint saved by the improved trainer."
        )

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
# COLLECTION  --  gather true & reconstructed showers in PHYSICAL space
# ==============================================================================

@torch.no_grad()
def collect_showers(model, dataloader, device, max_samples=3000):
    """Run the model and return matched true / reco showers + energies.

    Returns
    -------
    X_true : [N, Nz, Nphi, Nr] true showers, physical units (numpy)
    X_reco : [N, Nz, Nphi, Nr] reconstructed showers, physical units (numpy)
    E      : [N] incident energies in GeV (numpy, 1-D)
    """
    model.eval()
    true_list, reco_list, e_list = [], [], []
    n = 0
    for x_raw, energy in dataloader:
        x_raw = x_raw.to(device)
        energy = energy.to(device)

        x_log = physical_to_model(x_raw, energy)
        x_hat_log, _ = model(x_log, energy)
        x_hat = model_to_physical(x_hat_log, energy)

        true_list.append(x_raw.cpu())
        reco_list.append(x_hat.cpu())
        e_list.append(energy.reshape(-1).cpu())     # force 1-D
        n += x_raw.shape[0]
        if n >= max_samples:
            break

    X_true = torch.cat(true_list, 0)[:max_samples].numpy()
    X_reco = torch.cat(reco_list, 0)[:max_samples].numpy()
    E = torch.cat(e_list, 0)[:max_samples].numpy()
    print(f"Collected {X_true.shape[0]} showers, geometry {X_true.shape[1:]}")
    return X_true, X_reco, E


# ==============================================================================
# CORRELATION HELPERS
# ==============================================================================

def corr_matrix(M):
    """Pearson correlation matrix of columns of M ([N_samples, N_features]).

    Constant columns (zero variance -> undefined correlation) are handled by
    setting their std to 1 so the result is 0 rather than NaN.
    """
    Mc = M - M.mean(0, keepdims=True)
    std = M.std(0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    Mn = Mc / std
    C = (Mn.T @ Mn) / M.shape[0]
    return np.clip(C, -1.0, 1.0)


def top_variance_voxels(X_flat, max_voxels):
    """Indices of the highest-variance voxels -- the ones carrying shower
    energy, where correlations are physically meaningful."""
    var = X_flat.var(0)
    return np.argsort(var)[::-1][:max_voxels]


# ==============================================================================
# PLOT 1: full voxel correlation matrix -- true / reco / difference
# ==============================================================================

def plot_corr_matrix(X_true, X_reco, max_voxels, outdir):
    """Side-by-side correlation matrices on the top-variance voxel subset."""
    N = X_true.shape[0]
    Xt = X_true.reshape(N, -1)
    Xr = X_reco.reshape(N, -1)

    sel = top_variance_voxels(Xt, max_voxels)
    Ct = corr_matrix(Xt[:, sel])
    Cr = corr_matrix(Xr[:, sel])
    diff = Cr - Ct

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2),
                             constrained_layout=True)
    im0 = axes[0].imshow(Ct, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('True voxel correlation')
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(Cr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Reco voxel correlation')
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    dmax = max(np.abs(diff).max(), EPS)
    im2 = axes[2].imshow(diff, cmap='PuOr', vmin=-dmax, vmax=dmax)
    axes[2].set_title('Difference (reco - true)')
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel('voxel (top-variance subset)')
        ax.set_ylabel('voxel (top-variance subset)')

    mae = np.abs(diff).mean()
    fig.suptitle(f'Voxel correlation matrix  --  mean |reco - true| = {mae:.4f}',
                 fontsize=13)
    fig.savefig(os.path.join(outdir, 'corr_matrix.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    return Ct, Cr


# ==============================================================================
# PLOT 2: true vs reco correlation, pair by pair
# ==============================================================================

def plot_corr_scatter(Ct, Cr, outdir):
    """Each point is one voxel pair: x = true correlation, y = reco
    correlation. On the diagonal => preserved. Collapsed toward y=0 => the
    classic 'autoencoder washes out correlations' failure mode."""
    iu = np.triu_indices_from(Ct, k=1)             # upper triangle, no diagonal
    ct = Ct[iu]
    cr = Cr[iu]

    fig, ax = plt.subplots(figsize=(6.5, 6.2), constrained_layout=True)
    ax.scatter(ct, cr, s=4, alpha=0.15, color='teal')
    lim = [-1.05, 1.05]
    ax.plot(lim, lim, 'k--', lw=1, label='perfect preservation')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('true correlation')
    ax.set_ylabel('reco correlation')

    r = np.corrcoef(ct, cr)[0, 1]
    # slope through origin: how much correlation magnitude is retained
    slope = float(np.sum(ct * cr) / (np.sum(ct * ct) + EPS))
    ax.set_title(f'Correlation preservation\n'
                 f'agreement r = {r:.3f}, retention slope = {slope:.3f}')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(outdir, 'corr_scatter.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"correlation preservation: agreement r = {r:.3f}, "
          f"retention slope = {slope:.3f}")
    return r, slope


# ==============================================================================
# PLOT 3: layer-to-layer (longitudinal) correlation
# ==============================================================================

def plot_layer_layer_corr(X_true, X_reco, outdir):
    """Correlation between per-layer total energies. Captures whether shower
    DEPTH development is correlated the same way in reco as in truth."""
    # per-shower, per-layer energy: sum over (phi, r)
    Et = X_true.sum(axis=(2, 3))                   # [N, Nz]
    Er = X_reco.sum(axis=(2, 3))

    Ct = corr_matrix(Et)
    Cr = corr_matrix(Er)
    diff = Cr - Ct

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             constrained_layout=True)
    im0 = axes[0].imshow(Ct, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('True layer-layer correlation')
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(Cr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Reco layer-layer correlation')
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    dmax = max(np.abs(diff).max(), EPS)
    im2 = axes[2].imshow(diff, cmap='PuOr', vmin=-dmax, vmax=dmax)
    axes[2].set_title('Difference (reco - true)')
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xlabel('layer index')
        ax.set_ylabel('layer index')

    mae = np.abs(diff).mean()
    fig.suptitle(f'Longitudinal (layer-to-layer) correlation  --  '
                 f'mean |reco - true| = {mae:.4f}', fontsize=13)
    fig.savefig(os.path.join(outdir, 'corr_layer_layer.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
# PLOT 4: radial and angular profile correlations
# ==============================================================================

def plot_radial_angular_corr(X_true, X_reco, outdir):
    """Correlation matrices of the radial profile and the angular profile.
    Tests whether transverse structure keeps its internal correlations."""
    # radial profile: sum over (z, phi) -> [N, Nr]
    Rt = X_true.sum(axis=(1, 2))
    Rr = X_reco.sum(axis=(1, 2))
    # angular profile: sum over (z, r) -> [N, Nphi]
    At = X_true.sum(axis=(1, 3))
    Ar = X_reco.sum(axis=(1, 3))

    CRt, CRr = corr_matrix(Rt), corr_matrix(Rr)
    CAt, CAr = corr_matrix(At), corr_matrix(Ar)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                             constrained_layout=True)
    blocks = [
        ('Radial true', CRt, 'RdBu_r', -1, 1),
        ('Radial reco', CRr, 'RdBu_r', -1, 1),
        ('Radial diff', CRr - CRt, 'PuOr', None, None),
        ('Angular true', CAt, 'RdBu_r', -1, 1),
        ('Angular reco', CAr, 'RdBu_r', -1, 1),
        ('Angular diff', CAr - CAt, 'PuOr', None, None),
    ]
    for ax, (name, C, cmap, vlo, vhi) in zip(axes.ravel(), blocks):
        if vlo is None:
            v = max(np.abs(C).max(), EPS)
            vlo, vhi = -v, v
        im = ax.imshow(C, cmap=cmap, vmin=vlo, vmax=vhi)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle('Radial and angular profile correlations', fontsize=13)
    fig.savefig(os.path.join(outdir, 'corr_radial_angular.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
# PLOT 5: nearest-neighbor (adjacent-voxel) correlation
# ==============================================================================

def _neighbor_correlation(X, axis):
    """Mean Pearson correlation between voxels adjacent along one axis.

    X is [N, Nz, Nphi, Nr]; axis is 1 (z), 2 (phi) or 3 (r). For phi we treat
    the dimension as periodic (the calorimeter wraps around in angle)."""
    N = X.shape[0]
    a = X
    b = np.roll(X, shift=-1, axis=axis)
    if axis != 2:
        # non-periodic axes: drop the wrapped-around last slice
        sl = [slice(None)] * 4
        sl[axis] = slice(0, X.shape[axis] - 1)
        a = X[tuple(sl)]
        b_sl = [slice(None)] * 4
        b_sl[axis] = slice(1, X.shape[axis])
        b = X[tuple(b_sl)]

    a = a.reshape(N, -1)
    b = b.reshape(N, -1)
    # per voxel-pair correlation across the N showers, then averaged
    ac = a - a.mean(0, keepdims=True)
    bc = b - b.mean(0, keepdims=True)
    num = (ac * bc).mean(0)
    den = a.std(0) * b.std(0)
    den = np.where(den < EPS, 1.0, den)
    valid = (a.std(0) > EPS) & (b.std(0) > EPS)
    corr = (num / den)[valid]
    return float(corr.mean()) if corr.size else 0.0


def plot_neighbor_corr(X_true, X_reco, outdir):
    """Bar chart: mean adjacent-voxel correlation along z / phi / r,
    true vs reco. Tests LOCAL spatial coherence directly -- the thing an
    over-smoothing or noise-adding autoencoder gets most obviously wrong."""
    axes_names = ['longitudinal (z)', 'angular (phi)', 'radial (r)']
    axis_ids = [1, 2, 3]

    true_vals = [_neighbor_correlation(X_true, ax) for ax in axis_ids]
    reco_vals = [_neighbor_correlation(X_reco, ax) for ax in axis_ids]

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    xs = np.arange(len(axes_names))
    w = 0.38
    ax.bar(xs - w/2, true_vals, width=w, label='true', color='steelblue')
    ax.bar(xs + w/2, reco_vals, width=w, label='reco', color='indianred')
    for x, t, r in zip(xs, true_vals, reco_vals):
        ax.text(x - w/2, t + 0.01, f'{t:.3f}', ha='center', fontsize=8)
        ax.text(x + w/2, r + 0.01, f'{r:.3f}', ha='center', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(axes_names)
    ax.set_ylabel('mean adjacent-voxel correlation')
    ax.set_title('Nearest-neighbor correlation: true vs reco')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.savefig(os.path.join(outdir, 'corr_neighbor.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    print("nearest-neighbor correlation (true -> reco):")
    for name, t, r in zip(axes_names, true_vals, reco_vals):
        print(f"  {name:18s}: {t:.3f} -> {r:.3f}")


# ==============================================================================
# PLOT 6: correlation fidelity per incident-energy bin
# ==============================================================================

def plot_corr_by_ebin(X_true, X_reco, E, n_ebins, max_voxels, outdir):
    """Does correlation preservation degrade at certain energies?

    For each energy bin, compute the true and reco voxel correlation matrices
    (on the top-variance subset) and report the agreement: Pearson r between
    the two sets of pairwise correlations, and the mean absolute difference."""
    E = np.asarray(E).reshape(-1)
    log_e = np.log10(E + EPS)
    edges = np.quantile(log_e, np.linspace(0, 1, n_ebins + 1))
    bin_idx = np.clip(np.digitize(log_e, edges[1:-1]), 0, n_ebins - 1)

    N = X_true.shape[0]
    Xt_all = X_true.reshape(N, -1)
    sel = top_variance_voxels(Xt_all, max_voxels)   # fixed subset across bins

    labels, agree_r, mae = [], [], []
    for b in range(n_ebins):
        m = bin_idx == b
        if m.sum() < 10:                            # too few to estimate corr
            continue
        Ct = corr_matrix(X_true.reshape(N, -1)[m][:, sel])
        Cr = corr_matrix(X_reco.reshape(N, -1)[m][:, sel])
        iu = np.triu_indices_from(Ct, k=1)
        agree_r.append(np.corrcoef(Ct[iu], Cr[iu])[0, 1])
        mae.append(np.abs(Cr[iu] - Ct[iu]).mean())
        labels.append(f'{10**edges[b]:.0f}-{10**edges[b+1]:.0f}\nGeV '
                       f'(n={m.sum()})')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   constrained_layout=True)
    xs = np.arange(len(labels))
    ax1.bar(xs, agree_r, color='seagreen')
    ax1.set_xticks(xs); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('correlation agreement r')
    ax1.set_title('True-vs-reco correlation agreement, per energy bin')
    for x, v in zip(xs, agree_r):
        ax1.text(x, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)

    ax2.bar(xs, mae, color='indianred')
    ax2.set_xticks(xs); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('mean |reco - true| correlation')
    ax2.set_title('Correlation error, per energy bin')
    for x, v in zip(xs, mae):
        ax2.text(x, v + max(mae) * 0.02, f'{v:.3f}', ha='center', fontsize=8)

    fig.suptitle('Correlation fidelity vs incident energy', fontsize=13)
    fig.savefig(os.path.join(outdir, 'corr_by_ebin.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Voxel-correlation fidelity analysis for the improved "
                    "CaloAE: are spatial correlations preserved in reco?")
    parser.add_argument("--checkpoint", default="best_calo_ae_cond_improved.pt")
    parser.add_argument("--data", default="/scratch/sjose/PINN/python_files/calosim/data_set_2/CaloChallenge/data/Dataset2.pt")
    parser.add_argument("--outdir", default="latent_out",
                        help="same directory as the other viz scripts")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=3000,
                        help="showers used for correlation estimates")
    parser.add_argument("--max-voxels", type=int, default=400,
                        help="top-variance voxel subset for the full matrix")
    parser.add_argument("--n-ebins", type=int, default=5,
                        help="energy bins for the fidelity-by-energy plot")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    model = load_model_from_checkpoint(args.checkpoint, device)
    val_loader = build_val_loader(args.data, batch_size=args.batch_size)

    X_true, X_reco, E = collect_showers(
        model, val_loader, device, max_samples=args.max_samples)

    # 1: full voxel correlation matrix
    Ct, Cr = plot_corr_matrix(X_true, X_reco, args.max_voxels, args.outdir)
    # 2: pair-by-pair scatter
    agree_r, slope = plot_corr_scatter(Ct, Cr, args.outdir)
    # 3: layer-to-layer
    plot_layer_layer_corr(X_true, X_reco, args.outdir)
    # 4: radial + angular
    plot_radial_angular_corr(X_true, X_reco, args.outdir)
    # 5: nearest neighbor
    plot_neighbor_corr(X_true, X_reco, args.outdir)
    # 6: per energy bin
    plot_corr_by_ebin(X_true, X_reco, E, args.n_ebins,
                      args.max_voxels, args.outdir)

    print("\n" + "=" * 60)
    print("Voxel-correlation fidelity summary")
    print("-" * 60)
    print(f"  showers analyzed         : {X_true.shape[0]}")
    print(f"  voxel subset (full matrix): {args.max_voxels}")
    print(f"  correlation agreement r  : {agree_r:.3f}  "
          f"(1 = perfectly preserved)")
    print(f"  correlation retention    : {slope:.3f}  "
          f"(<1 = correlations washed out)")
    if slope < 0.85:
        print("  NOTE: retention well below 1 -- the autoencoder is")
        print("        damping voxel correlations. Mean-shower plots will")
        print("        not reveal this; the diffusion decoder will inherit it.")
    print(f"  figures written to       : {args.outdir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
