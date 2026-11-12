"""
================================================================================
visualize_latent_structure.py  --  STANDALONE latent global-structure diagnostics
================================================================================

Focused on ONE question: how do latent vectors cluster and separate by
incident particle energy? (Global structure -- not per-dimension breakdown.)

The latent is flattened to 96-D vectors ([8 channels x 12 longitudinal] -> 96),
because every method here -- PCA, t-SNE, pairwise distance, separation score --
operates on whole vectors.

Like the other viz scripts, this is fully standalone: it does NOT import the
trainer. It redefines the model + transforms and restores the energy-norm
constants from the checkpoint.

FIGURES PRODUCED (written to --outdir):
  1. latent_pca.png            PCA scatter, colored by energy + by energy bin.
  2. latent_tsne.png           t-SNE scatter, colored by energy.
  3. latent_dist_vs_egap.png   Pairwise latent distance vs |log10(E) gap|.
  4. latent_separation.png     How cleanly energy bins separate (silhouette
                               score + within-vs-between bin distances).

USAGE
  python visualize_latent_structure.py
      --checkpoint best_calo_ae_cond_improved.pt
      --data       data/Dataset2.pt
      --outdir     latent_out
      [--max-samples N]   # cap latents collected (t-SNE is O(N^2); default 4000)
      [--n-ebins K]       # energy bins for coloring / separation (default 6)
      [--device cuda|cpu]

INTERPRETATION NOTE
  Because the decoder is ALSO conditioned on energy, the latent is NOT required
  to encode energy -- the decoder can read energy from its own conditioning
  path. So weak energy separation here is not necessarily a bug; it can mean the
  model correctly offloaded energy to conditioning and kept the latent for
  shape. What matters for downstream diffusion is that latents are not
  artificially clumped and vary smoothly -- which the distance-vs-gap plot
  shows directly.

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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

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
# LATENT COLLECTION
# ==============================================================================

@torch.no_grad()
def collect_latents(model, dataloader, device, max_samples=4000):
    """Encode validation showers; return flattened latents + incident energies.

    Returns
    -------
    Z_flat : [N, C_lat*L_lat] flattened latents (numpy)
    E      : [N] incident energies in GeV (numpy)
    """
    model.eval()
    Z_list, E_list = [], []
    n = 0
    for x_raw, energy in dataloader:
        x_raw = x_raw.to(device)
        energy = energy.to(device)
        x_log = physical_to_model(x_raw, energy)
        z = model.encode(x_log, energy)               # [B, C_lat, L_lat]
        Z_list.append(z.reshape(z.shape[0], -1).cpu())
        # Some datasets store energies as [B, 1] rather than [B]. Flatten to
        # [B] here so every downstream array (bin_idx, masks) stays 1-D.
        E_list.append(energy.reshape(-1).cpu())
        n += x_raw.shape[0]
        if n >= max_samples:
            break

    Z_flat = torch.cat(Z_list, dim=0)[:max_samples].numpy()
    E = torch.cat(E_list, dim=0)[:max_samples].numpy()
    assert E.ndim == 1, f"collect_latents: energy should be 1-D, got {E.shape}"
    print(f"Collected {Z_flat.shape[0]} latents, flat dim {Z_flat.shape[1]}")
    return Z_flat, E


def make_energy_bins(energy, n_ebins):
    """Quantile energy bins -> integer bin index per sample + readable labels."""
    energy = np.asarray(energy).reshape(-1)        # defensive: force 1-D
    log_e = np.log10(energy + EPS)
    edges = np.quantile(log_e, np.linspace(0, 1, n_ebins + 1))
    bin_idx = np.clip(np.digitize(log_e, edges[1:-1]), 0, n_ebins - 1)
    labels = [f'{10**edges[b]:.0f}-{10**edges[b+1]:.0f} GeV'
              for b in range(n_ebins)]
    return bin_idx, labels


# ==============================================================================
# PLOT 1: PCA scatter (continuous energy + discrete bins side by side)
# ==============================================================================

def plot_pca(Z_flat, energy, bin_idx, bin_labels, outdir):
    """PCA to 2D. Left panel: continuous energy gradient. Right panel: discrete
    energy bins -- so you can see clusters AND the smooth ordering at once."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(Z_flat)
    ev = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1={ev[0]:.3f}, PC2={ev[1]:.3f} "
          f"(total {ev.sum():.3f})")

    log_e = np.log10(energy + EPS)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   constrained_layout=True)

    sc = ax1.scatter(coords[:, 0], coords[:, 1], c=log_e, cmap='viridis',
                     s=12, alpha=0.7)
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.set_title('Continuous energy gradient')
    fig.colorbar(sc, ax=ax1, label='log10(E / GeV)')

    n_ebins = len(bin_labels)
    cmap = plt.get_cmap('viridis', n_ebins)
    for b in range(n_ebins):
        m = bin_idx == b
        ax2.scatter(coords[m, 0], coords[m, 1], color=cmap(b),
                    s=12, alpha=0.7, label=bin_labels[b])
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_title('Discrete energy bins')
    ax2.legend(fontsize=8, markerscale=1.5)

    fig.suptitle(f'Latent space (PCA, {ev.sum()*100:.0f}% variance), '
                 f'by incident energy', fontsize=13)
    fig.savefig(os.path.join(outdir, 'latent_pca.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    return coords


# ==============================================================================
# PLOT 2: t-SNE scatter colored by energy
# ==============================================================================

def plot_tsne(Z_flat, energy, outdir, seed=42):
    """t-SNE to 2D, colored by log10(energy). Reveals nonlinear clustering
    that PCA's linear projection can miss."""
    n = Z_flat.shape[0]
    perplexity = min(30, max(5, n // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity,
                init='pca', random_state=seed)
    coords = tsne.fit_transform(Z_flat)

    log_e = np.log10(energy + EPS)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=log_e, cmap='viridis',
                    s=12, alpha=0.7)
    ax.set_xlabel('t-SNE component 1')
    ax.set_ylabel('t-SNE component 2')
    ax.set_title('Latent space (t-SNE), colored by incident energy')
    fig.colorbar(sc, ax=ax, label='log10(E / GeV)')
    fig.savefig(os.path.join(outdir, 'latent_tsne.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    return coords


# ==============================================================================
# PLOT 3: pairwise latent distance vs energy gap
# ==============================================================================

def plot_distance_vs_energy_gap(Z_flat, energy, outdir, n_pairs=20000,
                                seed=42):
    """Random shower pairs: latent L2 distance vs |log10(E) gap|. A positive
    trend means energy is encoded smoothly -- key for diffusion interpolation."""
    rng = np.random.default_rng(seed)
    n = Z_flat.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    log_e = np.log10(energy + EPS)
    e_gap = np.abs(log_e[i] - log_e[j])
    z_dist = np.linalg.norm(Z_flat[i] - Z_flat[j], axis=1)

    n_bins = 20
    edges = np.linspace(e_gap.min(), e_gap.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)
    idx = np.clip(np.digitize(e_gap, edges[1:-1]), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            means[b] = z_dist[m].mean()
            stds[b] = z_dist[m].std()

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter(e_gap, z_dist, s=4, alpha=0.08, color='gray',
               label='shower pairs')
    ax.plot(centers, means, 'o-', color='crimson', label='binned mean')
    ax.fill_between(centers, means - stds, means + stds,
                    color='crimson', alpha=0.2, label='+/- 1 std')

    valid = np.isfinite(e_gap) & np.isfinite(z_dist)
    r = np.corrcoef(e_gap[valid], z_dist[valid])[0, 1]
    ax.set_xlabel('|log10(E) gap| between two showers')
    ax.set_ylabel('latent L2 distance')
    ax.set_title(f'Latent distance vs energy gap  (Pearson r = {r:.3f})')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(outdir, 'latent_dist_vs_egap.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"latent-distance vs energy-gap correlation: r = {r:.3f}")
    return r


# ==============================================================================
# PLOT 4: quantified separation of energy bins
# ==============================================================================

def plot_separation(Z_flat, energy, bin_idx, bin_labels, outdir):
    """Quantify how cleanly energy bins separate, instead of eyeballing scatter.

    Left: mean within-bin vs between-bin latent distance, per bin. If between
          >> within, that bin is well separated from the others.
    Right: a single silhouette score (computed on standardized latents) -- near
           1 means tight, well-separated bins; near 0 means overlapping.
    """
    n_ebins = len(bin_labels)

    # standardize so no single high-variance dim dominates the distances
    Zc = (Z_flat - Z_flat.mean(0)) / (Z_flat.std(0) + EPS)
    centroids = np.stack([Zc[bin_idx == b].mean(0) for b in range(n_ebins)])

    within = np.zeros(n_ebins)
    between = np.zeros(n_ebins)
    for b in range(n_ebins):
        m = bin_idx == b
        within[b] = np.linalg.norm(Zc[m] - centroids[b], axis=1).mean()
        other = [np.linalg.norm(centroids[b] - centroids[o])
                 for o in range(n_ebins) if o != b]
        between[b] = np.mean(other)

    try:
        sil = silhouette_score(Zc, bin_idx)
    except Exception as exc:                       # e.g. a bin with 1 sample
        sil = float('nan')
        print(f"silhouette score could not be computed: {exc}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   constrained_layout=True,
                                   gridspec_kw={'width_ratios': [2, 1]})

    xs = np.arange(n_ebins)
    w = 0.38
    ax1.bar(xs - w/2, within, width=w, label='within-bin spread',
            color='slategray')
    ax1.bar(xs + w/2, between, width=w, label='between-bin distance',
            color='indianred')
    ax1.set_xticks(xs)
    ax1.set_xticklabels(bin_labels, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('latent distance (standardized)')
    ax1.set_title('Within-bin spread vs between-bin separation')
    ax1.legend(fontsize=9)

    ax2.bar([0], [sil], width=0.5, color='seagreen')
    ax2.set_xticks([0])
    ax2.set_xticklabels(['all bins'])
    ax2.set_ylim(-0.2, 1.0)
    ax2.axhline(0, color='k', lw=0.6)
    ax2.set_ylabel('silhouette score')
    ax2.set_title(f'Energy-bin silhouette = {sil:.3f}')
    ax2.text(0, sil + 0.04 if np.isfinite(sil) else 0.04,
             f'{sil:.3f}', ha='center', fontsize=11)

    fig.suptitle('How cleanly do energy bins separate in latent space?',
                 fontsize=13)
    fig.savefig(os.path.join(outdir, 'latent_separation.png'),
                dpi=140, bbox_inches='tight')
    plt.close(fig)
    return sil, within, between


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Latent global-structure diagnostics for the improved "
                    "CaloAE (clustering / separation by energy).")
    parser.add_argument("--checkpoint", default="best_calo_ae_cond_improved.pt")
    parser.add_argument("--data", default="data/Dataset2.pt")
    parser.add_argument("--outdir", default="latent_out")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=4000,
                        help="cap on latents collected (t-SNE is O(N^2))")
    parser.add_argument("--n-ebins", type=int, default=6,
                        help="energy bins for coloring / separation")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    model = load_model_from_checkpoint(args.checkpoint, device)
    val_loader = build_val_loader(args.data, batch_size=args.batch_size)

    Z_flat, E = collect_latents(
        model, val_loader, device, max_samples=args.max_samples)
    bin_idx, bin_labels = make_energy_bins(E, args.n_ebins)

    # global structure: clustering + separation by energy
    plot_pca(Z_flat, E, bin_idx, bin_labels, args.outdir)
    plot_tsne(Z_flat, E, args.outdir)
    r = plot_distance_vs_energy_gap(Z_flat, E, args.outdir)
    sil, within, between = plot_separation(
        Z_flat, E, bin_idx, bin_labels, args.outdir)

    print("\n" + "=" * 60)
    print("Latent global-structure summary")
    print("-" * 60)
    print(f"  samples analyzed         : {Z_flat.shape[0]}")
    print(f"  latent dim (flattened)   : {Z_flat.shape[1]}")
    print(f"  energy bins              : {args.n_ebins}")
    print(f"  distance vs energy-gap r : {r:.3f}")
    print(f"  energy-bin silhouette    : {sil:.3f}")
    sep_ratio = float(np.mean(between) / (np.mean(within) + EPS))
    print(f"  mean between/within ratio: {sep_ratio:.2f}  "
          f"(>1 = bins separated)")
    print(f"  figures written to       : {args.outdir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
