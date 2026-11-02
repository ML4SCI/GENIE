#!/usr/bin/env python
"""
train_autoencoder.py -- train FactorizedCaloAE, the compression half of the
latent-diffusion path.

The model, losses and training loop live in src/calods2/; this file is the
entrypoint. It is the only place that loads a dataset.

    python models/1_latent_diffusion/train_autoencoder.py \
        --data data/Dataset2.pt --epochs 250 --phi-circular \
        --checkpoint-path checkpoints/best_calo_ae_v4.pt

    # the focal hurdle objective (see docs/NOVA_TRANSFER.md); A/B against v4
    python models/1_latent_diffusion/train_autoencoder.py \
        --loss v5 --focal-lambda 0.1 --tau 0.40 --phi-circular --epochs 250 \
        --checkpoint-path checkpoints/best_calo_ae_v5.pt
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from calods2.data.calo_dataset import (compute_x_scale, load_calo_data,
                                       make_loaders, model_to_physical,
                                       physical_to_model)
from calods2.models.ae_losses import relative_error_phys, tune_threshold
from calods2.models.autoencoder import FactorizedCaloAE
from calods2.training.ae_trainer import (_step_loss, last_checkpoint_path,
                                         make_cosine_warmup_scheduler,
                                         train_autoencoder)


def main():
    import argparse
    cli = argparse.ArgumentParser(description="Train FactorizedCaloAE")
    cli.add_argument("--epochs", type=int, default=100,
                     help="Total epochs (default 100). Use a small value for "
                          "a quick validation run, e.g. --epochs 25.")
    cli.add_argument("--checkpoint-path", default="best_calo_ae_cond_improved.pt",
                     help="Where to save the best checkpoint. Use a distinct "
                          "path for experimental runs so they don't clobber "
                          "the reference checkpoint other scripts default to.")
    cli.add_argument("--resume", action="store_true",
                     help="Resume from --checkpoint-path instead of training "
                          "from scratch. Off by default.")
    cli.add_argument("--latent-channels", type=int, default=8,
                     help="Latent channels. Latent size = channels x length (default 8x12=96). "
                          "PCA at the same size reaches rel 0.539 and PCA-256 0.461, so a "
                          "larger latent has measured headroom.")
    cli.add_argument("--latent-length", type=int, default=12,
                     help="Latent longitudinal positions (default 12).")
    cli.add_argument("--phi-circular", action="store_true",
                     help="Pad phi circularly in every transverse conv (it wraps around the "
                          "calorimeter). Off reproduces the original zero padding.")
    cli.add_argument("--width-mult", type=float, default=1.0,
                     help="Scale every conv channel count, to sweep capacity independently "
                          "of the latent size.")
    cli.add_argument("--loss", choices=["v4", "v5"], default="v4",
                     help="v4: rel^2 + BCE + energy (current default). "
                          "v5: focal hurdle loss adapted from the NOvA sparse-CAE "
                          "report -- fixes over-painting at TRAINING time rather "
                          "than by a post-hoc cut. See docs/NOVA_TRANSFER.md.")
    cli.add_argument("--focal-lambda", type=float, default=0.1,
                     help="v5 only: weight of the charge term. At 1.0 the charge "
                          "term is 83%% of the loss and the hit term is starved; "
                          "0.1 balances the three terms ~0.36/0.33/0.31.")
    cli.add_argument("--tau", type=float, default=0.5,
                     help="Occupancy threshold for the gated read-out. v4 used a "
                          "hard-coded 0.5; tune_threshold() picks the F1-optimal "
                          "value on validation (NOvA found 0.35-0.40).")
    cli.add_argument("--hit-weight", type=float, default=0.1,
                     help="Weight of the hit-head BCE (v4 objective).")
    cli.add_argument("--energy-weight", type=float, default=1.0,
                     help="Weight of the squared relative total-energy error (v4).")
    cli.add_argument("--data", default="data/Dataset2.pt",
                     help="Dataset2.pt, as written by the conversion in the README. "
                          "Both arrays must be in MeV.")
    cli.add_argument("--batch-size", type=int, default=None,
                     help="Training batch size (default 4096, or $CALO_BATCH_SIZE).")
    cli_args = cli.parse_args()

    # Explicit data loading -- this used to happen at import time.
    showers, energies, data_stats = load_calo_data(cli_args.data)
    Nz, Nphi, Nr = data_stats["geometry"]
    print(f"{data_stats['n_showers']} showers, geometry {data_stats['geometry']}")
    print(f"log10(E) mean = {data_stats['log_e_mean']:.4f}, "
          f"std = {data_stats['log_e_std']:.4f}")
    train_loader, val_loader, train_indices = make_loaders(
        showers, energies, batch_size=cli_args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Global std of the log-space target over (a fixed subset of) the TRAIN
    # split, ~8e-4. The model works on x_log / X_SCALE so its inputs and value
    # head are O(1) instead of ~1e-4. Saved as a buffer in the checkpoint, so
    # a resumed run keeps the value it was trained with.
    X_SCALE = compute_x_scale(showers, energies, train_indices)
    print(f"x_scale (global std of x_log, train subset) = {X_SCALE:.4e}")

    model = FactorizedCaloAE(
        Nz=Nz,
        Nphi=Nphi,
        Nr=Nr,
        latent_channels=cli_args.latent_channels,
        latent_length=cli_args.latent_length,
        energy_embed_dim=64,
        two_head=True,
        x_scale=X_SCALE,
        phi_circular=cli_args.phi_circular,
        width_mult=cli_args.width_mult,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    EPOCHS = cli_args.epochs
    scheduler = make_cosine_warmup_scheduler(
        optimizer, warmup_epochs=5, total_epochs=EPOCHS
    )

    # ------------------------------------------------------------------
    # Resume (opt-in only, --resume). A fresh run with no checkpoint on disk
    # must never crash on import -- that was the bug here: the load used to
    # be unconditional, so a clean checkout with no checkpoint yet would fail
    # immediately, and every run after that silently "resumed" (whether that
    # was intended or not) from a fresh Adam optimizer state with
    # best_val_loss reset to inf -- meaning the very first post-resume epoch
    # could overwrite a better checkpoint with a worse one just because inf
    # loses to anything.
    # ------------------------------------------------------------------
    RESUME_TRAINING = cli_args.resume
    checkpoint_path = cli_args.checkpoint_path
    start_epoch = 0
    best_val_loss = float("inf")

    if RESUME_TRAINING:
        if os.path.exists(checkpoint_path):
            # Prefer <name>_last.pt (saved every epoch) when it is newer than
            # the best checkpoint; best_val_loss always comes from the best one
            # so a resumed run never overwrites a better model with a worse one.
            best_ck = torch.load(checkpoint_path, map_location=device)
            if not best_ck.get("two_head", False):
                raise SystemExit(
                    f"{checkpoint_path} is a pre-v4 (single-output) checkpoint and "
                    f"cannot be resumed with the v4 two-head model/objective. "
                    f"Start a fresh run with a new --checkpoint-path.")
            resume_path = checkpoint_path
            checkpoint = best_ck
            last_path = last_checkpoint_path(checkpoint_path)
            if os.path.exists(last_path):
                last_ck = torch.load(last_path, map_location=device)
                if last_ck.get("epoch", 0) > best_ck.get("epoch", 0):
                    resume_path, checkpoint = last_path, last_ck
            print(f"Resuming from {resume_path} ...")
            model.load_state_dict(checkpoint["model_state"])
            if checkpoint.get("optim_state") is not None:
                optimizer.load_state_dict(checkpoint["optim_state"])
            if checkpoint.get("sched_state") is not None:
                scheduler.load_state_dict(checkpoint["sched_state"])
            start_epoch = checkpoint.get("epoch", 0)
            best_val_loss = best_ck.get("val_loss", float("inf"))
            print(f"Resumed at epoch {start_epoch}, "
                  f"best_val_loss={best_val_loss:.5f}")
        else:
            print(f"--resume given but {checkpoint_path} does not "
                  f"exist yet -- starting fresh instead.")
    else:
        print("Starting a fresh training run from random initialization "
              "(pass --resume to continue from --checkpoint-path).")
    # ------------------------------------------------------------------
    # SMOKE TEST: one fwd + bwd pass on a single batch.
    # Catches shape/loss bugs in seconds instead of after a full epoch.
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Running smoke test on one batch...")
    print("-" * 70)

    smoke_batch = next(iter(train_loader))
    x_raw_s, energy_s = smoke_batch
    x_raw_s = x_raw_s.to(device)
    energy_s = energy_s.to(device)

    # Dataset contract
    assert x_raw_s.dim() == 4, (
        f"dataset: shower batch should be [B, Nz, Nphi, Nr], "
        f"got {tuple(x_raw_s.shape)}"
    )
    assert x_raw_s.shape[1:] == (Nz, Nphi, Nr), (
        f"dataset: shower geometry {tuple(x_raw_s.shape[1:])} "
        f"!= expected {(Nz, Nphi, Nr)}"
    )
    assert energy_s.numel() == x_raw_s.shape[0], (
        f"dataset: {energy_s.numel()} energies for "
        f"{x_raw_s.shape[0]} showers"
    )
    assert (energy_s > 0).all(), "dataset: found non-positive energy values"
    assert (x_raw_s >= 0).all(), "dataset: found negative voxel energies"

    x_log_s = physical_to_model(x_raw_s, energy_s)
    loss_s, recon_s, hit_s, E_s, x_hat_s = _step_loss(
        model, x_log_s, energy_s, cli_args.hit_weight, cli_args.energy_weight)
    z_s = model.encode(x_log_s, energy_s)
    loss_s.backward()

    print(f"  input shower batch  : {tuple(x_raw_s.shape)}")
    print(f"  energy batch        : {tuple(energy_s.shape)}")
    print(f"  log-space input     : {tuple(x_log_s.shape)} "
          f"range [{x_log_s.min():.3f}, {x_log_s.max():.3f}]")
    print(f"  latent              : {tuple(z_s.shape)}")
    print(f"  reconstruction      : {tuple(x_hat_s.shape)}")
    print(f"  loss components     : total={loss_s.item():.4f}, "
          f"recon={recon_s.item():.4f}, hit={hit_s.item():.4f}, "
          f"E={E_s.item():.4f}")

    # Check that gradients actually flowed
    n_params_with_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_params_total = sum(1 for _ in model.parameters())
    print(f"  params with non-zero grad: {n_params_with_grad}/{n_params_total}")
    assert n_params_with_grad > 0, (
        "smoke test: no parameters received gradients - "
        "something is disconnected in the graph"
    )

    optimizer.zero_grad(set_to_none=True)
    print("Smoke test passed.\n" + "-" * 70 + "\n")

    history = train_autoencoder(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=EPOCHS,
        scheduler=scheduler,
        use_amp=True,
        checkpoint_path=checkpoint_path,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        hit_weight=cli_args.hit_weight,
        energy_weight=cli_args.energy_weight,
        objective=cli_args.loss,
        tau=cli_args.tau,
        lam=cli_args.focal_lambda,
    )

    # =======================================================================
    # 11. EVALUATION + VISUALIZATION
    # =======================================================================

    # Was hardcoded to "best_calo_ae_cond_improved.pt" regardless of
    # --checkpoint-path -- meant every run's final residual plots/stats were
    # silently computed from whatever the ORIGINAL reference checkpoint was,
    # not the model this run just trained. Caught because a units-convention
    # experiment (showers now in GeV, checkpoint_path=..._TEST.pt) would have
    # reloaded an old MeV-trained checkpoint's weights and run them against
    # the new GeV-scale val_loader -- another silent 1000x-style mismatch.
    #
    # Guarded because --epochs 0 is a legitimate "check my setup" invocation
    # (it loads the data, builds the model and runs the one-batch smoke test),
    # and in that case no checkpoint was ever written.
    if not os.path.exists(checkpoint_path):
        print(f"\nno checkpoint at {checkpoint_path} -- nothing was trained, so "
              f"skipping the evaluation and plots.")
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    @torch.no_grad()
    def compute_mean_residual(model, dataloader, device):
        """Computes mean true / reco / abs-residual showers in PHYSICAL units."""
        model.eval()

        residual_sum = true_sum = reco_sum = None
        n_samples = 0

        for x_raw, energy in dataloader:
            x_raw = x_raw.to(device)
            energy = energy.to(device)
            B = x_raw.shape[0]

            # Forward through the full pipeline
            x_log = physical_to_model(x_raw, energy)
            x_hat_log, _ = model(x_log, energy)
            x_hat = model_to_physical(x_hat_log, energy)

            # x_raw is already physical
            residual = torch.abs(x_hat - x_raw)

            if residual_sum is None:
                residual_sum = residual.sum(dim=0)
                true_sum = x_raw.sum(dim=0)
                reco_sum = x_hat.sum(dim=0)
            else:
                residual_sum += residual.sum(dim=0)
                true_sum += x_raw.sum(dim=0)
                reco_sum += x_hat.sum(dim=0)

            n_samples += B

        return (
            (true_sum / n_samples).cpu().numpy(),
            (reco_sum / n_samples).cpu().numpy(),
            (residual_sum / n_samples).cpu().numpy(),
        )

    def plot_shower_polar(data, n=0, P=0.3, ftsize=12,
                          filename='shower_polar_NNN.png'):
        filename = filename.replace('NNN', '%6.6d' % n)
        shower = data[n]
        L_, A_, R_ = shower.shape

        theta_edges = np.linspace(0, 2 * np.pi, A_ + 1)
        r_edges = np.linspace(0.0, 1.0, R_ + 1)
        theta, r = np.meshgrid(theta_edges, r_edges, indexing="ij")

        ncols = 5
        nrows = int(np.ceil(L_ / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            subplot_kw={'projection': 'polar'},
            figsize=(2.0 * ncols, 2.0 * nrows),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()

        for ax, idx in zip(axes, range(L_)):
            energies_layer = shower[idx]
            ax.pcolormesh(theta, r, energies_layer, shading="auto",
                          norm=PowerNorm(gamma=P), cmap="viridis")
            ax.set_title(f"Layer {idx}", pad=8, fontsize=ftsize)
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(-1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xticklabels([]); ax.set_yticklabels([])
            ax.tick_params(axis='both', which='both', length=0, labelsize=0)
            ax.set_thetagrids(np.degrees(theta_edges))
            ax.set_rgrids(r_edges)
            ax.grid(True, color="white", linewidth=0.6, alpha=0.7)

        plt.savefig(filename)
        plt.show()

    mean_true, mean_reco, mean_residual = compute_mean_residual(
        model, val_loader, device
    )

    plot_shower_polar(mean_true[None],     n=0, filename='mean_true.png')
    plot_shower_polar(mean_reco[None],     n=0, filename='mean_reco.png')
    plot_shower_polar(mean_residual[None], n=0, filename='mean_residual.png')

    print("Residual stats (physical, GeV/voxel):")
    print(f"  min  = {mean_residual.min():.6e}")
    print(f"  max  = {mean_residual.max():.6e}")
    print(f"  mean = {mean_residual.mean():.6e}")


if __name__ == "__main__":
    main()
