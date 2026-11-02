"""
ae_trainer.py -- the autoencoder training loop.

Takes DataLoaders as arguments; it never builds or loads a dataset itself.
That was already true of train_autoencoder(), and the only remaining coupling
to module-level state was the checkpoint save reading LOG_E_MEAN/LOG_E_STD --
now read from the model's own buffers.

objective="v5" additionally gates the reconstruction at `tau` rather than at
h > 0, because focal loss is not proper and 0.5 is not the F1-optimal cut.
"""

import math
import os

import torch
from tqdm.auto import tqdm

from ..data.calo_dataset import THRESHOLD_LOG, model_to_physical, physical_to_model
from ..models.ae_losses import (calorimeter_loss, calorimeter_loss_v4,
                                calorimeter_loss_v5, energy_error_phys,
                                relative_error_phys)


def last_checkpoint_path(checkpoint_path):
    root, ext = os.path.splitext(checkpoint_path)
    return f"{root}_last{ext or '.pt'}"


def make_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from 0 -> 1
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # Cosine decay from 1 -> 0 over the remaining epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==============================================================================
# 9. TRAINING LOOP
# ==============================================================================

THRESHOLD_LOG = float(torch.log1p(torch.tensor(1.515e-5)))   # 15.15 keV / 1 GeV


def _step_loss(model, x_log, energy, hit_weight, energy_weight,
               objective="v4", tau=0.5, lam=0.1):
    """One forward + loss. Returns (total, recon-ish, hit, energy, x_hat_log),
    where x_hat_log is the gated reconstruction the metrics are computed on.
    For v4 the 'recon' slot is rel2; for v5 it is the charge term.

    objective="v5" uses the focal hurdle loss (see calorimeter_loss_v5) and
    gates at `tau` rather than at h > 0, because focal loss is not proper and
    0.5 is not the F1-optimal cut. Tune tau with tune_threshold()."""
    if model.two_head:
        v, h, _ = model.forward_heads(x_log, energy)
        if objective == "v5":
            total, focal, charge, e2 = calorimeter_loss_v5(
                v, h, x_log, model.x_scale, THRESHOLD_LOG,
                lam=lam, energy_weight=energy_weight)
            recon, hit_term = charge, focal
            cut = math.log(tau / (1.0 - tau))        # logit(tau)
        else:
            total, recon, hit_term, e2 = calorimeter_loss_v4(
                v, h, x_log, model.x_scale, hit_weight=hit_weight,
                energy_weight=energy_weight)
            cut = 0.0
        with torch.no_grad():
            x_hat_log = torch.where(h > cut, (v.float() * model.x_scale).clamp(min=0.0),
                                    torch.zeros_like(v, dtype=torch.float32))
        return total, recon, hit_term, e2, x_hat_log
    x_hat_log, _ = model(x_log, energy)
    total, recon, hit, E = calorimeter_loss(
        x_hat_log, x_log, energy, hit_logit_scale=model.hit_logit_scale)
    return total, recon, hit, E, x_hat_log


def train_autoencoder(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=100,
    scheduler=None,
    grad_clip=1.0,
    use_amp=True,
    checkpoint_path="best_calo_ae_cond_improved.pt",
    start_epoch=0,
    best_val_loss=float("inf"),
    hit_weight=0.1,
    energy_weight=1.0,
    objective="v4",
    tau=0.5,
    lam=0.1,
):
    # AMP setup using the legacy torch.cuda.amp API (works on older PyTorch).
    # This path uses fp16 + GradScaler. Only enabled if CUDA is available
    # (AMP is a no-op on CPU and would just slow things down with overhead).
    use_amp = use_amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # start_epoch/best_val_loss default to a fresh run (0, inf). When resuming,
    # the caller passes the checkpoint's own epoch/val_loss so that a) the loop
    # doesn't repeat epochs already trained, and b) the "save best" comparison
    # below can't be fooled by a fresh inf into overwriting a better checkpoint
    # with a worse one on the very first epoch back.
    history = {
        "train_loss": [], "val_loss": [],
        "train_rel": [],  "val_rel": [],
        "train_E": [],    "val_E": [],
    }

    for epoch in range(start_epoch, epochs):

        # ----------------------------------------------------------------
        # TRAIN
        # ----------------------------------------------------------------
        model.train()
        train_loss = train_rel = train_E = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for x_raw, energy in pbar:
            x_raw = x_raw.to(device, non_blocking=True)
            energy = energy.to(device, non_blocking=True)

            # SINGLE preprocessing step (no double-log bug)
            x_log = physical_to_model(x_raw, energy)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, recon_loss, hit_loss, E_loss, x_hat_log = _step_loss(
                    model, x_log, energy, hit_weight, energy_weight,
                    objective=objective, tau=tau, lam=lam)

            # Scaled backward + unscale before clipping + scaler.step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                rel_err = relative_error_phys(x_log, x_hat_log)
                E_err = energy_error_phys(x_log, x_hat_log)

            train_loss += loss.item()
            train_rel += rel_err
            train_E += E_err

            pbar.set_postfix({
                "loss":  f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "hit":   f"{hit_loss.item():.4f}",
                "E":     f"{E_loss.item():.4f}",
                "rel":   f"{rel_err:.4f}",
            })

        train_loss /= len(train_loader)
        train_rel /= len(train_loader)
        train_E /= len(train_loader)

        # ----------------------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------------------
        model.eval()
        val_loss = val_rel = val_E = 0.0

        with torch.no_grad():
            for x_raw, energy in val_loader:
                x_raw = x_raw.to(device, non_blocking=True)
                energy = energy.to(device, non_blocking=True)

                x_log = physical_to_model(x_raw, energy)

                # Same objective as training. (Pre-v4 this call omitted
                # hit_logit_scale, so val_loss -- which picks the saved
                # checkpoint -- used scale 1.0 while training used ~11.6.)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss, recon_loss, hit_loss, E_loss, x_hat_log = _step_loss(
                        model, x_log, energy, hit_weight, energy_weight,
                        objective=objective, tau=tau, lam=lam)

                val_loss += loss.item()
                val_rel += relative_error_phys(x_log, x_hat_log)
                val_E += energy_error_phys(x_log, x_hat_log)

        val_loss /= len(val_loader)
        val_rel /= len(val_loader)
        val_E /= len(val_loader)

        # ----------------------------------------------------------------
        # Scheduler step (cosine-warmup is per-epoch, no metric needed)
        # ----------------------------------------------------------------
        if scheduler is not None:
            scheduler.step()

        # ----------------------------------------------------------------
        # History + report
        # ----------------------------------------------------------------
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rel"].append(train_rel)
        history["val_rel"].append(val_rel)
        history["train_E"].append(train_E)
        history["val_E"].append(val_E)

        lr = optimizer.param_groups[0]['lr']

        print("\n" + "=" * 70)
        print(f"Epoch {epoch+1:03d}/{epochs}")
        print("-" * 70)
        print(f"TRAIN | loss={train_loss:.5f} | rel={train_rel:.5f} | E={train_E:.5f}")
        print(f"VALID | loss={val_loss:.5f} | rel={val_rel:.5f} | E={val_E:.5f}")
        print(f"LR    = {lr:.3e}")

        # ----------------------------------------------------------------
        # Save best model.
        # Include EVERYTHING needed to resume / use the decoder for
        # diffusion: model weights, optimizer, scheduler state, epoch,
        # and the energy normalization stats.
        # ----------------------------------------------------------------
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
        ckpt = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict() if scheduler else None,
                "val_loss": val_loss,
                "log_e_mean": float(model.log_e_mean),
                "log_e_std": float(model.log_e_std),
                "Nz": model.Nz,
                "Nphi": model.Nphi,
                "Nr": model.Nr,
                "latent_channels": model.latent_channels,
                "latent_length": model.latent_length,
                # Explicit units marker -- checkpoints saved before this fix
                # were trained on showers left in MeV against energies in
                # GeV (see the units note at the top of this file). Anything
                # consuming a checkpoint (generate_caloeval_hdf5.py, the
                # visualize_*.py scripts) should check this before assuming
                # which convention applies; absence of this key means the
                # old (MeV-shower) convention.
                "shower_units_gev": True,
                "best_val_loss": best_val_loss,
                # v4 marker: two-head decoder + scaled I/O (x_scale is also a
                # buffer in model_state). Absent => pre-v4 single-output model.
                "two_head": model.two_head,
                "phi_circular": model.phi_circular,
                "width_mult": model.width_mult,
                "x_scale": float(model.x_scale),
                "mid_transverse": list(model._mid_size()),
                "hit_weight": hit_weight,
                "energy_weight": energy_weight,
        }
        if improved:
            torch.save(ckpt, checkpoint_path)
            print(f"✅ Saved new best model -> {checkpoint_path}")
        # Also save every epoch to <name>_last.pt, so --resume continues from
        # the latest epoch rather than falling back to the last *best* one
        # (which silently discarded every epoch since the last improvement).
        torch.save(ckpt, last_checkpoint_path(checkpoint_path))

        print("=" * 70)

    return history
