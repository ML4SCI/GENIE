"""
quick_residual_check.py
========================
Standalone, inference-only: loads a checkpoint, runs it over a held-out
sample of Dataset2.pt, and saves mean true/reco/residual shower plots plus
summary stats. Built to correctly redo what train_auto_encoder_improved.py's
end-of-run plotting section does, without re-running training and without
that section's now-fixed hardcoded-checkpoint-path bug.

Usage:
    python quick_residual_check.py --checkpoint best_calo_ae_units_fixed_TEST.pt --n 4096
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnose_training_plateau import (
    FactorizedCaloAE, physical_to_model, model_to_physical,
    load_checkpoint_model, load_data, EPS,
)


def compute_mean_residual(model, showers, energies_gev, device, batch_size=64):
    model.eval()
    residual_sum = true_sum = reco_sum = None
    n = 0
    rel_errs = []
    with torch.no_grad():
        for i in range(0, showers.shape[0], batch_size):
            x_raw = showers[i:i+batch_size].to(device)
            energy = energies_gev[i:i+batch_size].to(device)
            x_log = physical_to_model(x_raw, energy)
            x_hat_log, _ = model(x_log, energy)
            x_hat = model_to_physical(x_hat_log, energy)
            residual = torch.abs(x_hat - x_raw)

            x_flat = x_raw.reshape(x_raw.shape[0], -1)
            xhat_flat = x_hat.reshape(x_hat.shape[0], -1)
            rel = (torch.norm(x_flat - xhat_flat, dim=1) / (torch.norm(x_flat, dim=1) + EPS))
            rel_errs.append(rel.cpu())

            if residual_sum is None:
                residual_sum = residual.sum(dim=0)
                true_sum = x_raw.sum(dim=0)
                reco_sum = x_hat.sum(dim=0)
            else:
                residual_sum += residual.sum(dim=0)
                true_sum += x_raw.sum(dim=0)
                reco_sum += x_hat.sum(dim=0)
            n += x_raw.shape[0]
    rel_errs = torch.cat(rel_errs)
    return (true_sum / n).cpu().numpy(), (reco_sum / n).cpu().numpy(), \
           (residual_sum / n).cpu().numpy(), rel_errs.numpy()


def plot_layer_profile(mean_true, mean_reco, filename, title):
    true_per_layer = mean_true.sum(axis=(1, 2))
    reco_per_layer = mean_reco.sum(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(true_per_layer, label="true (mean)", lw=2)
    ax.plot(reco_per_layer, label="reconstructed (mean)", lw=2, ls="--")
    ax.set_xlabel("Layer (z)")
    ax.set_ylabel("Mean energy per layer (GeV)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/Dataset2.pt")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", default=".")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    model, ckpt = load_checkpoint_model(args.checkpoint, device)
    units_gev = ckpt.get('shower_units_gev', False)
    print(f"Checkpoint: epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.5f}, "
          f"shower_units_gev={units_gev}")

    # load_data() returns showers in their raw MeV form (diagnose_training_
    # plateau.py's other tests convert units themselves per-test as needed).
    # This checkpoint's own "shower_units_gev" flag says whether IT expects
    # GeV-scale showers -- convert to match, or feeding 1000x-too-large MeV
    # inputs into a GeV-trained model produces garbage (caught this by
    # getting a suspicious "reconstructs ~0.4% of true energy" result below
    # before adding this conversion).
    showers, energies_gev = load_data(args.data, n=args.n, device="cpu")
    if units_gev:
        showers = showers / 1000.0
        print("Converted showers to GeV to match checkpoint's units convention.")
    mean_true, mean_reco, mean_residual, rel_errs = compute_mean_residual(
        model, showers, energies_gev, device
    )

    tag = args.checkpoint.replace("/", "_").replace(".pt", "")
    plot_layer_profile(mean_true, mean_reco,
                        f"{args.outdir}/quick_residual_layer_profile_{tag}.png",
                        f"Mean energy per layer -- {args.checkpoint}")

    print("\nResidual stats (physical units, same convention as the checkpoint):")
    print(f"  mean|residual| min/mean/max = {mean_residual.min():.6e} / "
          f"{mean_residual.mean():.6e} / {mean_residual.max():.6e}")
    print(f"  mean true shower sum   = {mean_true.sum():.6f}")
    print(f"  mean reco shower sum   = {mean_reco.sum():.6f}")
    print(f"\nPer-shower relative error over n={len(rel_errs)} held-out samples:")
    print(f"  mean   = {rel_errs.mean():.4f}")
    print(f"  median = {np.median(rel_errs):.4f}")
    print(f"  p10/p90 = {np.percentile(rel_errs,10):.4f} / {np.percentile(rel_errs,90):.4f}")
    print(f"  max    = {rel_errs.max():.4f}")
    print(f"  frac with rel_err > 1.0 (worse than predicting all-zero) = "
          f"{(rel_errs > 1.0).mean():.4f}")

    np.save(f"{args.outdir}/quick_residual_rel_errs_{args.checkpoint.replace('/', '_')}.npy", rel_errs)
    print(f"\nWrote {args.outdir}/quick_residual_layer_profile_{tag}.png")


if __name__ == "__main__":
    main()
