"""
train_latent.py -- latent PINNDE: the same reverse-time diffusion sampler,
run in the autoencoder's 96-dimensional latent space instead of the 6480
voxel space.

                 encode              sample                decode
    showers  ------------>  latents  ------>  latents  ------------>  showers
              (AE encoder)   (96-d)   (PINNDE)  (96-d)   (AE decoder)

Why this is worth doing on top of the direct voxel-space sampler:

  * No sparsity problem. Latents are dense, so the hit head -- needed in voxel
    space because a conditional mean fills empty voxels with spurious deposits
    -- is simply switched off. The AE decoder reproduces sparsity itself
    (measured: lit-voxel fraction 0.256 against a real 0.250).
  * No geometry to learn. The decoder already knows the (z, phi, r) structure.
  * The PINNDE formalism was validated at 1-3 dimensions and measurably fails
    at 6480 (see the README). 96 is far closer to the regime where it works.

And the cost, which is a hard ceiling rather than a tunable: whatever the
autoencoder cannot reconstruct, no latent generator can recover. Run with
--ceiling to measure that bound directly -- it encodes and decodes REAL
showers, with no diffusion model involved, and exports the result for the
same metrics. Any latent-sampler number should be read against that file,
not against real showers alone.

Usage:
    # the reconstruction ceiling (do this first)
    python models/1_latent_diffusion/train_latent.py --ae best_calo_ae_v4.pt --ceiling \
        --export generated/latent_ceiling.hdf5 --n-sample 20000

    # train the latent sampler, then sample + decode + export
    python models/1_latent_diffusion/train_latent.py --ae best_calo_ae_v4.pt \
        --checkpoint checkpoints/latent_velocity.pt --steps 20000 \
        --export generated/latent_recon_ds2.hdf5 --n-sample 100000

    python -m calods2.metrics.quick_metrics --gen generated/latent_recon_ds2.hdf5
"""

import argparse
import os
import sys

import torch

from calods2.data import calo_dataset, export
from calods2.models import autoencoder, velocity_net


def load_ae(path, device):
    """Load the trained autoencoder, frozen.

    This used to set sys.argv and CALO_BATCH_SIZE before importing the trainer,
    because importing it loaded Dataset 2 and built DataLoaders as a side
    effect. That is gone: the model is importable on its own, and the data is
    loaded explicitly by the caller.

    The transforms the model was trained with come from the checkpoint via
    from_checkpoint(), never re-derived here.
    """
    ae, ck = autoencoder.from_checkpoint(path, device=device, verbose=False)
    for p_ in ae.parameters():
        p_.requires_grad_(False)
    print(f"autoencoder {path}: epoch {ck.get('epoch')}, "
          f"{'v4 two-head' if ck.get('two_head') else 'pre-v4'}, "
          f"latent {ae.latent_channels}x{ae.latent_length} = {ae.latent_dim}")
    return ae, ck


@torch.no_grad()
def encode_all(ae, showers, energies, device, batch=256):
    """Dataset 2 -> standardized latents. Per-dimension standardization is used
    here (unlike the voxel path's single global scale): all 96 latent
    dimensions are active, so there are no dead coordinates to amplify."""
    zs = []
    n = showers.shape[0]
    for s in range(0, n, batch):
        x = showers[s:s + batch].float().to(device)
        e = energies[s:s + batch].float().to(device)
        zs.append(ae.encode(calo_dataset.physical_to_model(x, e), e).reshape(x.shape[0], -1).cpu())
        if s % (batch * 100) == 0:
            print(f"  encoding {s}/{n}", flush=True)
    z = torch.cat(zs)
    mean, std = z.mean(0), z.std(0)
    print(f"  latents {tuple(z.shape)}; per-dim std min {std.min():.4f} "
          f"max {std.max():.4f}")
    return (z - mean) / (std + 1e-8), mean, std


@torch.no_grad()
def decode_to_mev(ae, z_std, mean, std, energy_gev, device, batch=256):
    """Standardized latents -> physical showers in MeV, via the AE decoder."""
    out = []
    for s in range(0, z_std.shape[0], batch):
        z = (z_std[s:s + batch].cpu() * std + mean).to(device)
        e = energy_gev[s:s + batch].to(device)
        z = z.reshape(z.shape[0], ae.latent_channels, ae.latent_length)
        x_log = ae.decode(z, e)
        out.append((calo_dataset.model_to_physical(x_log, e) * 1000.0).cpu())
    return torch.cat(out).reshape(z_std.shape[0], -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae", default="best_calo_ae_v4.pt", help="trained autoencoder checkpoint")
    p.add_argument("--checkpoint", default="checkpoints/latent_velocity.pt")
    p.add_argument("--ceiling", action="store_true",
                   help="Encode+decode REAL showers and export that, with no diffusion "
                        "model: the reconstruction bound every latent sampler is read against.")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--cond", choices=["concat", "film"], default="film")
    p.add_argument("--sigma0", type=float, default=1e-2)
    p.add_argument("--export", default=None)
    p.add_argument("--n-sample", type=int, default=100000)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--sample-batch", type=int, default=2048)
    p.add_argument("--readout", choices=["raw", "denoise"], default="raw",
                   help="raw = the ODE end point (a sample); denoise = its conditional "
                        "mean, which shrinks diversity. No gate: latents are dense.")
    p.add_argument("--threshold-kev", type=float, default=15.15)
    p.add_argument("--data", default="data/Dataset2.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    ae, ae_ck = load_ae(args.ae, device)
    geom = {"Nz": ae.Nz, "Nphi": ae.Nphi, "Nr": ae.Nr}

    # Explicit data load -- importing the autoencoder no longer does it.
    showers, energies, _ = calo_dataset.load_calo_data(args.data)

    print("encoding Dataset 2 into latent space")
    z_all, z_mean, z_std = encode_all(ae, showers, energies, device)
    e_cond_all = ae.energy_to_cond(energies.float()).cpu()
    energy_gev_all = energies.float().reshape(-1, 1)

    # ---------------- reconstruction ceiling ----------------
    if args.ceiling:
        n = min(args.n_sample, z_all.shape[0])
        idx = torch.randperm(z_all.shape[0])[:n]
        print(f"decoding {n} REAL encoded showers (no diffusion model)")
        sh = decode_to_mev(ae, z_all[idx], z_mean, z_std, energy_gev_all[idx], device)
        if args.export:
            export.write_hdf5(args.export, sh, energy_gev_all[idx] * 1000.0, geom,
                              args.threshold_kev)
            print(f"check with:  python -m calods2.metrics.quick_metrics --gen {args.export}")
        r = sh.sum(1) / (energy_gev_all[idx].reshape(-1) * 1000.0)
        print(f"ceiling E_dep/E_inc median {r.median():.4f}   "
              f"lit fraction {(sh > args.threshold_kev / 1000.0).float().mean():.4f}")
        return

    # ---------------- train the latent field ----------------
    model = velocity_net.VelocityNet(
        shower_dim=z_all.shape[1], hidden=args.hidden, sigma0=args.sigma0,
        hit_head=False, cond=args.cond).to(device)
    print(f"latent VelocityNet: {sum(q.numel() for q in model.parameters()):,} parameters, "
          f"dim {z_all.shape[1]}, cond={args.cond}")

    stats = {"latent_mean": z_mean, "latent_std": z_std, "ae": args.ae, **geom}
    model, _ = velocity_net.train(
        model, z_all, e_cond_all, None, steps=args.steps, batch=args.batch, lr=args.lr,
        device=device, checkpoint_path=args.checkpoint, stats=stats, target="clean",
        ema_decay=args.ema)
    print(f"\ncheckpoint -> {args.checkpoint}  (model_state = EMA weights)")

    if not args.export:
        return

    # ---------------- sample, decode, export ----------------
    print(f"\nsampling {args.n_sample} latents ({args.sample_steps} Heun steps each)")
    idx = torch.randint(0, energy_gev_all.shape[0], (args.n_sample,))
    eg = energy_gev_all[idx]
    ec = e_cond_all[idx]
    chunks = []
    for s in range(0, args.n_sample, args.sample_batch):
        chunks.append(velocity_net.sample(
            model, ec[s:s + args.sample_batch].to(device), n_steps=args.sample_steps,
            device=device, readout=args.readout).cpu())
        if s % (args.sample_batch * 20) == 0:
            print(f"  {min(s + args.sample_batch, args.n_sample)}/{args.n_sample}", flush=True)
    z_gen = torch.cat(chunks)
    print(f"  generated latents: per-dim std {z_gen.std(0).mean():.4f} "
          f"(real standardized latents: 1.0)")

    sh = decode_to_mev(ae, z_gen, z_mean, z_std, eg, device)
    r = sh.sum(1) / (eg.reshape(-1) * 1000.0)
    print(f"E_dep/E_inc median {r.median():.4f}  mean {r.mean():.4f}   (real ~0.78)")
    print(f"lit fraction {(sh > args.threshold_kev / 1000.0).float().mean():.4f}   (real ~0.25)")
    export.write_hdf5(args.export, sh, eg * 1000.0, geom, args.threshold_kev)
    print(f"check with:  python -m calods2.metrics.quick_metrics --gen {args.export}")
    print("  (compare against the --ceiling file, not only against real showers)")


if __name__ == "__main__":
    main()
