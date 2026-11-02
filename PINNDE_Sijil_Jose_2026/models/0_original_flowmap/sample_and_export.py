"""
sample_and_export.py -- generate showers with a trained PINNDE model and
write them in the exact CaloChallenge HDF5 layout generate_caloeval_hdf5.py
uses, so the output plugs directly into evaluate.py / run_full_evaluation.py
/ build_comparison_plots.py without any changes to that pipeline.

This is the payoff of the whole PINN construction: sampling a shower is
ONE forward pass through x_function at t=0 (a network evaluation costing
microseconds), not a multi-step numerical ODE solve through the expensive
Monte Carlo qVectorField the way flow_de.FlowDE (used only to build the
training anchors) needs.

Usage:
    python -m pinnde_calo.sample_and_export \
        --checkpoint pinnde_calo/pinnde_calo.pt \
        --data data/Dataset2.pt --n 100000 \
        --out pinnde_calo/pinnde_recon_ds2.hdf5
"""

import argparse
import numpy as np
import torch
import h5py

from . import networks_calo, gendata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="pinnde_calo/pinnde_calo.pt")
    p.add_argument("--data", default="data/Dataset2.pt",
                   help="Used only to draw the incident-energy distribution to condition on, "
                        "matching how the official reference file's energies are distributed.")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--threshold-kev", type=float, default=15.15)
    p.add_argument("--out", default="pinnde_calo/pinnde_recon_ds2.hdf5")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"device={device}")
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stats = ckpt["stats"]
    shower_dim = ckpt["shower_dim"]

    model = networks_calo.PhysicsNN(
        ckpt["x0_ref"].to(device), ckpt["e0_ref"].to(device),
        shower_dim=shower_dim, hidden=ckpt["hidden"], sigma0=ckpt["sigma0"],
        energy_kernel_scale=ckpt["energy_kernel_scale"], device=device,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Draw incident energies to condition on from the real dataset's own
    # distribution -- same convention the official evaluation pipeline
    # expects (a model that only ever gets tested at a handful of fixed
    # energies isn't comparable to submissions evaluated across the full
    # spectrum).
    _, _, energy_gev_all, _ = gendata.load_dataset2(args.data, device="cpu")
    idx = torch.randint(0, energy_gev_all.shape[0], (args.n,))
    energy_gev = energy_gev_all[idx]

    recon_gev_chunks = []
    with torch.no_grad():
        for start in range(0, args.n, args.batch_size):
            end = min(start + args.batch_size, args.n)
            b = end - start
            z = torch.randn(b, shower_dim, device=device)
            e_gev_b = energy_gev[start:end].to(device)
            e_cond_b = gendata.energy_gev_to_cond(e_gev_b, stats)
            t0 = torch.zeros(b, 1, device=device)

            x0_std = networks_calo.x_function(model, t0, z, e_cond_b)  # ONE forward pass
            # The model works in standardized space (see load_dataset2) --
            # undo that before exponentiating, or every shower comes out
            # scaled by ~1/0.0008 and expm1 turns it into garbage.
            x0_log = gendata.unstandardize_x0(x0_std, stats)
            showers_gev = gendata.x0_to_physical_showers(x0_log, e_gev_b, stats)
            recon_gev_chunks.append(showers_gev.cpu())

            if start % (args.batch_size * 20) == 0:
                print(f"  sampled {end}/{args.n}")

    recon_gev = torch.cat(recon_gev_chunks, dim=0).numpy().astype(np.float32)
    recon_mev = recon_gev * 1000.0
    einc_mev = (energy_gev * 1000.0).numpy().astype(np.float32)

    ratio_out = (recon_mev.sum(axis=(1, 2, 3)) / einc_mev.squeeze()).mean()
    print(f"Output E_dep/E_inc = {ratio_out:.4f}")

    if args.threshold_kev > 0:
        thr_mev = args.threshold_kev / 1000.0
        n_before = int((recon_mev > 0).sum())
        recon_mev = np.where(recon_mev < thr_mev, 0.0, recon_mev)
        n_after = int((recon_mev > 0).sum())
        print(f"Applied {args.threshold_kev} keV threshold: {n_before:,} -> {n_after:,} non-zero voxels")

    N = recon_mev.shape[0]
    recon_flat = recon_mev.reshape(N, -1)
    einc_flat = einc_mev.reshape(N, -1)

    assert recon_flat.shape[1] == stats["Nz"] * stats["Nphi"] * stats["Nr"]
    assert (recon_flat >= 0).all(), "Negative voxel energies in output!"
    assert np.isfinite(recon_flat).all(), "Non-finite values in output!"

    print(f"Output shapes: incident_energies {einc_flat.shape}, showers {recon_flat.shape}")

    with h5py.File(args.out, "w") as f:
        f.create_dataset("incident_energies", data=einc_flat, compression="gzip")
        f.create_dataset("showers", data=recon_flat, compression="gzip")
    print(f"Wrote {args.out}")
    print(f"Run: bash run_calo_eval.sh {args.out} <reference_geant4.hdf5>  "
          f"(or scripts/cluster/run_full_evaluation.py --gen-file {args.out} --skip-generate)")


if __name__ == "__main__":
    main()
