"""
train_calo_pinnde.py -- main entrypoint. Loads precomputed anchor
trajectories (generate_anchor_trajectories.py), builds PhysicsNN with the
same reference ensemble the anchors were solved against, and trains it
with the ramp-step Adam curriculum (networks_calo.train_adam).

Usage:
    python -m pinnde_calo.train_calo_pinnde --anchors pinnde_calo/anchors.pt \
        --checkpoint pinnde_calo/pinnde_calo.pt --epochs 200 --ramp-steps 4
"""

import argparse
import torch

from . import networks_calo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anchors", default="pinnde_calo/anchors.pt")
    p.add_argument("--checkpoint", default="pinnde_calo/pinnde_calo.pt")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200, help="Epochs per ramp stage.")
    p.add_argument("--ramp-steps", type=int, default=4)
    p.add_argument("--initial-frac", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-warmup-factor", type=float, default=0.3)
    p.add_argument("--n-colloc", type=int, default=4096,
                   help="Physics-residual collocation points per training step.")
    p.add_argument("--physics-weight", type=float, default=10.0)
    p.add_argument("--data-weight", type=float, default=1.0,
                   help="Weight on the anchor-data term MSE_u. Set 0 for pure-PINN "
                        "(physics-only) training -- the t=1 boundary condition is "
                        "hard-enforced by the ansatz regardless, so that mode is "
                        "well-posed. MSE_u is still logged as a diagnostic.")
    p.add_argument("--sigma0", type=float, default=1e-2)
    p.add_argument("--energy-kernel-scale", type=float, default=5.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"device={device}")

    networks_calo.set_seed(args.seed)

    anchors = torch.load(args.anchors, map_location="cpu")
    shower_dim = anchors["shower_dim"]
    print(f"Loaded {anchors['t_u'].shape[0]} anchor points, shower_dim={shower_dim}, "
          f"reference ensemble size={anchors['x0_ref'].shape[0]}")

    model = networks_calo.PhysicsNN(
        anchors["x0_ref"].to(device), anchors["e0_ref"].to(device),
        shower_dim=shower_dim, hidden=args.hidden, sigma0=args.sigma0,
        energy_kernel_scale=args.energy_kernel_scale, device=device,
    ).to(device)

    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"PhysicsNN.net parameters: {n_params:,}")

    model, history = networks_calo.train_adam(
        model, anchors,
        n_colloc=args.n_colloc, n_epochs=args.epochs, ramp_steps=args.ramp_steps,
        initial_frac=args.initial_frac, lr_target=args.lr,
        lr_warmup_factor=args.lr_warmup_factor, physics_weight=args.physics_weight,
        data_weight=args.data_weight,
        device=device, checkpoint_path=args.checkpoint,
    )

    torch.save({
        "model_state": model.state_dict(),
        "shower_dim": shower_dim,
        "hidden": args.hidden,
        "sigma0": args.sigma0,
        "energy_kernel_scale": args.energy_kernel_scale,
        "x0_ref": anchors["x0_ref"],
        "e0_ref": anchors["e0_ref"],
        "stats": anchors["stats"],
        "history": history,
    }, args.checkpoint)
    print(f"\nFinal checkpoint -> {args.checkpoint}")


if __name__ == "__main__":
    main()
