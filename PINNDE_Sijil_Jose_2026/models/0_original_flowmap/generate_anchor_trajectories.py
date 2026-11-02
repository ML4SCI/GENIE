"""
generate_anchor_trajectories.py -- precompute the "exact" supervised anchor
data (t_u, z_u, e_u, x_u) that PhysicsNN's MSE_u data-loss term is fit to.

Mirrors the data-prep block at the top of the reference repo's
train_1d_GMM.py: pick a reference ensemble (x0, e0) for the Monte Carlo
q-field, numerically integrate the reverse-time ODE from a batch of
Gaussian starting points using flow_de.FlowDE's exact (but expensive) RK2
solver, and save the resulting (t, z, x(t)) trajectory points -- computed
once, up front, since it's far too expensive to redo at every training
step (that cost is exactly what the PINN is being trained to amortize
away).

Chunked over the trajectory batch because the qVectorField evaluated
inside FlowDE's RK2 loop needs an (N_batch, M, 6480) tensor at every step
-- with 6480 dimensions that's much larger per-trajectory than the
reference repo's 1-3 dimensional toy cases, so N is processed in modest
sub-batches to keep memory bounded regardless of the total anchor count
requested.

Usage:
    python -m pinnde_calo.generate_anchor_trajectories \
        --data ../data/Dataset2.pt --out anchors.pt \
        --n-traj 64 --n-ref 256 --n-steps 250 --time-stride 5
"""

import argparse
import torch

from . import flow_de
from . import gendata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/Dataset2.pt")
    p.add_argument("--out", default="pinnde_calo/anchors.pt")
    p.add_argument("--n-traj", type=int, default=64,
                   help="Number of reverse-time trajectories (starting z's) to solve exactly.")
    p.add_argument("--n-ref", type=int, default=256,
                   help="Size M of the Monte Carlo reference ensemble inside qVectorField. "
                        "Kept small relative to the toy case's M=12000 -- an (N_batch, M, 6480) "
                        "tensor gets large fast; this is the main lever for that memory/quality "
                        "tradeoff.")
    p.add_argument("--n-steps", type=int, default=250, help="RK2 steps from t=1 to t=0.")
    p.add_argument("--time-stride", type=int, default=5,
                   help="Keep every k-th RK2 step as an anchor point (all 250 steps for 64 "
                        "trajectories is already 16000 anchor points; stride thins this out).")
    p.add_argument("--chunk", type=int, default=16,
                   help="Trajectories solved per FlowDE.forward call -- keeps the (chunk, n_ref, "
                        "6480) intermediate tensors bounded regardless of --n-traj.")
    p.add_argument("--device", default="cpu",
                   help="Defaults to CPU: this is a one-time precompute step, not the repeated "
                        "training loop, so it doesn't need to compete for GPU memory with anything.")
    args = p.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"device={device}")

    x0_all, e_all, energy_gev_all, stats = gendata.load_dataset2(args.data, device="cpu")
    print(f"Loaded {x0_all.shape[0]} showers, shower_dim={x0_all.shape[1]}")

    x0_ref, e0_ref = flow_de.get_target_sample(x0_all, e_all, args.n_ref)
    x0_ref, e0_ref = x0_ref.to(device), e0_ref.to(device)

    # Conditioning energies for the trajectories to solve: drawn from the
    # real dataset's energy distribution, same as the reference showers,
    # so anchors cover the same energy range the model will be asked to
    # sample at.
    traj_idx = torch.randint(0, x0_all.shape[0], (args.n_traj,))
    e_traj = e_all[traj_idx].to(device)

    shower_dim = x0_all.shape[1]

    t_u_chunks, z_u_chunks, e_u_chunks, x_u_chunks = [], [], [], []

    for start in range(0, args.n_traj, args.chunk):
        end = min(start + args.chunk, args.n_traj)
        n_chunk = end - start
        print(f"Solving trajectories {start}:{end} of {args.n_traj} ...")

        z0 = flow_de.get_normal_sample((n_chunk, shower_dim), device=device)
        e_chunk = e_traj[start:end]

        flow = flow_de.FlowDE(x0_ref, e0_ref, T=args.n_steps, savepath=True, debug=True)
        path, times = flow(z0, e_chunk)

        for i in range(0, len(path), args.time_stride):
            x_t = path[i]                              # (n_chunk, shower_dim), on CPU already
            t_val = times[i]
            t_u_chunks.append(torch.full((n_chunk, 1), t_val))
            z_u_chunks.append(z0.cpu())
            e_u_chunks.append(e_chunk.cpu())
            x_u_chunks.append(x_t)

    anchors = {
        "t_u": torch.cat(t_u_chunks, dim=0),
        "z_u": torch.cat(z_u_chunks, dim=0),
        "e_u": torch.cat(e_u_chunks, dim=0),
        "x_u": torch.cat(x_u_chunks, dim=0),
        # Same reference ensemble the anchors were solved against -- the
        # PINN's own qVectorField (inside PhysicsNN) is built from this
        # exact (x0_ref, e0_ref) pair too, matching the reference repo's
        # PhysicsNN(x0, z) reusing the ensemble the numerical solve used.
        "x0_ref": x0_ref.cpu(),
        "e0_ref": e0_ref.cpu(),
        "stats": stats,
        "shower_dim": shower_dim,
    }
    n_anchor = anchors["t_u"].shape[0]
    print(f"\nTotal anchor points: {n_anchor}  (n_traj={args.n_traj}, "
          f"steps_kept={n_anchor // max(args.n_traj,1)})")

    torch.save(anchors, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
