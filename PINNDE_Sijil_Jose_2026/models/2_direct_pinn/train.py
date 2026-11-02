"""
train_velocity.py -- train the q/velocity network and export CaloChallenge HDF5.

Replaces the anchor-trajectory + collocation pipeline entirely: no
generate_anchor_trajectories step, no anchors file, no collocation points.
Training data is sampled fresh every batch from the real dataset, and with
the default clean-x0 target there is no reference ensemble either -- the only
input is Dataset2.pt.

Usage:
    # train then immediately sample + export for evaluate.py
    python models/2_direct_pinn/train.py --data data/Dataset2.pt \
        --checkpoint checkpoints/velocity.pt \
        --export generated/pinnde_recon_ds2.hdf5 --n-sample 100000

Do not simply raise --steps: with the plain (unpreconditioned) net, samples
got measurably WORSE past ~8k steps at batch 32, and at constant LR they blew
up to inf by 16k.
"""

import argparse
import sys
import torch

from calods2.data import export
import calods2.data.dataset as gendata
from calods2.models import velocity_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/Dataset2.pt")
    p.add_argument("--checkpoint", default="checkpoints/velocity.pt")
    p.add_argument("--target", choices=["clean", "mc"], default="clean",
                   help="clean: regress the clean shower x0 (recommended -- no "
                        "reference ensemble, faster, far more diverse samples). "
                        "mc: regress the Monte-Carlo q-field (collapses toward "
                        "the mean; kept for comparison). See velocity_net.train.")
    p.add_argument("--n-ref", type=int, default=512,
                   help="Reference-ensemble size, only used with --target mc. "
                        "Raising it does not improve samples (measured).")
    p.add_argument("--representation", choices=["fraction", "threshold"], default="fraction",
                   help="fraction: log1p(E/E_inc) (default, and the AE pipeline's). "
                        "threshold: log1p(E/15.15 keV) -- log-scaled, which biases the "
                        "energy low (see gendata and the README table).")
    p.add_argument("--precond", action="store_true",
                   help="EDM skip-connection preconditioning (see VelocityNet.coeffs). "
                        "Measured HARMFUL here -- kept only for comparison.")
    p.add_argument("--steps", type=int, default=16000,
                   help="Total steps; the cosine LR schedule decays to 0 here, so this "
                        "sets the schedule, not just a stopping point.")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4, help="Peak LR (500-step warmup, cosine decay).")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--no-hit-head", action="store_true",
                   help="Disable the per-voxel hit head that gates the final read-out. "
                        "Without it ~77%% of generated voxels come out lit (real: 25%%).")
    p.add_argument("--lit-head", action="store_true",
                   help="Add a third output block predicting E[value | lit], trained only on "
                        "lit voxels, for use with --readout lit. See VelocityNet.")
    p.add_argument("--cond", choices=["concat", "film", "conv"], default="concat",
                   help="Network trunk/conditioning: concat (flat MLP, two extra inputs), "
                        "film (flat MLP + sinusoidal embedding -> per-layer scale/shift), "
                        "or conv (3D U-Net over (z,phi,r) + film). See VelocityNet.")
    p.add_argument("--hit-weight", type=float, default=0.1,
                   help="Weight of the hit-head BCE in the training loss.")
    p.add_argument("--sigma0", type=float, default=1e-2)
    p.add_argument("--resume", action="store_true",
                   help="Continue an interrupted run from its checkpoint "
                        "(same schedule, optimizer state restored).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    # optional sampling/export
    p.add_argument("--export", default=None, help="If set, write this HDF5 after training.")
    p.add_argument("--n-sample", type=int, default=100000)
    p.add_argument("--sample-steps", type=int, default=100,
                   help="Heun steps for the reverse-time integration at sampling time.")
    p.add_argument("--sample-batch", type=int, default=512)
    p.add_argument("--threshold-kev", type=float, default=15.15)
    p.add_argument("--readout", choices=["denoise", "raw", "lit"], default="denoise",
                   help="Values of the final sample: denoise = q_hat(0, x) (a conditional "
                        "mean, diluted by occupancy uncertainty); raw = the ODE end point "
                        "itself (already a sample). Occupancy comes from the hit head either "
                        "way. See velocity_net.sample.")
    p.add_argument("--value-correct", action="store_true",
                   help="Divide kept voxel values by the hit probability, i.e. use "
                        "E[value | lit] instead of the diluted conditional mean. "
                        "See VelocityNet.denoise.")
    p.add_argument("--gate", choices=["topk", "gumbel", "sample", "threshold"], default="topk",
                   help="How the hit head decides a voxel is lit: topk (default), sample\n"
                        "from its probability, or\n"
                        "threshold at 0.5 (biased sparse). See VelocityNet.denoise.")
    args = p.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    print(f"device={device}")

    x0_all, e_all, energy_gev, stats = gendata.load_dataset2(
        args.data, device="cpu", representation=args.representation)
    d = x0_all.shape[1]
    print(f"loaded {x0_all.shape[0]} showers, shower_dim={d}, representation={args.representation}")

    q_field = None
    if args.target == "mc":
        q_field = velocity_net.build_q_field(x0_all, e_all, args.n_ref, device=device,
                                              sigma0=args.sigma0)
    model = velocity_net.VelocityNet(shower_dim=d, hidden=args.hidden, sigma0=args.sigma0,
                                      precond=args.precond,
                                      hit_head=not args.no_hit_head,
                                      empty_value=gendata.empty_value(stats),
                                      cond=args.cond, lit_head=args.lit_head).to(device)
    print(f"VelocityNet parameters: {sum(p_.numel() for p_ in model.parameters()):,}"
          f"   target={args.target}   precond={model.precond}   "
          f"hit_head={model.hit_head}   cond={model.cond}   "
          f"empty_value={float(model.empty_value):.4f}")

    # returns the EMA copy, which is what gets sampled from
    model, _ = velocity_net.train(model, x0_all, e_all, q_field, steps=args.steps,
                                  batch=args.batch, lr=args.lr, device=device,
                                  checkpoint_path=args.checkpoint, stats=stats,
                                  target=args.target, ema_decay=args.ema,
                                  hit_weight=args.hit_weight, resume=args.resume)
    print(f"\ncheckpoint -> {args.checkpoint}  (model_state = EMA weights)")

    if not args.export:
        return

    sample_batch = velocity_net.max_sample_batch(model, args.sample_batch,
                                              explicit="--sample-batch" in sys.argv)
    if sample_batch != args.sample_batch:
        print(f"  sampling batch capped {args.sample_batch} -> {sample_batch} "
              f"(conv activations; see velocity_net.CONV_SAMPLE_BATCH)")
    print(f"\nsampling {args.n_sample} showers ({args.sample_steps} Heun steps each)")
    model.eval()
    idx = torch.randint(0, energy_gev.shape[0], (args.n_sample,))
    e_cond_all = e_all[idx]
    eg_all = energy_gev[idx]

    chunks = []
    for start in range(0, args.n_sample, sample_batch):
        end = min(start + sample_batch, args.n_sample)
        e_c = e_cond_all[start:end].to(device)
        chunks.append(velocity_net.sample(model, e_c, n_steps=args.sample_steps,
                                          device=device, gate=args.gate,
                                          value_correct=args.value_correct,
                                          readout=args.readout).cpu())
        if start % (sample_batch * 20) == 0:
            print(f"  {end}/{args.n_sample}")
    x0_std = torch.cat(chunks, dim=0)

    # Shared, failure-tolerant conversion: clamps to physically possible values
    # and reports how much it had to clamp, instead of asserting away a
    # finished run (a 100k-step run was lost exactly that way).
    recon_mev, diag = export.physical_showers_mev(x0_std, eg_all, stats)

    thr_mev = args.threshold_kev / 1000.0
    real_mev = (gendata.x0_to_physical_showers(
        gendata.unstandardize_x0(x0_all[:4096], stats),
        energy_gev[:4096], stats) * 1000.0)
    real_ratio = (real_mev.sum(dim=(1, 2, 3)) / (energy_gev[:4096].squeeze() * 1000.0))
    print(f"Output E_dep/E_inc median {diag['ratio_median']:.4f} mean {diag['ratio_mean']:.4f}"
          f"   (real: {real_ratio.median():.4f})")
    print(f"Output hit fraction (> {args.threshold_kev} keV) = "
          f"{(recon_mev > thr_mev).float().mean():.4f}"
          f"   (real: {(real_mev > thr_mev).float().mean():.4f})")

    export.write_hdf5(args.export, recon_mev, eg_all * 1000.0, stats, args.threshold_kev)
    print(f"check with:      python -m calods2.metrics.quick_metrics --gen {args.export}")
    print(f"evaluate with:   python scripts/cluster/run_full_evaluation.py --skip-generate "
          f"--gen-file {args.export}")


if __name__ == "__main__":
    main()
