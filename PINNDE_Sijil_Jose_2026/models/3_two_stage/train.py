"""
train_two_stage.py -- the factorized sampler: energy first, then shape.

    stage A   p(u | E_inc)            45 numbers: total energy + how it is
                                      distributed along z, as bounded ratios
    stage B   p(w | E_inc, u)         6480 numbers: each voxel's share of its
                                      own layer, every layer summing to 1

Both stages are the same VelocityNet machinery and the same reverse-time ODE;
only the variables differ. Sampling draws u, then draws w conditioned on it,
then combines them analytically (calods2.data.layerdata.decode).

Why factorize, in terms of what we measured on the single-stage model:

  * Energy conservation stops being something the network has to learn. The
    single-stage model carried a persistent ~10% energy deficit that survived
    every read-out, gate and representation change; here the total is one of
    45 numbers a small network models directly, and the shape model cannot
    affect it at all.
  * The shape model no longer has to encode layer totals, so the per-voxel
    logit representation is usable -- and that is what addresses the defect
    the classifier keys on (dim voxels 12x too rare, 5th-percentile lit voxel
    3.4 MeV against a real 0.63).

This is the factorization used by CaloDREAM (arXiv:2405.09629), the best
CaloChallenge dataset-2 submission.

Usage:
    python models/3_two_stage/train.py --data data/Dataset2.pt \
        --checkpoint-energy src/calods2/two_stage_energy.pt \
        --checkpoint-shape  src/calods2/two_stage_shape.pt \
        --steps-energy 20000 --steps-shape 100000 --cond conv \
        --export generated/two_stage_ds2.hdf5 --n-sample 100000
"""

import argparse

import torch

from calods2.data import export
from calods2.data import layerdata
from calods2.models import velocity_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/Dataset2.pt")
    p.add_argument("--checkpoint-energy", default="src/calods2/two_stage_energy.pt")
    p.add_argument("--checkpoint-shape", default="src/calods2/two_stage_shape.pt")
    p.add_argument("--steps-energy", type=int, default=20000)
    p.add_argument("--steps-shape", type=int, default=100000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--hidden-energy", type=int, default=512)
    p.add_argument("--hidden-shape", type=int, default=1024)
    p.add_argument("--cond", choices=["concat", "film", "conv"], default="conv",
                   help="trunk for the SHAPE model; the energy model is always a film MLP")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--sigma0", type=float, default=1e-2)
    p.add_argument("--skip-energy", action="store_true", help="reuse an existing energy checkpoint")
    p.add_argument("--skip-shape", action="store_true", help="reuse an existing shape checkpoint")
    p.add_argument("--export", default=None)
    p.add_argument("--n-sample", type=int, default=100000)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--sample-batch", type=int, default=512)
    p.add_argument("--threshold-kev", type=float, default=15.15)
    p.add_argument("--resume", action="store_true",
                   help="Continue an interrupted run from its checkpoint "
                        "(same schedule, optimizer state restored).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)

    data = torch.load(args.data, map_location="cpu")
    showers = data["showers"].float()
    einc = data["energies"].float().reshape(-1, 1)
    del data
    print(f"loaded {showers.shape[0]} showers")

    u, w, stats = layerdata.encode(showers, einc)
    del showers
    log_e = torch.log10(einc / 1000.0 + 1e-8)
    stats["log_e_mean"], stats["log_e_std"] = float(log_e.mean()), float(log_e.std())
    e_cond = (log_e - stats["log_e_mean"]) / stats["log_e_std"]
    print(f"u {tuple(u.shape)}  w {tuple(w.shape)}  empty voxel -> "
          f"{layerdata.empty_value(stats):.4f}  f_scale {stats['f_scale']:.3f}")

    # ---------------- stage A: energy ----------------
    e_model = velocity_net.VelocityNet(
        shower_dim=u.shape[1], hidden=args.hidden_energy, sigma0=args.sigma0,
        hit_head=False, cond="film").to(device)
    if args.skip_energy:
        ck = torch.load(args.checkpoint_energy, map_location=device, weights_only=False)
        e_model.load_state_dict(ck["model_state"])
        print(f"stage A: reusing {args.checkpoint_energy}")
    else:
        print(f"\nstage A: energy, {sum(q.numel() for q in e_model.parameters()):,} params")
        e_model, _ = velocity_net.train(
            e_model, u, e_cond, None, steps=args.steps_energy, batch=256, lr=args.lr,
            device=device, checkpoint_path=args.checkpoint_energy, stats=stats,
            resume=args.resume)

    # ---------------- stage B: shape ----------------
    # conditioned on the incident energy AND the 45 energy numbers
    cond_shape = torch.cat([e_cond, u], dim=1)
    s_model = velocity_net.VelocityNet(
        shower_dim=w.shape[1], hidden=args.hidden_shape, sigma0=args.sigma0,
        hit_head=True, empty_value=layerdata.empty_value(stats), cond=args.cond,
        cond_extra=u.shape[1]).to(device)
    if args.skip_shape:
        ck = torch.load(args.checkpoint_shape, map_location=device, weights_only=False)
        s_model.load_state_dict(ck["model_state"])
        print(f"stage B: reusing {args.checkpoint_shape}")
    else:
        print(f"\nstage B: shape ({args.cond}), "
              f"{sum(q.numel() for q in s_model.parameters()):,} params")
        s_model, _ = velocity_net.train(
            s_model, w, cond_shape, None, steps=args.steps_shape, batch=args.batch,
            lr=args.lr, device=device, checkpoint_path=args.checkpoint_shape, stats=stats,
            resume=args.resume)

    if not args.export:
        return

    # ---------------- sample ----------------
    e_model.eval()
    s_model.eval()
    sb = velocity_net.max_sample_batch(s_model, args.sample_batch)
    print(f"\nsampling {args.n_sample} showers (batch {sb}, {args.sample_steps} Heun steps)")
    idx = torch.randint(0, einc.shape[0], (args.n_sample,))
    eg, ec = einc[idx], e_cond[idx]
    out = []
    for s in range(0, args.n_sample, sb):
        ec_b = ec[s:s + sb].to(device)
        u_b = velocity_net.sample(e_model, ec_b, n_steps=args.sample_steps,
                                  device=device, readout="raw")
        w_b = velocity_net.sample(s_model, torch.cat([ec_b, u_b], dim=1),
                                  n_steps=args.sample_steps, device=device,
                                  gate="topk", readout="raw")
        out.append(layerdata.decode(u_b.cpu(), w_b.cpu(), eg[s:s + sb], stats)
                   .reshape(u_b.shape[0], -1))
        if s % (sb * 20) == 0:
            print(f"  {min(s + sb, args.n_sample)}/{args.n_sample}", flush=True)
    sh = torch.cat(out)

    r = sh.sum(1) / eg.reshape(-1)
    print(f"E_dep/E_inc median {r.median():.4f}  mean {r.mean():.4f}   (real ~0.78)")
    print(f"lit fraction {(sh > args.threshold_kev / 1000.0).float().mean():.4f}   (real ~0.25)")
    export.write_hdf5(args.export, sh, eg, stats, args.threshold_kev)
    print(f"check with:  python -m calods2.metrics.quick_metrics --gen {args.export}")


if __name__ == "__main__":
    main()
