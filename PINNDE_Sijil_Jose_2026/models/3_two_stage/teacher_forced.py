"""
teacher_forced.py -- measure the cost of the stage A -> stage B handoff.

Stage B is trained on the TRUE u of each shower (train_two_stage.py builds
cond_shape from layerdata.encode, never from stage A's samples), but at
sampling time it receives stage A's GENERATED u. It has therefore never seen
an imperfect u during training -- textbook exposure bias.

This script isolates that. It samples the shape model conditioned on REAL u
and decodes with that same real u, so stage A is removed from the pipeline
entirely:

    end-to-end     u ~ stage A        -> w ~ stage B(.|u)     -> decode(u, w)
    teacher-forced u  = encode(data)  -> w ~ stage B(.|u)     -> decode(u, w)

The difference between the two output files is the handoff cost. This is the
same diagnostic move as train_latent.py --ceiling: feed the pipeline real data
at the join and see how much error belongs to the generator versus the join.

What to expect, and why the energy observables are the WRONG place to look:
decode() reconstructs the total as f * E_inc * u_0 whatever w is, and the
per-layer projection forces sum_j w = 1, so E_tot/E_inc is exact by
construction in BOTH runs. Any handoff cost shows up in the SHAPE
observables -- centroids, widths, per-layer voxel distributions -- i.e.
"is this shape appropriate for the energy profile I was handed".

Note this is not a generalization test: the real u come from the same file the
models trained on. It measures sensitivity to u-error, not held-out quality.

Usage:
    python models/3_two_stage/teacher_forced.py \
        --data data/Dataset2.pt --checkpoint-shape checkpoints/ts_shape.pt \
        --export generated/ts_teacher_forced_100k.hdf5 --n-sample 100000

    python run_full_evaluation.py --skip-generate \
        --gen-file generated/ts_teacher_forced_100k.hdf5 \
        --out-dir evaluation_teacher_forced
"""

import argparse

import torch

from calods2.data import export
from calods2.data import layerdata
from calods2.models import velocity_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/Dataset2.pt")
    p.add_argument("--checkpoint-shape", default="checkpoints/ts_shape.pt")
    p.add_argument("--export", required=True)
    p.add_argument("--n-sample", type=int, default=100000)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--sample-batch", type=int, default=512)
    p.add_argument("--threshold-kev", type=float, default=15.15)
    p.add_argument("--gate", default="topk",
                   choices=["topk", "gumbel", "sample", "threshold"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)

    ck = torch.load(args.checkpoint_shape, map_location=device, weights_only=False)
    stats = ck["stats"]
    if stats is None:
        raise SystemExit("checkpoint has no stats; cannot encode consistently")

    data = torch.load(args.data, map_location="cpu")
    showers = data["showers"].float()
    einc = data["energies"].float().reshape(-1, 1)
    del data

    # Re-encode with the checkpoint's OWN stats, so f_scale / w_mean / w_std
    # match training exactly. Re-deriving them here would silently shift the
    # representation (f_scale is 1.05 * max(E_tot/E_inc) over whatever subset
    # it is given -- a 20k subset gives 1.97 where the full 100k gives 2.98).
    u, _, _ = layerdata.encode(showers, einc, stats=stats)
    del showers
    log_e = torch.log10(einc / 1000.0 + 1e-8)
    e_cond = (log_e - stats["log_e_mean"]) / stats["log_e_std"]

    model = velocity_net.VelocityNet(
        shower_dim=ck["shower_dim"], hidden=ck["hidden"], sigma0=ck["sigma0"],
        precond=ck.get("precond", False), hit_head=ck.get("hit_head", True),
        empty_value=ck.get("empty_value", 0.0), cond=ck.get("cond", "conv"),
        lit_head=ck.get("lit_head", False), cond_extra=ck.get("cond_extra", 0),
        pos_channels=ck.get("pos_channels", False)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"stage B: {args.checkpoint_shape} at step {ck.get('step')}, "
          f"cond={model.cond}, cond_extra={model.cond_extra}")

    if model.cond_extra != u.shape[1]:
        raise SystemExit(f"checkpoint expects cond_extra={model.cond_extra} "
                         f"but u has {u.shape[1]} columns")

    idx = torch.randint(0, einc.shape[0], (args.n_sample,))
    eg, ec, ub = einc[idx], e_cond[idx], u[idx]

    sb = velocity_net.max_sample_batch(model, args.sample_batch)
    print(f"sampling {args.n_sample} showers on REAL u "
          f"(batch {sb}, {args.sample_steps} Heun steps, gate {args.gate})")
    out = []
    for s in range(0, args.n_sample, sb):
        ec_b = ec[s:s + sb].to(device)
        u_b = ub[s:s + sb].to(device)
        w_b = velocity_net.sample(model, torch.cat([ec_b, u_b], dim=1),
                                  n_steps=args.sample_steps, device=device,
                                  gate=args.gate, readout="raw")
        # decode with the SAME real u that conditioned the sample
        out.append(layerdata.decode(u_b.cpu(), w_b.cpu(), eg[s:s + sb], stats)
                   .reshape(u_b.shape[0], -1))
        if s % (sb * 20) == 0:
            print(f"  {min(s + sb, args.n_sample)}/{args.n_sample}", flush=True)
    sh = torch.cat(out)

    r = sh.sum(1) / eg.reshape(-1)
    print(f"E_dep/E_inc median {r.median():.4f}  (exact by construction here)")
    print(f"lit fraction {(sh > args.threshold_kev / 1000.0).float().mean():.4f}"
          f"   (real ~0.25)")
    export.write_hdf5(args.export, sh, eg, stats, args.threshold_kev)
    print("\ncompare the SHAPE observables against the end-to-end file:")
    print(f"  python -m calods2.metrics.quick_metrics --gen {args.export}")


if __name__ == "__main__":
    main()
