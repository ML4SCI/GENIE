"""
sample_velocity.py -- sample + export from an ALREADY TRAINED velocity
checkpoint, without retraining.

Also the tool for diagnosing sampler blow-up: --diagnose reports, per
incident-energy bin, how far the raw ODE output goes and how many voxels end
up above the incident energy, before anything is clamped. Fewer sampling
steps or a different time grid can be tried here in seconds rather than by
repeating a training run.

Usage:
    # re-export from a finished run
    python models/2_direct_pinn/sample.py --checkpoint src/calods2/velocity_full.pt \
        --data data/Dataset2.pt --export generated/pinnde_recon_ds2.hdf5 \
        --n-sample 100000

    # just look at how bad the blow-up is (no file written)
    python models/2_direct_pinn/sample.py --checkpoint src/calods2/velocity_full.pt \
        --data data/Dataset2.pt --n-sample 2048 --diagnose
"""

import argparse
import sys

import torch

from calods2.data import export
import calods2.data.dataset as gendata
from calods2.models import velocity_net


def load_model(checkpoint, device):
    """Thin wrapper; the implementation lives with the model class so that
    sweep.py and any other caller do not have to import an entrypoint."""
    return velocity_net.from_checkpoint(checkpoint, device)


@torch.no_grad()
def diagnose(x0_std, energy_gev, stats):
    """Where does the raw (unclamped) output go, by incident energy?"""
    x_log = gendata.unstandardize_x0(x0_std.double(), stats)
    eg = energy_gev.double().reshape(-1)
    print(f"\n{'E_inc [GeV]':>16s} {'n':>6s} {'max x_log':>12s} {'voxels>E_inc':>13s} "
          f"{'showers hit':>12s} {'ratio median':>13s}")
    edges = torch.tensor([0.0, 1.0, 10.0, 100.0, 1e9], dtype=torch.float64)
    for i in range(4):
        m = (eg >= edges[i]) & (eg < edges[i + 1])
        if m.sum() == 0:
            continue
        xl = x_log[m]
        over = xl > export.MAX_X_LOG
        r = (torch.expm1(xl.clamp(max=export.MAX_X_LOG)).clamp(min=0).sum(1))
        print(f"{edges[i]:7.0f}-{edges[i+1]:<8.0f} {int(m.sum()):6d} {xl.max():12.3f} "
              f"{int(over.sum()):13d} {float(over.any(dim=1).float().mean()):12.4f} "
              f"{r.median():13.4f}")
    print("\n  x_log above log(2) = 0.693 means one voxel holds the whole incident\n"
          "  energy: unphysical. ratio median is E_dep/E_inc after clamping (real ~0.78).")


@torch.no_grad()
def check_denoiser(model, data_path, stats, device, n=1024, seed=0):
    """
    Is the energy deficit in the FIELD or in the TRAJECTORY?

    Takes REAL held-out showers, noises them to x_t exactly as training does,
    and asks the network to denoise them back. This never integrates the ODE,
    so it isolates q itself:

      * ungated ratio ~1 at small t  => the field reproduces energy correctly,
        and any deficit in generated showers comes from where the ODE lands
        (or from the gate).
      * ungated ratio < 1 at small t => the field itself under-predicts energy
        and no sampling-side change can fix it.

    The gated column shows separately what the hit gate costs on inputs whose
    true answer is known.
    """
    torch.manual_seed(seed)
    x0_all, e_all, eg_all, _ = gendata.load_dataset2(
        data_path, device="cpu", representation=stats.get("representation", "fraction"))
    idx = torch.randperm(x0_all.shape[0])[:n]
    x0, e, eg = x0_all[idx].to(device), e_all[idx].to(device), eg_all[idx].to(device)
    s0 = model.sigma0

    def energy(x_std):
        sh = gendata.x0_to_physical_showers(
            gendata.unstandardize_x0(x_std.double(), stats), eg.double(), stats)
        return sh.reshape(sh.shape[0], -1).sum(1)

    e_true = energy(x0)
    lit = x0 > model.empty_value + 1e-6                      # truly lit voxels
    empty_v = model.empty_value.to(x0.dtype)
    print(f"\n{'t':>5s} {'on true hits':>13s} {'spurious':>10s} {'gated':>8s} "
          f"{'missed':>8s} {'hits':>7s}   (real hits "
          f"{int(lit.float().sum(1).mean())})")
    for t in (0.0, 0.05, 0.2, 0.5):
        tt = torch.full((x0.shape[0], 1), float(t), device=device)
        xt = (1 - t) * x0 + (s0 + (1 - s0) * t) * torch.randn_like(x0)
        q = model.q_hat(tt, xt, e)
        g = model.denoise(tt, xt, e, gate="topk")
        on_true = energy(torch.where(lit, q, empty_v)) / e_true      # field on real deposits
        spurious = energy(torch.where(lit, empty_v, q)) / e_true     # invented energy
        gated = energy(g) / e_true                                   # what the sampler emits
        # true energy sitting in voxels the gate switched off
        off = (g <= model.empty_value + 1e-6) & lit
        missed = energy(torch.where(off, x0, empty_v)) / e_true
        hits = (g > model.empty_value + 1e-6).float().sum(1).mean()
        print(f"{t:5.2f} {on_true.median():13.4f} {spurious.median():10.4f} "
              f"{gated.median():8.4f} {missed.median():8.4f} {hits:7.0f}")
    print("  All columns are medians of E/E(x0) over held-out real showers; t=0 is the\n"
          "  read-out the sampler uses. 'on true hits' is the field's energy on voxels that\n"
          "  really are lit -- below 1 means the FIELD is biased low and no sampling-side\n"
          "  change can fix it. 'spurious' is energy invented in truly-empty voxels (what\n"
          "  the gate exists to remove). 'missed' is true energy in voxels the gate switched\n"
          "  off. gated ~ on_true - missed + (spurious that survived the gate).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/Dataset2.pt")
    p.add_argument("--export", default=None, help="HDF5 to write; omit to only diagnose")
    p.add_argument("--n-sample", type=int, default=100000)
    p.add_argument("--sample-steps", type=int, default=100)
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
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--fixed-eval", action="store_true",
                   help="Deterministic validation loss on held-out real showers (fixed noise, "
                        "fixed t grid). Safe to run against a checkpoint that is still "
                        "training: re-run it later and compare. See VelocityNet.fixed_eval.")
    p.add_argument("--check-denoiser", action="store_true",
                   help="Denoise REAL held-out showers and compare total energy: separates a "
                        "biased field from a biased ODE trajectory. Writes nothing.")
    p.add_argument("--max-retries", type=int, default=2,
                   help="Re-sample diverged showers, x4 time steps per attempt "
                        "(0 disables). See export.diverged_mask.")
    p.add_argument("--drop-diverged", action="store_true",
                   help="Exclude showers still diverged after the retries, instead "
                        "of clamping them.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    model, ck = load_model(args.checkpoint, device)
    if args.fixed_eval:
        stats_ = ck["stats"]
        x0_all, e_all, _, _ = gendata.load_dataset2(
            args.data, device="cpu", representation=stats_.get("representation", "fraction"))
        torch.manual_seed(0)
        n_eval = 512 if model.cond != "conv" else 64      # conv activations are large
        idx = torch.randperm(x0_all.shape[0])[:n_eval]
        val, per_t = model.fixed_eval(x0_all[idx].to(device), e_all[idx].to(device))
        steps_done = (ck.get("history") or [[0]])[-1][0]
        print(f"\nfixed_eval at {steps_done} trained steps: mean MSE(q, x0) = {val:.6f}")
        print(f"  {'t':>7s} {'MSE':>10s}")
        for v, t in per_t:
            print(f"  {t:7.3f} {v:10.5f}")
        print("  Deterministic: re-run later on the same checkpoint and compare. A flat mean\n"
              "  with a falling low-t column still means progress -- high t is nearly\n"
              "  irreducible (the best prediction there is the energy-conditional mean).")
        if not args.export:
            return

    if args.check_denoiser:
        check_denoiser(model, args.data, ck["stats"], device)
        if not args.export:
            return

    # The checkpoint carries the stats it was trained with, including the
    # representation -- never re-derive them from the data here, or a
    # representation mismatch would silently corrupt the inverse transform.
    stats = ck["stats"]
    _, _, energy_gev, _ = gendata.load_dataset2(
        args.data, device="cpu", representation=stats.get("representation", "fraction"))
    idx = torch.randint(0, energy_gev.shape[0], (args.n_sample,))
    eg_all = energy_gev[idx]
    e_cond_all = gendata.energy_gev_to_cond(eg_all, stats)

    sample_batch = velocity_net.max_sample_batch(model, args.sample_batch,
                                              explicit="--sample-batch" in sys.argv)
    if sample_batch != args.sample_batch:
        print(f"  sampling batch capped {args.sample_batch} -> {sample_batch} "
              f"(conv activations; see velocity_net.CONV_SAMPLE_BATCH)")
    print(f"sampling {args.n_sample} showers ({args.sample_steps} Heun steps each)")
    chunks = []
    for s in range(0, args.n_sample, sample_batch):
        e_c = e_cond_all[s:s + sample_batch].to(device)
        chunks.append(velocity_net.sample(model, e_c, n_steps=args.sample_steps,
                                          device=device, gate=args.gate,
                                          value_correct=args.value_correct,
                                          readout=args.readout).cpu())
        if s % (sample_batch * 20) == 0:
            print(f"  {min(s + sample_batch, args.n_sample)}/{args.n_sample}", flush=True)
    x0_std = torch.cat(chunks)

    # Retry the showers whose ODE ran away, each attempt with a finer time grid
    # and a fresh noise draw. Rejecting divergent draws biases the sample, so
    # the counts are printed rather than quietly absorbed -- if the retried
    # fraction is not tiny, the model is the problem, not the integrator.
    if args.max_retries > 0:
        bad = export.diverged_mask(x0_std, stats)
        steps = args.sample_steps
        for attempt in range(1, args.max_retries + 1):
            if not bad.any():
                break
            steps *= 4
            idx_bad = bad.nonzero(as_tuple=True)[0]
            print(f"  retry {attempt}: {len(idx_bad)} diverged shower(s) "
                  f"re-sampled with {steps} steps", flush=True)
            fixed = []
            for s in range(0, len(idx_bad), sample_batch):
                sel = idx_bad[s:s + sample_batch]
                e_c = e_cond_all[sel].to(device)
                fixed.append(velocity_net.sample(model, e_c, n_steps=steps,
                                                 device=device, gate=args.gate,
                                          value_correct=args.value_correct,
                                          readout=args.readout).cpu())
            x0_std[idx_bad] = torch.cat(fixed)
            bad = export.diverged_mask(x0_std, stats)
        n_bad = int(bad.sum())
        if n_bad:
            print(f"  {n_bad} shower(s) ({n_bad / x0_std.shape[0]:.4%}) still diverged "
                  f"after {args.max_retries} retries; they are CLAMPED below, which "
                  f"distorts them -- use --drop-diverged to exclude them instead")
            if args.drop_diverged:
                keep = ~bad
                x0_std, eg_all = x0_std[keep], eg_all[keep]
                print(f"  dropped them: {x0_std.shape[0]} showers remain")

    if args.diagnose:
        diagnose(x0_std, eg_all, stats)

    sh_mev, diag = export.physical_showers_mev(x0_std, eg_all, stats)
    print(f"  E_dep/E_inc median {diag['ratio_median']:.4f}  mean {diag['ratio_mean']:.4f}"
          f"   (real ~0.78)")
    if args.export:
        export.write_hdf5(args.export, sh_mev, eg_all * 1000.0, stats, args.threshold_kev)
        print(f"check with:  python -m calods2.metrics.quick_metrics --gen {args.export}")


if __name__ == "__main__":
    main()
