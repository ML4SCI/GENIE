"""
quick_metrics.py -- fast physics check on an exported HDF5 of generated showers.

Not a replacement for the official evaluation (run_full_evaluation.py); this is
the loop you iterate on while training, and it reports the quantities this
sampler has actually been failing on, each next to the same quantity measured
on REAL showers from Dataset2.pt:

  E_dep/E_inc      mean and spread, overall and in incident-energy bins
                   (the sampler's worst metric: 0.95 +- 0.38 vs a real
                   0.78 +- 0.12 at the best checkpoint measured so far)
  hit fraction     voxels above the readout threshold (77% vs a real 25%
                   before the hit head existed)
  layer / radial    relative L1 error of the mean longitudinal and radial
    profile         energy profiles, normalized by incident energy
  NN distance      distance from each generated shower to its nearest REAL
                   shower, in log space, against the same statistic measured
                   real-to-real. Much smaller => memorization; much larger
                   => off-manifold samples. (The exact Monte-Carlo solver
                   scored 0.4 against a real-to-real 47: it copies showers.)

Usage:
    python -m calods2.metrics.quick_metrics --gen generated/pinnde_recon_ds2.hdf5 \
        --data data/Dataset2.pt [--n 2048] [--threshold-kev 15.15]
"""

import argparse

import h5py
import torch

THRESHOLD_MEV_DEFAULT = 0.01515


def load_real(data_path, n, device):
    data = torch.load(data_path, map_location="cpu")
    sh = data["showers"].float().reshape(-1, 6480)
    e = data["energies"].float().reshape(-1, 1)
    idx = torch.randperm(sh.shape[0])[:n]
    return sh[idx].to(device), e[idx].to(device), sh, e


def profiles(sh, einc, Nz=45, Nphi=16, Nr=9):
    g = sh.reshape(-1, Nz, Nphi, Nr) / einc.reshape(-1, 1, 1, 1)
    return g.sum(dim=(2, 3)).mean(0), g.sum(dim=(1, 2)).mean(0)


def rel_l1(a, b):
    return (a - b).abs().sum().item() / b.abs().sum().item()


@torch.no_grad()
def nn_distance(x, ref, chunk=4096):
    """Mean distance from each row of x to its nearest row of ref, in log space."""
    best = torch.full((x.shape[0],), float("inf"), device=x.device)
    for s in range(0, ref.shape[0], chunk):
        d = torch.cdist(x, ref[s:s + chunk].to(x.device)).min(dim=1).values
        best = torch.minimum(best, d)
    return best.mean().item()


def log_space(sh, thr):
    return torch.log1p(sh / thr)


def report(gen_sh, gen_e, real_sh, real_e, real_pool, thr, device,
           paired_true=None):
    rows = []
    qs = torch.tensor([0.05, 0.5, 0.95], device=device)
    for tag, sh, e in (("generated", gen_sh, gen_e), ("real", real_sh, real_e)):
        r = sh.sum(1) / e.squeeze()
        hits = (sh > thr).float()
        # quantiles as well as mean+-sd: in the log-based "threshold"
        # representation a few exponentiated outliers dominate the mean and sd
        # (measured: 0.72 +- 2.54 against a real 0.78 +- 0.11), so the median
        # and the 5/95% band say much more about the bulk of the showers.
        rows.append((tag, r.mean().item(), r.std().item(),
                     torch.quantile(r, qs).tolist(),
                     hits.mean().item(), hits.sum(1).mean().item()))
    print(f"\n{'':10s} {'E_dep/E_inc':>18s} {'5%':>7s} {'50%':>7s} {'95%':>7s} "
          f"{'hit frac':>10s} {'hits/shower':>12s}")
    for tag, m, s, q, hf, hs in rows:
        print(f"{tag:10s} {m:11.4f} +-{s:5.4f} {q[0]:7.3f} {q[1]:7.3f} {q[2]:7.3f} "
              f"{hf:10.4f} {hs:12.0f}")

    lg, rg = profiles(gen_sh, gen_e)
    lr, rr = profiles(real_sh, real_e)
    print(f"\nprofile relative L1 error:  longitudinal {rel_l1(lg, lr):.4f}   "
          f"radial {rel_l1(rg, rr):.4f}")

    # Per-voxel energy distribution. Every other metric here is an aggregate
    # over voxels, so a model can get total energy, hit count and both profiles
    # right while putting the energy in the wrong voxels -- which is exactly
    # what a classifier detects. Measured: generated showers under-produce dim
    # voxels badly (5th percentile 3.9 MeV against a real 0.63) while matching
    # the median and the bright tail.
    print(f"\n{'nonzero voxel MeV':20s} {'5%':>9s} {'50%':>9s} {'95%':>9s} "
          f"{'<1 MeV':>8s} {'min':>10s} {'max':>8s}")
    qs = torch.tensor([0.05, 0.5, 0.95], device=device)
    for tag, sh in (("generated", gen_sh), ("real", real_sh)):
        nz = sh[sh > 0]
        # quantile() caps at 2^24 elements; sample when there are more
        s = nz[torch.randperm(nz.numel(), device=device)[:1 << 22]] if nz.numel() > (1 << 22) else nz
        q = torch.quantile(s, qs).tolist()
        print(f"{tag:20s} {q[0]:9.3f} {q[1]:9.3f} {q[2]:9.1f} "
              f"{(nz < 1.0).float().mean():8.4f} {nz.min():10.6f} {nz.max():8.0f}")
    print("  dim voxels are the ones aggregate metrics cannot see; '<1 MeV' is the\n"
          "  fraction of lit voxels that are faint, and min should sit at the 15.15 keV readout")

    print("\nE_dep/E_inc by incident energy:")
    edges = torch.tensor([0.0, 1e3, 1e4, 1e5, 1e9], device=device)   # MeV
    for i in range(4):
        out = []
        for tag, sh, e in (("gen", gen_sh, gen_e), ("real", real_sh, real_e)):
            m = (e.squeeze() >= edges[i]) & (e.squeeze() < edges[i + 1])
            if m.sum() < 5:
                out.append(f"{tag} n/a")
            else:
                r = sh[m].sum(1) / e.squeeze()[m]
                out.append(f"{tag} {r.mean():.3f}+-{r.std():.3f} (n={int(m.sum())})")
        print(f"  {edges[i]/1000:7.0f}-{edges[i+1]/1000:<8.0f} GeV   " + "   ".join(out))

    pool = log_space(real_pool[:30000], thr).to(device)
    g_l = log_space(gen_sh, thr)
    r_l = log_space(real_sh, thr)
    print(f"\nNN distance to real showers (log space):  generated {nn_distance(g_l, pool):.2f}"
          f"   real-to-real {nn_distance(r_l, pool):.2f}")
    print("  (much smaller than real-to-real => memorizing; much larger => off-manifold)")

    if paired_true is not None:
        occupancy_report(gen_sh, paired_true, thr)


def occupancy_report(gen_sh, true_sh, thr, cuts=(1.0, 2.0)):
    """
    Precision / recall / F1 on which voxels are lit -- only meaningful when
    gen_sh is a RECONSTRUCTION of true_sh, voxel for voxel.

    Why this exists: every other metric here is a distribution comparison, and
    a model can match the lit FRACTION exactly while lighting the wrong
    voxels. Measured on the pre-v4 autoencoder: lit fraction 0.42 against a
    real 0.245 came with recall 0.92 but precision 0.54 -- it finds almost
    every real hit and then paints 2.3x too many. That is the "over-painting"
    signature, and no aggregate in this script could see it.

    Borrowed from the NOvA sparse-CAE report (Prosper 2026), where F1 is the
    figure of merit for 99.4%-empty pixel maps.

    The extra `cuts` sweep a model-side energy cut, which trades recall for
    precision without retraining -- the same knob as that report's tau, and
    the same thing CaloDREAM does with its 1 MeV cut. NOTE it buys F1 by
    DELETING the dim population (real showers carry ~7% of hits below 1 MeV),
    so it is not a free win for a classifier metric. See docs/NOVA_TRANSFER.md.
    """
    def prf(g, t):
        tp = (t & g).sum().item()
        fp = ((~t) & g).sum().item()
        fn = (t & (~g)).sum().item()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return p, r, (2 * p * r / max(p + r, 1e-9))

    ht = true_sh > thr
    print(f"\noccupancy placement (reconstruction only; voxel-paired)")
    print(f"{'cut [MeV]':>10s}{'precision':>11s}{'recall':>9s}{'F1':>8s}"
          f"{'lit frac':>10s}{'hits r/t':>10s}")
    for c in (thr,) + tuple(cuts):
        hg = gen_sh > c
        p, r, f1 = prf(hg, ht)
        ratio = (hg.sum(1).float() / ht.sum(1).clamp(min=1).float()).mean().item()
        print(f"{c:>10.3f}{p:>11.4f}{r:>9.4f}{f1:>8.4f}{hg.float().mean():>10.4f}{ratio:>10.3f}")
    print(f"  true lit fraction {ht.float().mean():.4f}")
    print("  precision << recall => over-painting; recall << precision => biased sparse")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True, help="HDF5 written by train_velocity --export")
    p.add_argument("--data", default="data/Dataset2.pt")
    p.add_argument("--n", type=int, default=2048, help="showers compared per side")
    p.add_argument("--threshold-kev", type=float, default=15.15)
    p.add_argument("--paired", action="store_true",
                   help="The generated file is a voxel-for-voxel RECONSTRUCTION of "
                        "--data (same showers, same order), so occupancy precision/"
                        "recall/F1 are meaningful. Verify the incident energies match "
                        "before trusting this.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    thr = args.threshold_kev / 1000.0

    with h5py.File(args.gen, "r") as f:
        gen_sh = torch.tensor(f["showers"][:args.n]).float().to(device)
        gen_e = torch.tensor(f["incident_energies"][:args.n]).float().reshape(-1, 1).to(device)
    print(f"{args.gen}: {gen_sh.shape[0]} showers, {gen_sh.shape[1]} voxels (MeV)")

    real_sh, real_e, pool, _ = load_real(args.data, args.n, device)

    paired_true = None
    if args.paired:
        data = torch.load(args.data, map_location="cpu")
        pt = data["showers"].reshape(data["showers"].shape[0], -1)[:gen_sh.shape[0]]
        pe = data["energies"].reshape(-1)[:gen_sh.shape[0]]
        if not torch.allclose(pe, gen_e.squeeze().cpu(), rtol=1e-4):
            raise SystemExit("--paired given but incident energies do not match "
                             "the data file row for row; the files are not paired")
        paired_true = pt.float().to(device)

    report(gen_sh, gen_e, real_sh, real_e, pool, thr, device, paired_true)


if __name__ == "__main__":
    main()
