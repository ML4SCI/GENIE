"""
sweep.py -- grid sweeps over the PINNDE samplers, with one metrics table out.

The point of the split below is cost. TRAIN-time parameters (width, steps,
batch, conditioning) need a run each; SAMPLE-time parameters (gate, read-out,
integration steps) reuse a trained checkpoint and cost seconds. Sweeping them
together would retrain for nothing, which is how the gate comparison was done
by hand the first time.

    # train-time sweep, each config sampled with the default read-out
    python -m calods2.metrics.sweep --train "hidden=512,1024,2048" --steps 16000

    # sample-time sweep over an existing checkpoint: no training at all
    python -m calods2.metrics.sweep --checkpoint src/calods2/velocity_film.pt \
        --sample "gate=topk,threshold,sample readout=denoise,raw"

    # both: every trained model evaluated under every read-out
    python -m calods2.metrics.sweep --train "cond=concat,film" --steps 16000 \
        --sample "gate=topk,threshold"

Results go to a markdown table on stdout and a CSV next to --out-dir, sorted
by the metric given to --sort (default: the dim-voxel deficit, which is the
defect the classifier keys on and which aggregate metrics cannot see).
"""

import argparse
import csv
import itertools
import os
import time

import torch

from calods2.data import export
import calods2.data.dataset as gendata
from calods2.metrics import quick_metrics
from calods2.models import velocity_net

THR = 0.01515


def parse_grid(spec):
    """'hidden=512,1024 steps=8000,16000' -> [{'hidden':512,...}, ...]"""
    if not spec:
        return [{}]
    axes = {}
    for part in spec.split():
        key, _, values = part.partition("=")
        out = []
        for v in values.split(","):
            try:
                out.append(int(v))
            except ValueError:
                try:
                    out.append(float(v))
                except ValueError:
                    out.append(v)
        axes[key.strip()] = out
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]


@torch.no_grad()
def evaluate(sh_mev, eg_mev, real_sh, real_e, pool, device):
    """Compact metric set; same definitions as quick_metrics."""
    r = sh_mev.sum(1) / eg_mev.reshape(-1)
    rr = real_sh.sum(1) / real_e.reshape(-1)
    nz = sh_mev[sh_mev > 0]
    nzr = real_sh[real_sh > 0]
    qs = torch.tensor([0.05, 0.5], device=device)
    lg, rg = quick_metrics.profiles(sh_mev, eg_mev)
    lr_, rr_ = quick_metrics.profiles(real_sh, real_e)
    g_l = quick_metrics.log_space(sh_mev, THR)
    r_l = quick_metrics.log_space(real_sh, THR)
    nn_g = quick_metrics.nn_distance(g_l, pool)
    nn_r = quick_metrics.nn_distance(r_l, pool)
    sub = (lambda t: t[torch.randperm(t.numel(), device=device)[:1 << 22]]) if nz.numel() > (1 << 22) else (lambda t: t)
    q = torch.quantile(sub(nz), qs).tolist()
    qr = torch.quantile(sub(nzr), qs).tolist()
    return {
        "ratio": round(r.mean().item(), 4),
        "ratio_sd": round(r.std().item(), 4),
        "hits": round((sh_mev > THR).float().sum(1).mean().item()),
        "dim_frac": round((nz < 1.0).float().mean().item(), 4),
        "vox5": round(q[0], 3),
        "vox50": round(q[1], 2),
        "long_err": round(quick_metrics.rel_l1(lg, lr_), 4),
        "rad_err": round(quick_metrics.rel_l1(rg, rr_), 4),
        "nn_ratio": round(nn_g / nn_r, 3),
        "_real": {"ratio": round(rr.mean().item(), 4), "ratio_sd": round(rr.std().item(), 4),
                  "hits": round((real_sh > THR).float().sum(1).mean().item()),
                  "dim_frac": round((nzr < 1.0).float().mean().item(), 4),
                  "vox5": round(qr[0], 3), "vox50": round(qr[1], 2),
                  "long_err": 0.0, "rad_err": 0.0, "nn_ratio": 1.0},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/Dataset2.pt")
    p.add_argument("--train", default="", help="train-time grid, e.g. 'hidden=512,1024 cond=concat,film'")
    p.add_argument("--sample", default="", help="sample-time grid, e.g. 'gate=topk,threshold readout=denoise,raw'")
    p.add_argument("--checkpoint", default=None,
                   help="Skip training and sweep sample-time options over this checkpoint.")
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--cond", default="film")
    p.add_argument("--representation", default="fraction")
    p.add_argument("--n-sample", type=int, default=2048)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--n-real", type=int, default=2048)
    p.add_argument("--out-dir", default="sweeps")
    p.add_argument("--sort", default="dim_frac")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---- real reference, loaded once ----
    data = torch.load(args.data, map_location="cpu")
    sh_all = data["showers"].float().reshape(-1, 6480)
    e_all_mev = data["energies"].float().reshape(-1, 1)
    del data
    idx = torch.randperm(sh_all.shape[0])[:args.n_real]
    real_sh, real_e = sh_all[idx].to(device), e_all_mev[idx].to(device)
    pool = quick_metrics.log_space(sh_all[:30000], THR).to(device)

    train_grid = parse_grid(args.train)
    sample_grid = parse_grid(args.sample) or [{}]
    rows = []

    for ti, tcfg in enumerate(train_grid):
        hidden = tcfg.get("hidden", args.hidden)
        steps = tcfg.get("steps", args.steps)
        batch = tcfg.get("batch", args.batch)
        cond = tcfg.get("cond", args.cond)
        rep = tcfg.get("representation", args.representation)

        if args.checkpoint:
            model, ck = velocity_net.from_checkpoint(args.checkpoint, device)
            stats = ck["stats"]
            _, e_cond_all, eg_all, _ = gendata.load_dataset2(
                args.data, device="cpu", representation=stats.get("representation", "fraction"))
            tag_train = os.path.basename(args.checkpoint)
        else:
            x0_all, e_cond_all, eg_all, stats = gendata.load_dataset2(
                args.data, device="cpu", representation=rep)
            model = velocity_net.VelocityNet(
                shower_dim=x0_all.shape[1], hidden=hidden, hit_head=True,
                empty_value=gendata.empty_value(stats), cond=cond).to(device)
            tag_train = f"h{hidden}_s{steps}_b{batch}_{cond}_{rep}"
            ckpt = os.path.join(args.out_dir, f"{tag_train}.pt")
            print(f"\n=== [{ti+1}/{len(train_grid)}] training {tag_train} "
                  f"({sum(q.numel() for q in model.parameters()):,} params)", flush=True)
            t0 = time.time()
            model, _ = velocity_net.train(model, x0_all, e_cond_all, None, steps=steps,
                                          batch=batch, lr=2e-4, device=device,
                                          checkpoint_path=ckpt, stats=stats, log_every=max(1, steps // 2))
            print(f"    trained in {time.time()-t0:.0f}s", flush=True)
            del x0_all

        for scfg in sample_grid:
            gate = scfg.get("gate", "topk")
            readout = scfg.get("readout", "denoise")
            nsteps = scfg.get("sample_steps", args.sample_steps)
            tag = f"{tag_train} | gate={gate} readout={readout} T={nsteps}"
            sidx = torch.randint(0, eg_all.shape[0], (args.n_sample,))
            # conv activations are ~16x the MLP's per sample; 1024 OOMs a 4 GB card
            chunk = 128 if getattr(model, "cond", "") == "conv" else 1024
            chunks = []
            for s in range(0, args.n_sample, chunk):
                chunks.append(velocity_net.sample(
                    model, e_cond_all[sidx[s:s + chunk]].to(device), n_steps=nsteps,
                    device=device, gate=gate, readout=readout).cpu())
            x0_std = torch.cat(chunks)
            sh, diag = export.physical_showers_mev(x0_std, eg_all[sidx], stats, report=False)
            sh = sh.reshape(sh.shape[0], -1).to(device)      # (N, Nz, Nphi, Nr) -> (N, 6480)
            sh = torch.where(sh < THR, torch.zeros_like(sh), sh)
            m = evaluate(sh, (eg_all[sidx] * 1000.0).to(device), real_sh, real_e, pool, device)
            real = m.pop("_real")
            m["diverged"] = diag["showers_affected"]
            rows.append({"config": tag, **m})
            print(f"  {tag}\n    " + "  ".join(f"{k}={v}" for k, v in m.items()), flush=True)
            del x0_std, sh
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    # ---- table ----
    rows.sort(key=lambda r: abs(r.get(args.sort, 0) - real.get(args.sort, 0))
              if args.sort in real else r.get(args.sort, 0))
    cols = ["config", "ratio", "ratio_sd", "hits", "dim_frac", "vox5", "vox50",
            "long_err", "rad_err", "nn_ratio", "diverged"]
    print("\n\n| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    print("| **real** | " + " | ".join(str(real.get(c, "-")) for c in cols[1:]) + " |")
    print(f"\n(sorted by closeness to real on '{args.sort}')")

    csv_path = os.path.join(args.out_dir, "sweep_results.csv")
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if f.tell() == 0:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"appended {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
