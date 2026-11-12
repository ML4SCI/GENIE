"""
ae_sweep.py -- sweep autoencoder latent sizes and architecture options, one table out.

Loads Dataset 2 once and trains every configuration in-process, which is the
whole point: the dataset is 2.5 GB and re-importing the trainer per config
would dominate the runtime.

Each row is scored on the SAME held-out showers with the same metric as
ae_baselines.py, and against the PCA code of the same size -- the only
reference that says whether a non-linear model is earning its keep:

    PCA-96 = 0.539    PCA-256 = 0.461    PCA-512 = 0.387    (measured)

The autoencoder at 96 numbers reaches 0.367, i.e. it already beats PCA-512, so
the question a latent sweep answers is how much further a bigger latent goes
before the architecture rather than the bottleneck binds.

    # latent size, ~10 epochs each (ranking run, not converged)
    python ae_sweep.py --grid "latent=8x12,16x12,16x24,32x24" --epochs 10

    # architecture options at a fixed latent
    python ae_sweep.py --grid "phi=0,1 width=1.0,2.0" --epochs 10

    # quick plumbing check (minutes, meaningless numbers)
    python ae_sweep.py --grid "latent=8x12" --epochs 1 --max-batches 20

Short runs RANK configurations; they do not converge them. v4 reached rel 0.563
at epoch 2 and 0.367 at epoch 255, so treat these as a shortlist, then train the
winner properly.
"""

import argparse
import csv
import itertools
import os
import sys
import time

import torch

os.environ.setdefault("CALO_BATCH_SIZE", "128")
# --- compatibility shim -------------------------------------------------
# These scripts were written against train_auto_encoder_improved.py, which
# loaded Dataset 2 and built DataLoaders at IMPORT time and exposed them as
# module globals (T.showers, T.train_dataset, T.train_loader, ...). That
# module is gone; the library loads nothing on import. `T` below reproduces
# the handful of attributes these scripts use, built explicitly.
# ------------------------------------------------------------------------
import os as _os, sys as _sys, types as _types
# Walk up to the repo root rather than assuming a fixed depth: these scripts
# have been moved once already (out of the flat CaloChallenge/ directory) and
# a hardcoded "../src" silently breaks on the next move.
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d):
    if _os.path.isdir(_os.path.join(_d, "src", "calods2")):
        _sys.path.insert(0, _os.path.join(_d, "src"))
        break
    _d = _os.path.dirname(_d)
from calods2.data import calo_dataset as _cd
from calods2.models import autoencoder as _ae
from calods2.models import ae_losses as _losses

def _make_compat():
    showers, energies, stats = _cd.load_calo_data(
        _os.environ.get("CALO_DATA", "data/Dataset2.pt"))
    train_loader, val_loader, train_idx = _cd.make_loaders(showers, energies)
    T = _types.SimpleNamespace(
        showers=showers, energies=energies,
        train_loader=train_loader, val_loader=val_loader,
        train_dataset=_types.SimpleNamespace(indices=train_idx),
        val_dataset=_types.SimpleNamespace(indices=val_loader.dataset.indices),
        physical_to_model=_cd.physical_to_model,
        model_to_physical=_cd.model_to_physical,
        FactorizedCaloAE=_ae.FactorizedCaloAE,
        from_checkpoint=_ae.from_checkpoint,
        relative_error_phys=_losses.relative_error_phys,
        energy_error_phys=_losses.energy_error_phys,
        LOG_E_MEAN=stats["log_e_mean"], LOG_E_STD=stats["log_e_std"],
    )
    return T

class _LazyT:
    """`T` was built at import time, which made even `--help` require a 2.5 GB
    dataset on disk. Build it on first attribute access instead."""
    _obj = None

    def __getattr__(self, name):
        if _LazyT._obj is None:
            _LazyT._obj = _make_compat()
        return getattr(_LazyT._obj, name)


T = _LazyT()
# --- end shim -----------------------------------------------------------


class LimitedLoader:
    """Wraps a DataLoader to yield at most n batches, for smoke runs."""

    def __init__(self, loader, n):
        self.loader, self.n = loader, n

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.n)

    def __len__(self):
        return min(self.n, len(self.loader))


def parse_grid(spec):
    if not spec:
        return [{}]
    axes = {}
    for part in spec.split():
        key, _, values = part.partition("=")
        out = []
        for v in values.split(","):
            if "x" in v and key == "latent":
                c, _, l = v.partition("x")
                out.append((int(c), int(l)))
            else:
                try:
                    out.append(float(v) if "." in v else int(v))
                except ValueError:
                    out.append(v)
        axes[key.strip()] = out
    keys = list(axes)
    return [dict(zip(keys, c)) for c in itertools.product(*(axes[k] for k in keys))]


@torch.no_grad()
def evaluate(model, val_idx, device):
    """rel error, energy error and lit fraction on held-out showers."""
    model.eval()
    errs, e_errs, lit = [], [], []
    xs, es = T.showers[val_idx].float(), T.energies[val_idx].float()
    for s in range(0, len(val_idx), 256):
        x, e = xs[s:s + 256].to(device), es[s:s + 256].to(device)
        xl = T.physical_to_model(x, e)
        xh = model(xl, e)[0].float()
        B = xl.shape[0]
        t = torch.expm1(xl).clamp(min=0).reshape(B, -1)
        r = torch.expm1(xh).clamp(min=0).reshape(B, -1)
        errs.append(((t - r).norm(dim=1) / (t.norm(dim=1) + 1e-8)).cpu())
        e_errs.append(((r.sum(1) - t.sum(1)).abs() / (t.sum(1) + 1e-8)).cpu())
        lit.append((r > 0).float().mean(1).cpu())
    model.train()
    return (torch.cat(errs).mean().item(), torch.cat(e_errs).mean().item(),
            torch.cat(lit).mean().item())


_PCA_CACHE = {}


@torch.no_grad()
def pca_reference(k, fit_idx, val_idx, device):
    """Best possible LINEAR code of k numbers, as the bar each latent must clear."""
    if k in _PCA_CACHE:
        return _PCA_CACHE[k]
    frac = lambda idx: (T.showers[idx].float().reshape(len(idx), -1)
                        / (T.energies[idx].float().reshape(-1, 1) + 1e-8))
    Xv = frac(val_idx).to(device)
    if "V" not in _PCA_CACHE or _PCA_CACHE["q"] < k:
        # Randomized PCA, not a full eigendecomposition: only the top-k
        # directions are ever used, and a 6480x6480 eigh is both wasteful and
        # fragile (it fails outright on some CPU LAPACK builds).
        Xt = frac(fit_idx).to(device)
        mu = Xt.mean(0)
        q = min(k + 32, min(Xt.shape))
        _, _, V = torch.pca_lowrank(Xt - mu, q=q, center=False, niter=4)
        _PCA_CACHE.update({"V": V, "mu": mu, "q": q})
        del Xt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    V = _PCA_CACHE["V"][:, :k]
    mu = _PCA_CACHE["mu"]
    rec = (((Xv - mu) @ V) @ V.T + mu).clamp(min=0)
    val = ((Xv - rec).norm(dim=1) / (Xv.norm(dim=1) + 1e-8)).mean().item()
    _PCA_CACHE[k] = val
    return val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="latent=8x12,16x12,16x24",
                   help="e.g. 'latent=8x12,16x24 phi=0,1 width=1.0,2.0'")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-batches", type=int, default=0, help="cap batches per epoch (smoke runs)")
    p.add_argument("--n-val", type=int, default=2048)
    p.add_argument("--n-fit", type=int, default=15000, help="training showers for the PCA bar")
    p.add_argument("--out-dir", default="ae_sweeps")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    val_idx = torch.tensor(T.val_dataset.indices[:args.n_val])
    fit_idx = torch.tensor(T.train_dataset.indices[:args.n_fit])

    sub = torch.tensor(T.train_dataset.indices[:20000])
    x_scale = T.physical_to_model(T.showers[sub].float(), T.energies[sub].float()).std().item()
    print(f"x_scale = {x_scale:.4e}; {args.epochs} epochs per config"
          f"{' (capped at %d batches)' % args.max_batches if args.max_batches else ''}\n")

    train_loader = (LimitedLoader(T.train_loader, args.max_batches)
                    if args.max_batches else T.train_loader)
    val_loader = (LimitedLoader(T.val_loader, max(1, args.max_batches // 4))
                  if args.max_batches else T.val_loader)

    rows = []
    for i, cfg in enumerate(parse_grid(args.grid)):
        lc, ll = cfg.get("latent", (8, 12))
        pc = bool(cfg.get("phi", 0))
        wm = float(cfg.get("width", 1.0))
        tag = f"latent{lc}x{ll}={lc*ll}_phi{int(pc)}_w{wm}"
        torch.manual_seed(args.seed)
        model = T.FactorizedCaloAE(
            Nz=T.L, Nphi=T.A, Nr=T.R, latent_channels=lc, latent_length=ll,
            energy_embed_dim=64, two_head=True, x_scale=x_scale,
            phi_circular=pc, width_mult=wm).to(device)
        n_par = sum(q.numel() for q in model.parameters())
        print(f"=== [{i+1}] {tag}  ({n_par:,} params)", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        sched = T.make_cosine_warmup_scheduler(opt, warmup_epochs=min(5, max(1, args.epochs // 4)),
                                               total_epochs=args.epochs)
        t0 = time.time()
        T.train_autoencoder(model, train_loader, val_loader, opt, device, epochs=args.epochs,
                            scheduler=sched, use_amp=True,
                            checkpoint_path=os.path.join(args.out_dir, f"{tag}.pt"))
        rel, e_err, lit = evaluate(model, val_idx, device)
        pca = pca_reference(lc * ll, fit_idx, val_idx, device)
        rows.append({"config": tag, "latent": lc * ll, "params": n_par, "rel": round(rel, 4),
                     "pca_same_size": round(pca, 4), "beats_pca": "yes" if rel < pca else "NO",
                     "energy_err": round(e_err, 4), "lit_frac": round(lit, 4),
                     "minutes": round((time.time() - t0) / 60, 1)})
        print(f"    rel={rel:.4f} (PCA-{lc*ll}={pca:.4f})  energy_err={e_err:.4f}  "
              f"lit={lit:.4f}  [{rows[-1]['minutes']} min]\n", flush=True)
        del model, opt
        torch.cuda.empty_cache()

    cols = ["config", "latent", "params", "rel", "pca_same_size", "beats_pca",
            "energy_err", "lit_frac", "minutes"]
    rows.sort(key=lambda r: r["rel"])
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print(f"\nreal lit fraction ~0.25; v4 at latent 96 converged to rel 0.367 in 255 epochs, so "
          f"these {args.epochs}-epoch numbers rank configurations rather than converge them.")

    path = os.path.join(args.out_dir, "ae_sweep_results.csv")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if f.tell() == 0:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
