"""
ae_baselines.py -- is the autoencoder's plateau a good place to stop?

Reconstruction error alone does not say whether a plateau is acceptable. This
compares the trained autoencoder, on the same held-out showers and the same
metric, against:

  zeros              predict nothing          (rel = 1 by construction)
  mean shower        the average shower in the same incident-energy bin;
                     encodes NOTHING about the individual shower
  PCA-k              the best possible LINEAR code of k numbers, fitted on
                     training showers. PCA-96 is the direct reference for a
                     96-number latent.

Reading the result:
  AE worse than the mean shower   -> the objective is broken, not the capacity
                                     (this is exactly what caught v3: 0.688
                                      against 0.638)
  AE ~ PCA-96                     -> the model only matches linear compression;
                                     the latent size or architecture binds, and
                                     more epochs will not help
  AE well below PCA-96            -> the non-linear model is earning its keep;
                                     a plateau here is a reasonable stop

    python ae_baselines.py --checkpoint best_calo_ae_v4.pt
"""

import argparse
import os
import sys

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


def rel(x, x_hat):
    return ((x - x_hat).norm(dim=1) / (x.norm(dim=1) + 1e-8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="best_calo_ae_v4.pt")
    p.add_argument("--n-val", type=int, default=2048, help="held-out showers compared")
    p.add_argument("--n-fit", type=int, default=15000, help="training showers for the PCA fit")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    # from_checkpoint reads the architecture from the file and tolerates the
    # one parameter that some pre-v4 checkpoints legitimately lack
    # (hit_logit_scale, used only by the v3 loss).
    model, ck = T.from_checkpoint(args.checkpoint, device=dev, strict=False)
    two_head = ck.get("two_head", False)

    # A units mismatch and a broken objective look identical in this table --
    # both give a huge rel error -- and the verdict below would blame the
    # objective either way. Checkpoints trained before the units fix learned
    # log1p(E_MeV / E_GeV), a ratio inflated ~1000x, and this script feeds them
    # log1p(E_GeV / E_GeV). Say so rather than letting the verdict mislead.
    if not ck.get("shower_units_gev", False):
        print(f"\n  WARNING: {args.checkpoint} has no `shower_units_gev` marker, so it\n"
              f"  was trained under the OLD convention log1p(E_MeV / E_GeV). The\n"
              f"  autoencoder row below is therefore a UNITS MISMATCH, not a\n"
              f"  reconstruction result, and the verdict does not apply to it.\n"
              f"  The PCA and mean-shower baselines are unaffected.")
    latent = model.latent_channels * model.latent_length
    print(f"\n{args.checkpoint}: epoch {ck.get('epoch')}, val_loss {ck.get('val_loss', float('nan')):.5f}, "
          f"{'v4 two-head' if two_head else 'pre-v4'}, latent = {latent} numbers")

    val_idx = torch.tensor(T.val_dataset.indices[:args.n_val])
    fit_idx = torch.tensor(T.train_dataset.indices[:args.n_fit])

    def fractions(idx):
        return (T.showers[idx].float().reshape(len(idx), -1)
                / (T.energies[idx].float().reshape(-1, 1) + 1e-8))

    Xv = fractions(val_idx).to(dev)
    xs, es = T.showers[val_idx].float(), T.energies[val_idx].float()

    # ---- the autoencoder itself ----
    errs, e_errs, hits = [], [], []
    with torch.no_grad():
        for s in range(0, len(val_idx), 256):
            x = xs[s:s + 256].to(dev)
            e = es[s:s + 256].to(dev)
            xl = T.physical_to_model(x, e)
            xh = model(xl, e)[0].float()
            B = xl.shape[0]
            t = torch.expm1(xl).clamp(min=0).reshape(B, -1)
            r = torch.expm1(xh).clamp(min=0).reshape(B, -1)
            errs.append(rel(t, r).cpu())
            e_errs.append(((r.sum(1) - t.sum(1)).abs() / t.sum(1)).cpu())
            hits.append((r > 0).float().mean(1).cpu())
    ae_rel = torch.cat(errs).mean().item()
    print(f"  energy error {torch.cat(e_errs).mean():.4f}   "
          f"lit-voxel fraction {torch.cat(hits).mean():.4f} "
          f"(real {(Xv > 0).float().mean():.4f})")

    # ---- baselines ----
    rows = [("autoencoder", ae_rel, "")]
    rows.append(("predict zeros", rel(Xv, torch.zeros_like(Xv)).mean().item(), ""))

    Xt = fractions(fit_idx).to(dev)
    et = T.energies[fit_idx].float().reshape(-1).to(dev)
    ev = es.reshape(-1).to(dev)
    edges = torch.quantile(torch.log10(et), torch.linspace(0, 1, 21).to(dev))
    bt = torch.bucketize(torch.log10(et), edges[1:-1])
    bv = torch.bucketize(torch.log10(ev), edges[1:-1])
    means = torch.stack([Xt[bt == b].mean(0) if (bt == b).any() else Xt.mean(0)
                         for b in range(20)])
    rows.append(("mean shower per energy bin", rel(Xv, means[bv]).mean().item(),
                 "encodes nothing per-shower"))
    del bt

    mu = Xt.mean(0)
    Xc = Xt - mu
    del Xt
    torch.cuda.empty_cache()
    C = (Xc.T @ Xc) / Xc.shape[0]
    del Xc
    torch.cuda.empty_cache()
    evecs = torch.linalg.eigh(C).eigenvectors.flip(1)
    del C
    torch.cuda.empty_cache()
    for k in (8, 32, latent, 256, 512):
        V = evecs[:, :k]
        rec = (((Xv - mu) @ V) @ V.T + mu).clamp(min=0)
        note = "<- same size as the latent" if k == latent else ""
        rows.append((f"PCA-{k}", rel(Xv, rec).mean().item(), note))

    print(f"\n  {'method':<30s} {'rel error':>10s}")
    print("  " + "-" * 58)
    for name, v, note in rows:
        mark = " *" if name == "autoencoder" else "  "
        print(f" {mark}{name:<30s} {v:10.4f}   {note}")

    pca_same = next(v for n, v, _ in rows if n == f"PCA-{latent}")
    mean_b = next(v for n, v, _ in rows if n.startswith("mean shower"))
    print()
    if ae_rel > mean_b:
        print("  VERDICT: worse than the per-energy mean shower -- the objective is broken,")
        print("           not the capacity. More epochs will not help.")
    elif ae_rel > 0.95 * pca_same:
        print(f"  VERDICT: only matching linear compression (PCA-{latent} = {pca_same:.4f}).")
        print("           The latent size or architecture binds; more epochs will not help.")
        print("           Consider a larger latent or a stronger decoder.")
    else:
        print(f"  VERDICT: clearly better than PCA-{latent} ({pca_same:.4f}); the non-linear")
        print("           model is earning its keep and this plateau is a reasonable stop.")


if __name__ == "__main__":
    main()
