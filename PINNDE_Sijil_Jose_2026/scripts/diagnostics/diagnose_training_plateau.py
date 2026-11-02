"""
diagnose_training_plateau.py
==============================
Training plateaus at val_loss~0.24 / rel_err~0.39 regardless of how long it
runs -- confirmed on the *identical* architecture at both epoch 85
(best_calo_ae_cond_improved.pt) and epoch 460 (the latent sweep's
lat_c8_l12_d96), so "just train longer" is ruled out empirically. This script
does NOT run more long training. It runs a battery of short, targeted tests
(minutes, not hours) that each isolate one specific mechanism that could
explain a plateau this early and this stubborn, so the answer is evidence,
not another guess.

Model/preprocessing code below is copied verbatim from
train_auto_encoder_improved.py, same pattern as the other standalone scripts
in this repo -- importing that file directly would re-run its module-level
training setup.

Tests (see TESTS dict at the bottom for the authoritative list):
  1. overfit_tiny_batch   -- can the model even memorize 16 fixed examples?
                             If not, there's a real bug in the training
                             mechanics, not a generalization limit.
  2. latent_ablation      -- swap out the encoded latent z (zeros / shuffled)
                             but keep the true energy -- if reconstruction
                             barely gets worse, the decoder is mostly
                             ignoring z and leaning on energy conditioning
                             (bottleneck collapse).
  3. loss_term_balance    -- gradient norm contributed by each loss term
                             separately, to see if one term (usually recon_l1
                             under the x_log<=6 range) drowns out the others.
  4. hit_loss_floor       -- is the sparsity BCE term stuck at the trivial
                             "predict 0.5 always" floor of ln(2)=0.693?
  5. units_target_range   -- how much does the current MeV-shower/GeV-energy
                             convention stretch the log1p target vs. a
                             same-units (GeV/GeV) convention?
  6. grad_clip_engagement -- how often, and by how much, does grad_clip=1.0
                             actually engage?
  7. amp_scaler_skips     -- is mixed-precision silently skipping optimizer
                             steps on overflow?
  8. lr_sensitivity       -- from identical random init, does a *lower*
                             constant LR make more progress in an equal
                             number of steps than the current 3e-4?

Usage:
    python diagnose_training_plateau.py --test all
    python diagnose_training_plateau.py --test overfit_tiny_batch latent_ablation
    python diagnose_training_plateau.py --list
    python diagnose_training_plateau.py --test all --device cuda   # opt-in;
        defaults to CPU so this never contends with a GPU job you have running

All tests use small batches / step counts by design -- this is about
mechanism, not about training a good model.
"""

import argparse
import copy
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

# ==============================================================================
# Model / preprocessing -- verbatim from train_auto_encoder_improved.py
# ==============================================================================

LOG_E_MEAN = None
LOG_E_STD = None


def physical_to_model(x, energy):
    E = energy.reshape(-1, 1, 1, 1)
    return torch.log1p(x / (E + EPS))


def model_to_physical(x_log, energy):
    E = energy.reshape(-1, 1, 1, 1)
    return torch.clamp(torch.expm1(x_log), min=0.0) * E


# energy_to_cond is a METHOD on the model now (it needs the model's own
# standardization constants); the module-level copy here was dead.


# Model: imported, not copied.
#
# This file carried its own 668-line copy of FactorizedCaloAE and its blocks.
# That copy used a decoder intermediate transverse resolution of (8, 4) while
# the training script had been corrected to the encoder's true mirror (8, 5).
# Conv weights do not encode spatial size, so each loaded into the other with
# zero missing or unexpected keys and silently reconstructed ~10% worse. Every
# checkpoint in this project was trained at (8, 4), so the copy matched the
# models and the main script had drifted away from them.
#
# LEGACY_MID_TRANSVERSE preserves that, so this script keeps diagnosing the
# models as they were actually trained -- which is the whole point of a
# post-mortem tool.
import os as _os, sys as _sys
# Walk up to the repo root rather than assuming a fixed depth: these scripts
# have been moved once already (out of the flat CaloChallenge/ directory) and
# a hardcoded "../src" silently breaks on the next move.
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d):
    if _os.path.isdir(_os.path.join(_d, "src", "calods2")):
        _sys.path.insert(0, _os.path.join(_d, "src"))
        break
    _d = _os.path.dirname(_d)
from calods2.models.autoencoder import FactorizedCaloAE, LEGACY_MID_TRANSVERSE
from calods2.models.ae_losses import relative_error_phys


def calorimeter_loss(x_hat_log, x_log, energy, mse_weight=1.0, l1_weight=1.0,
                      energy_weight=0.5, hit_weight=0.1, hit_threshold=1e-4):
    recon_mse = F.mse_loss(x_hat_log, x_log)
    weights = 1.0 + 10.0 * x_log
    recon_l1 = (weights * (x_hat_log - x_log).abs()).mean()
    recon_loss = mse_weight * recon_mse + l1_weight * recon_l1

    target_hit = (x_log > hit_threshold).float()
    hit_loss = F.binary_cross_entropy_with_logits(x_hat_log, target_hit)

    x_phys = torch.clamp(torch.expm1(x_log), min=0.0)
    xhat_phys = torch.clamp(torch.expm1(x_hat_log), min=0.0)
    E_true = x_phys.sum(dim=(1, 2, 3))
    E_pred = xhat_phys.sum(dim=(1, 2, 3))
    energy_loss = (torch.abs(E_true - E_pred) / (E_true + EPS)).mean()

    x_phys_target = torch.expm1(x_log)
    phys_loss = F.mse_loss(xhat_phys, x_phys_target)

    total = recon_loss + 0.02 * hit_loss + 0.05 * energy_loss + 0.1 * phys_loss
    return total, recon_mse, recon_l1, recon_loss, hit_loss, energy_loss, phys_loss


@torch.no_grad()

# ==============================================================================
# Shared setup
# ==============================================================================

def load_data(data_path, n=None, device='cpu'):
    data = torch.load(data_path, map_location='cpu')
    showers = data['showers'].float()
    energies_mev = data['energies'].float()
    if n is not None:
        idx = torch.randperm(showers.shape[0])[:n]
        showers, energies_mev = showers[idx], energies_mev[idx]
    energies_gev = (energies_mev / 1000.0).reshape(-1, 1)
    return showers.to(device), energies_gev.to(device)


def set_log_e_stats(mean, std):
    global LOG_E_MEAN, LOG_E_STD
    LOG_E_MEAN, LOG_E_STD = mean, std


def fresh_model(device, **kw):
    kw.setdefault('Nz', 45); kw.setdefault('Nphi', 16); kw.setdefault('Nr', 9)
    kw.setdefault('latent_channels', 8); kw.setdefault('latent_length', 12)
    kw.setdefault('energy_embed_dim', 64)
    # the resolution the checkpoints under diagnosis were trained at
    kw.setdefault('mid_transverse', LEGACY_MID_TRANSVERSE)
    if LOG_E_MEAN is not None:
        kw.setdefault('log_e_mean', LOG_E_MEAN)
        kw.setdefault('log_e_std', LOG_E_STD)
    return FactorizedCaloAE(**kw).to(device)


def load_checkpoint_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    set_log_e_stats(ckpt['log_e_mean'], ckpt['log_e_std'])
    model = fresh_model(device, Nz=ckpt['Nz'], Nphi=ckpt['Nphi'], Nr=ckpt['Nr'],
                        latent_channels=ckpt['latent_channels'],
                        latent_length=ckpt['latent_length']).to(device)
    model.load_state_dict(ckpt['model_state'])
    return model, ckpt


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ==============================================================================
# Test 1: can the model even memorize a tiny fixed batch?
# ==============================================================================

def test_overfit_tiny_batch(args, showers, energies_gev):
    hr("TEST 1: overfit_tiny_batch -- is the training loop even capable of "
       "driving loss to ~0 on a fixed handful of examples?")
    n = 16
    x_raw = showers[:n].to(args.device)
    energy = energies_gev[:n].to(args.device)
    set_log_e_stats(torch.log10(energy + EPS).mean().item(), torch.log10(energy + EPS).std().item())

    torch.manual_seed(0)
    model = fresh_model(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    x_log = physical_to_model(x_raw, energy)
    rel_trace = []
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        x_hat_log, _ = model(x_log, energy)
        loss, *_ = calorimeter_loss(x_hat_log, x_log, energy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            with torch.no_grad():
                rel = relative_error_phys(x_log, x_hat_log)
            rel_trace.append((step, loss.item(), rel))
            print(f"  step {step:4d}  loss={loss.item():.5f}  rel_err={rel:.4f}")

    final_rel = rel_trace[-1][2]
    print(f"\n  Final relative error on the memorized batch: {final_rel:.4f}")
    if final_rel < 0.05:
        print("  VERDICT: PASS -- the model can memorize this batch essentially "
              "perfectly. The plateau on full data is NOT a broken-gradient or "
              "dead-architecture bug; it's a generalization/optimization-at-scale "
              "issue (see the other tests).")
    elif final_rel < 0.2:
        print("  VERDICT: PARTIAL -- meaningful memorization but not clean. Some "
              "mechanism is limiting how far loss can go even on 16 fixed "
              "examples. Worth cross-checking against loss_term_balance.")
    else:
        print("  VERDICT: FAIL -- the model can't even memorize 16 examples. "
              "This points at a real bug in the training mechanics themselves "
              "(gradient flow, loss balance, or a numerical issue), not a "
              "generalization limit. Re-run with --device cuda and compare, "
              "and check loss_term_balance / amp_scaler_skips next.")
    return {"final_rel_err": final_rel, "trace": rel_trace}


# ==============================================================================
# Test 2: does the decoder actually use the latent, or lean on energy alone?
# ==============================================================================

def test_latent_ablation(args, showers, energies_gev):
    hr("TEST 2: latent_ablation -- swap out z (keeping the true energy) and "
       "see how much reconstruction actually degrades")
    model, ckpt = load_checkpoint_model(args.checkpoint, args.device)
    model.eval()
    n = min(512, showers.shape[0])
    x_raw = showers[:n].to(args.device)
    energy = energies_gev[:n].to(args.device)
    x_log = physical_to_model(x_raw, energy)

    with torch.no_grad():
        z_true = model.encode(x_log, energy)
        x_hat_true = model.decode(z_true, energy)
        rel_true = relative_error_phys(x_log, x_hat_true)

        z_zero = torch.zeros_like(z_true)
        x_hat_zero = model.decode(z_zero, energy)
        rel_zero = relative_error_phys(x_log, x_hat_zero)

        perm = torch.randperm(n, device=args.device)
        z_shuffled = z_true[perm]
        x_hat_shuf = model.decode(z_shuffled, energy)
        rel_shuf = relative_error_phys(x_log, x_hat_shuf)

        z_mean = z_true.mean(dim=0, keepdim=True).expand_as(z_true)
        x_hat_mean = model.decode(z_mean, energy)
        rel_mean = relative_error_phys(x_log, x_hat_mean)

    print(f"  rel_err with TRUE z              : {rel_true:.4f}")
    print(f"  rel_err with z = zeros           : {rel_zero:.4f}  (+{100*(rel_zero-rel_true)/rel_true:.1f}%)")
    print(f"  rel_err with z = batch mean       : {rel_mean:.4f}  (+{100*(rel_mean-rel_true)/rel_true:.1f}%)")
    print(f"  rel_err with z = shuffled (other sample's z): {rel_shuf:.4f}  (+{100*(rel_shuf-rel_true)/rel_true:.1f}%)")

    worst_degradation = max(rel_zero, rel_mean, rel_shuf) / rel_true - 1
    if worst_degradation < 0.15:
        print("\n  VERDICT: LATENT LARGELY UNUSED -- replacing z with zeros/mean/"
              "another sample's z barely hurts reconstruction. The decoder is "
              "mostly reconstructing an energy-conditioned average shower and "
              "ignoring per-event latent detail (bottleneck collapse). This "
              "would explain an early, stubborn plateau: once the model learns "
              "the 'average shape at this energy' solution, there's little "
              "gradient pressure left to refine z further under the current "
              "loss balance.")
    elif worst_degradation < 0.6:
        print("\n  VERDICT: PARTIAL -- z matters somewhat but a shuffled/zeroed "
              "z doesn't fully break reconstruction. Some latent underuse, "
              "not full collapse.")
    else:
        print("\n  VERDICT: LATENT IS LOAD-BEARING -- ablating z clearly wrecks "
              "reconstruction. Bottleneck collapse is NOT the explanation here; "
              "look at loss_term_balance / units_target_range instead.")
    return {"rel_true": rel_true, "rel_zero": rel_zero, "rel_mean": rel_mean, "rel_shuf": rel_shuf}


# ==============================================================================
# Test 3: which loss term actually drives the gradient?
# ==============================================================================

def test_loss_term_balance(args, showers, energies_gev):
    hr("TEST 3: loss_term_balance -- gradient norm contributed by each loss "
       "term separately, on the same batch")
    model, ckpt = load_checkpoint_model(args.checkpoint, args.device)
    model.train()
    # Kept small on purpose: the B*Nz-reshape-into-batch trick (see encode/
    # decode) turns n=256 into 256*45=11520 images through the 2D conv stack.
    # A single retain_graph=True forward+backward chain over 5 loss terms
    # OOM'd a 3.63GB card at n=256 even alone -- each term now gets its own
    # forward+backward (a bit more compute, no graph retained across terms,
    # so peak memory is one term's worth, not five held simultaneously).
    n = min(128, showers.shape[0])
    x_raw = showers[:n].to(args.device)
    energy = energies_gev[:n].to(args.device)
    x_log = physical_to_model(x_raw, energy)

    term_weights = {
        "recon_mse (weight 1.0)": ("recon_mse", 1.0),
        "recon_l1  (weight 1.0, incl. 1+10*x_log per-voxel factor)": ("recon_l1", 1.0),
        "hit_loss  (weight 0.02)": ("hit_loss", 0.02),
        "energy_loss (weight 0.05)": ("energy_loss", 0.05),
        "phys_loss (weight 0.1)": ("phys_loss", 0.1),
    }
    component_names = ["recon_mse", "recon_l1", "hit_loss", "energy_loss", "phys_loss"]

    terms = {}
    grad_norms = {}
    for label, (component, weight) in term_weights.items():
        model.zero_grad(set_to_none=True)
        x_hat_log, _ = model(x_log, energy)
        recon_mse = F.mse_loss(x_hat_log, x_log)
        w = 1.0 + 10.0 * x_log
        recon_l1 = (w * (x_hat_log - x_log).abs()).mean()
        target_hit = (x_log > 1e-4).float()
        hit_loss = F.binary_cross_entropy_with_logits(x_hat_log, target_hit)
        x_phys = torch.clamp(torch.expm1(x_log), min=0.0)
        xhat_phys = torch.clamp(torch.expm1(x_hat_log), min=0.0)
        E_true = x_phys.sum(dim=(1, 2, 3))
        E_pred = xhat_phys.sum(dim=(1, 2, 3))
        energy_loss = (torch.abs(E_true - E_pred) / (E_true + EPS)).mean()
        phys_loss = F.mse_loss(xhat_phys, torch.expm1(x_log))
        raw = {"recon_mse": recon_mse, "recon_l1": recon_l1, "hit_loss": hit_loss,
               "energy_loss": energy_loss, "phys_loss": phys_loss}
        term = weight * raw[component]
        terms[label] = term
        term.backward()
        total_norm = math.sqrt(sum(
            p.grad.detach().pow(2).sum().item()
            for p in model.parameters() if p.grad is not None
        ))
        grad_norms[label] = total_norm
        del x_hat_log, recon_mse, recon_l1, hit_loss, energy_loss, phys_loss, raw, term
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"  {'term':55s} {'raw value':>12s} {'grad norm':>12s}")
    for name in terms:
        print(f"  {name:55s} {terms[name].item():12.5f} {grad_norms[name]:12.5f}")

    biggest = max(grad_norms, key=grad_norms.get)
    smallest = min(grad_norms, key=grad_norms.get)
    ratio = grad_norms[biggest] / max(grad_norms[smallest], 1e-12)
    print(f"\n  Largest gradient contributor : {biggest} ({grad_norms[biggest]:.5f})")
    print(f"  Smallest gradient contributor: {smallest} ({grad_norms[smallest]:.5f})")
    print(f"  Ratio largest/smallest       : {ratio:.1f}x")
    if ratio > 20:
        print("\n  VERDICT: SEVERELY IMBALANCED -- one term dominates the total "
              "gradient by >20x. Whatever the smallest term is trying to teach "
              "the model (sparsity, absolute energy, physical-space accuracy) "
              "is effectively drowned out no matter its nominal loss weight. "
              "This directly supports raising that term's weight and/or fixing "
              "the units convention that inflates recon_l1's 1+10*x_log factor.")
    else:
        print("\n  VERDICT: Reasonably balanced -- no single term is drowning "
              "out the others by an order of magnitude.")
    return grad_norms


# ==============================================================================
# Test 4: is the sparsity (hit/no-hit) term stuck at the trivial floor?
# ==============================================================================

def test_hit_loss_floor(args, showers, energies_gev):
    hr("TEST 4: hit_loss_floor -- is the BCE sparsity term stuck at the "
       "trivial 'always predict 0.5' floor of ln(2)=0.6931?")
    model, ckpt = load_checkpoint_model(args.checkpoint, args.device)
    model.eval()
    n = min(512, showers.shape[0])
    x_raw = showers[:n].to(args.device)
    energy = energies_gev[:n].to(args.device)
    x_log = physical_to_model(x_raw, energy)

    with torch.no_grad():
        x_hat_log, _ = model(x_log, energy)
        target_hit = (x_log > 1e-4).float()
        actual_hit_loss = F.binary_cross_entropy_with_logits(x_hat_log, target_hit).item()

        trivial_logits = torch.zeros_like(x_hat_log)
        trivial_hit_loss = F.binary_cross_entropy_with_logits(trivial_logits, target_hit).item()

        oracle_logits = torch.where(target_hit > 0.5, torch.full_like(x_hat_log, 10.0),
                                    torch.full_like(x_hat_log, -10.0))
        oracle_hit_loss = F.binary_cross_entropy_with_logits(oracle_logits, target_hit).item()

        pred_hit = (torch.sigmoid(x_hat_log) > 0.5).float()
        agreement = (pred_hit == target_hit).float().mean().item()
        sparsity = 1.0 - target_hit.mean().item()

    print(f"  dataset sparsity (frac. voxels below hit threshold): {sparsity:.4f}")
    print(f"  model's actual hit_loss                              : {actual_hit_loss:.4f}")
    print(f"  trivial baseline (logit=0 everywhere) hit_loss       : {trivial_hit_loss:.4f}  (=ln 2)")
    print(f"  oracle (perfect) hit_loss                            : {oracle_hit_loss:.4f}")
    print(f"  model's hit/no-hit classification accuracy           : {agreement:.4f}")

    closeness = 1 - abs(actual_hit_loss - trivial_hit_loss) / trivial_hit_loss
    if closeness > 0.9:
        print("\n  VERDICT: AT THE TRIVIAL FLOOR -- the model has learned "
              "essentially nothing about which voxels are hit vs. empty; its "
              "logits are barely more informative than always predicting 0.5. "
              "Combined with loss_term_balance, this is consistent with "
              "hit_weight=0.02 being too small for this signal to matter, or "
              "the recon_l1 term already implicitly handling zeros well enough "
              "that this term never gets meaningful gradient priority.")
    else:
        print("\n  VERDICT: the model has learned real hit/no-hit structure "
              "above the trivial baseline.")
    return {"actual": actual_hit_loss, "trivial": trivial_hit_loss, "oracle": oracle_hit_loss,
            "accuracy": agreement}


# ==============================================================================
# Test 5: how much does the mixed-units convention stretch the target range?
# ==============================================================================

def test_units_target_range(args, showers, energies_gev):
    hr("TEST 5: units_target_range -- current (MeV shower / GeV energy) vs. "
       "same-units (GeV / GeV) log1p target range")
    n = min(5000, showers.shape[0])
    x_mev = showers[:n]
    e_gev = energies_gev[:n]

    x_log_current = physical_to_model(x_mev, e_gev)              # showers left in MeV (the bug-prone convention)
    x_log_fixed = physical_to_model(x_mev / 1000.0, e_gev)        # showers converted to GeV too

    def stats(name, t):
        nz = t[t > 1e-4]
        w = 1.0 + 10.0 * t
        print(f"  {name}:")
        print(f"    x_log        min/mean/max over nonzero: "
              f"{nz.min().item():.4f} / {nz.mean().item():.4f} / {nz.max().item():.4f}")
        print(f"    1+10*x_log L1 weight  mean/max         : "
              f"{w.mean().item():.4f} / {w.max().item():.4f}")

    stats("CURRENT (shower in MeV / energy in GeV)", x_log_current)
    stats("SAME-UNITS (shower in GeV / energy in GeV)", x_log_fixed)

    ratio = x_log_current.max().item() / max(x_log_fixed.max().item(), 1e-8)
    print(f"\n  Current target range is {ratio:.1f}x wider than the same-units "
          f"convention would give.")
    print("  VERDICT: this is a quantified, reproducible measurement, not a "
          "pass/fail test -- feed it into loss_term_balance's interpretation: "
          "a wider target range directly inflates the 1+10*x_log weighting "
          "used in recon_l1.")
    return {"current_max": x_log_current.max().item(), "fixed_max": x_log_fixed.max().item()}


# ==============================================================================
# Test 6: how hard is grad clipping actually engaging?
# ==============================================================================

def test_grad_clip_engagement(args, showers, energies_gev):
    hr("TEST 6: grad_clip_engagement -- how often does clip_grad_norm_(...,1.0) "
       "actually reduce the gradient, and by how much?")
    model, ckpt = load_checkpoint_model(args.checkpoint, args.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    set_log_e_stats(ckpt['log_e_mean'], ckpt['log_e_std'])

    n_steps = args.steps
    batch = 64
    pre_clip_norms = []
    for step in range(n_steps):
        idx = torch.randint(0, showers.shape[0], (batch,))
        x_raw = showers[idx].to(args.device)
        energy = energies_gev[idx].to(args.device)
        x_log = physical_to_model(x_raw, energy)

        opt.zero_grad(set_to_none=True)
        x_hat_log, _ = model(x_log, energy)
        loss, *_ = calorimeter_loss(x_hat_log, x_log, energy)
        loss.backward()
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        pre_clip_norms.append(pre_clip_norm.item())
        opt.step()

    pre_clip_norms = np.array(pre_clip_norms)
    frac_clipped = (pre_clip_norms > 1.0).mean()
    print(f"  steps sampled                : {n_steps}")
    print(f"  pre-clip grad norm  mean/median/max: "
          f"{pre_clip_norms.mean():.3f} / {np.median(pre_clip_norms):.3f} / {pre_clip_norms.max():.3f}")
    print(f"  fraction of steps where clipping actually engaged (norm > 1.0): {frac_clipped:.1%}")
    if frac_clipped > 0.5:
        print("\n  VERDICT: CLIPPING IS ALMOST ALWAYS ACTIVE -- the model wants "
              "to take much bigger steps than grad_clip=1.0 allows, on the "
              "majority of iterations. This artificially throttles the "
              "effective learning rate and could easily be why training "
              "crawls to a stop early: try grad_clip=2-5 or removing it "
              "entirely for a short comparison run.")
    elif frac_clipped > 0.1:
        print("\n  VERDICT: Clipping engages occasionally -- not the dominant "
              "bottleneck, but worth a quick ablation anyway since it's cheap "
              "to test.")
    else:
        print("\n  VERDICT: Clipping rarely engages -- not a likely cause of "
              "the plateau.")
    return {"frac_clipped": float(frac_clipped), "mean_norm": float(pre_clip_norms.mean())}


# ==============================================================================
# Test 7: is AMP silently skipping optimizer steps?
# ==============================================================================

def test_amp_scaler_skips(args, showers, energies_gev):
    hr("TEST 7: amp_scaler_skips -- does mixed precision silently skip "
       "optimizer.step() on detected overflow, and does turning AMP off "
       "change the short-run trajectory?")
    if args.device != 'cuda':
        print("  AMP is a CUDA-only path (train_autoencoder disables it on CPU "
              "automatically) -- skipping; re-run with --device cuda to test this.")
        return {"skipped": True}

    def run(use_amp, n_steps):
        torch.manual_seed(1)
        model = fresh_model(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        losses, skips = [], 0
        for step in range(n_steps):
            idx = torch.randint(0, showers.shape[0], (64,))
            x_raw = showers[idx].to(args.device)
            energy = energies_gev[idx].to(args.device)
            x_log = physical_to_model(x_raw, energy)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                x_hat_log, _ = model(x_log, energy)
                loss, *_ = calorimeter_loss(x_hat_log, x_log, energy)
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skips += 1
            losses.append(loss.item())
        return losses, skips

    set_log_e_stats(torch.log10(energies_gev + EPS).mean().item(),
                    torch.log10(energies_gev + EPS).std().item())
    losses_amp, skips = run(True, args.steps)
    losses_fp32, _ = run(False, args.steps)

    print(f"  AMP steps sampled       : {args.steps}")
    print(f"  AMP steps skipped (overflow detected): {skips} ({skips/args.steps:.1%})")
    print(f"  loss after {args.steps} steps, AMP  : {losses_amp[-1]:.5f}")
    print(f"  loss after {args.steps} steps, FP32 : {losses_fp32[-1]:.5f}")
    if skips / args.steps > 0.05:
        print("\n  VERDICT: AMP IS SKIPPING A MEANINGFUL FRACTION OF STEPS -- "
              "worth trying a from-scratch run with use_amp=False, or fixing "
              "the units convention first (a narrower value range is less "
              "likely to overflow fp16).")
    else:
        print("\n  VERDICT: skip rate is low -- AMP overflow skipping is "
              "unlikely to be the main driver of the plateau, though compare "
              "the two loss values above for any other divergence.")
    return {"skips": skips, "loss_amp": losses_amp[-1], "loss_fp32": losses_fp32[-1]}


# ==============================================================================
# Test 8: LR sensitivity from identical random init
# ==============================================================================

def test_lr_sensitivity(args, showers, energies_gev):
    hr("TEST 8: lr_sensitivity -- from the SAME random init, does a lower "
       "constant LR make more progress in an equal step budget than the "
       "current 3e-4?")
    set_log_e_stats(torch.log10(energies_gev + EPS).mean().item(),
                    torch.log10(energies_gev + EPS).std().item())
    lrs = [1e-5, 1e-4, 3e-4, 1e-3]
    n_steps = args.steps
    batch = 64
    results = {}
    for lr in lrs:
        torch.manual_seed(2)
        model = fresh_model(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        torch.manual_seed(123)  # identical batch sequence across LRs
        last_loss, last_rel = None, None
        for step in range(n_steps):
            idx = torch.randint(0, showers.shape[0], (batch,))
            x_raw = showers[idx].to(args.device)
            energy = energies_gev[idx].to(args.device)
            x_log = physical_to_model(x_raw, energy)
            opt.zero_grad(set_to_none=True)
            x_hat_log, _ = model(x_log, energy)
            loss, *_ = calorimeter_loss(x_hat_log, x_log, energy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_loss = loss.item()
            if step == n_steps - 1:
                last_rel = relative_error_phys(x_log, x_hat_log)
        results[lr] = (last_loss, last_rel)
        print(f"  lr={lr:<8g}  loss after {n_steps} steps = {last_loss:.5f}   rel_err = {last_rel:.4f}")

    best_lr = min(results, key=lambda k: results[k][0])
    print(f"\n  Lowest loss after {n_steps} steps: lr={best_lr}")
    if best_lr < 3e-4:
        print("  VERDICT: a LOWER learning rate makes more progress in the same "
              "step budget than the current 3e-4 -- consistent with the "
              "current LR being too high to settle into a minimum once close, "
              "which combined with having no LR decay (now fixed by enabling "
              "the cosine scheduler) would produce exactly this kind of early, "
              "stubborn plateau.")
    elif best_lr == 3e-4:
        print("  VERDICT: 3e-4 is already the best of the LRs tried in this "
              "short budget -- LR magnitude alone doesn't look like the "
              "culprit; the cosine schedule (decaying FROM 3e-4) should still "
              "help with later-epoch fine convergence regardless.")
    else:
        print("  VERDICT: a HIGHER learning rate makes more progress -- the "
              "current LR may be too conservative early on; worth trying a "
              "higher peak LR under the cosine schedule.")
    return results


# ==============================================================================
# Runner
# ==============================================================================

TESTS = {
    "overfit_tiny_batch": test_overfit_tiny_batch,
    "latent_ablation": test_latent_ablation,
    "loss_term_balance": test_loss_term_balance,
    "hit_loss_floor": test_hit_loss_floor,
    "units_target_range": test_units_target_range,
    "grad_clip_engagement": test_grad_clip_engagement,
    "amp_scaler_skips": test_amp_scaler_skips,
    "lr_sensitivity": test_lr_sensitivity,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/Dataset2.pt")
    p.add_argument("--checkpoint", default="best_calo_ae_cond_improved.pt",
                   help="Checkpoint used by tests that need a trained model "
                        "(latent_ablation, loss_term_balance, hit_loss_floor, "
                        "grad_clip_engagement).")
    p.add_argument("--device", default="cpu",
                   help="Defaults to CPU on purpose -- these tests are small "
                        "and this avoids contending with any GPU job you have "
                        "running. Pass --device cuda to use the GPU.")
    p.add_argument("--steps", type=int, default=200,
                   help="Step budget for the training-loop-based tests "
                        "(overfit_tiny_batch, grad_clip_engagement, "
                        "amp_scaler_skips, lr_sensitivity).")
    p.add_argument("--n-data", type=int, default=8000,
                   help="How many showers to load from --data (loading the "
                        "full 100k is slow and unnecessary for these tests).")
    p.add_argument("--test", nargs="+", default=["all"],
                   choices=list(TESTS.keys()) + ["all"])
    p.add_argument("--list", action="store_true", help="List available tests and exit.")
    args = p.parse_args()

    if args.list:
        for name, fn in TESTS.items():
            print(f"  {name:22s} {fn.__doc__ or ''}")
        return

    print(f"device={args.device}  steps={args.steps}  n_data={args.n_data}  "
          f"checkpoint={args.checkpoint}")
    t0 = time.time()
    print(f"Loading {args.n_data} showers from {args.data} ...")
    showers, energies_gev = load_data(args.data, n=args.n_data, device='cpu')
    print(f"Loaded in {time.time()-t0:.1f}s")

    tests_to_run = list(TESTS.keys()) if "all" in args.test else args.test
    results = {}
    for name in tests_to_run:
        t1 = time.time()
        try:
            results[name] = TESTS[name](args, showers, energies_gev)
        except Exception as e:
            print(f"\n  TEST {name} RAISED AN EXCEPTION: {e!r}")
            results[name] = {"error": str(e)}
        # Each test builds its own model(s)/optimizer(s); on a small GPU these
        # can accumulate across tests faster than Python's GC reclaims them
        # (seen in practice: loss_term_balance OOM'd on a 3.63GB card after
        # overfit_tiny_batch's 1500-step run, with plenty of memory that
        # should have been free). Force cleanup between tests.
        import gc
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print(f"  ({name} took {time.time()-t1:.1f}s)")

    hr("SUMMARY")
    for name in tests_to_run:
        print(f"  {name:22s} -> {results[name]}")

    total = time.time() - t0
    print(f"\nAll done in {total/60:.1f} minutes.")


if __name__ == "__main__":
    main()
