"""
ae_losses.py -- autoencoder objectives and reconstruction metrics.

Three generations, kept together because the differences between them are the
project's main result on this model:

  calorimeter_loss     v3. log-space MSE + L1, a hit BCE sharing the value
                       output, and an energy term in physical space. Measured
                       to reconstruct WORSE than the energy-binned mean shower
                       (rel 0.688 vs 0.638; PCA-96 gets ~0.60). The flat
                       validation curve was the objective, not the capacity.
                       Kept only so pre-v4 checkpoints remain interpretable.

  calorimeter_loss_v4  rel^2 (the square of the reported metric, so objective
                       and metric agree) + 0.1 * BCE on a SEPARATE hit head +
                       an energy term on the SOFT-GATED prediction. rel 0.367,
                       beating PCA-512 with 96 numbers.

  calorimeter_loss_v5  the focal hurdle objective adapted from the NOvA sparse
                       CAE report. v4's reconstruction over-paints -- precision
                       0.54 against recall 0.92, 2.3x too many lit voxels --
                       and plain/weighted cross-entropy fixes class imbalance
                       by DISTORTING probabilities. Focal loss down-weights
                       what is already easy instead. See docs/NOVA_TRANSFER.md.

tune_threshold picks the read-out cut on validation rather than leaving it at
0.5, which matters especially for v5: focal loss is not proper, so sigmoid(h)
is not a calibrated occupancy probability.
"""

import math

import torch
import torch.nn.functional as F


def calorimeter_loss(
    x_hat_log,
    x_log,
    energy,
    mse_weight=1.0,
    l1_weight=1.0,
    energy_weight=0.1,
    hit_weight=0.3,
    phys_weight=0.1,
    rel_weight=0.05,
    hit_threshold=1e-4,
    hit_logit_scale=1.0,
):
    # ---- input contract ----
    assert x_hat_log.shape == x_log.shape, (
        f"calorimeter_loss: x_hat_log {tuple(x_hat_log.shape)} != "
        f"x_log {tuple(x_log.shape)}"
    )
    assert x_log.dim() == 4, (
        f"calorimeter_loss: expected 4D tensors, got {tuple(x_log.shape)}"
    )
    # Sanity check on the log-space target range. log1p(x/E) of normalized
    # showers is typically in [0, ~few]. Wildly large values usually mean a
    # missing /E normalization upstream.
    if not torch.isfinite(x_log).all():
        raise ValueError("calorimeter_loss: x_log contains NaN or Inf")
    if not torch.isfinite(x_hat_log).all():
        raise ValueError("calorimeter_loss: x_hat_log contains NaN or Inf")

    # ----------------------------------------------------------------------
    # 1. Hybrid reconstruction loss in log space
    # ----------------------------------------------------------------------
    recon_mse = F.mse_loss(x_hat_log, x_log)
    weights = 1.0 + 10.0 * x_log
    recon_l1 = (weights * (x_hat_log - x_log).abs()).mean()
    recon_loss = mse_weight * recon_mse + l1_weight * recon_l1

    # ----------------------------------------------------------------------
    # 2. Hit / no-hit BCE
    #
    # We need a "lit-ness" probability for each voxel. To stay numerically
    # stable under fp16 autocast, we produce LOGITS (unbounded reals) and
    # use binary_cross_entropy_with_logits, which fuses sigmoid + BCE in a
    # stable way. Plain F.binary_cross_entropy is rejected by autocast.
    #
    # The logit we feed in is x_hat_log itself, scaled by hit_logit_scale:
    # when the scaled logit is large positive (lit voxel), sigmoid ~ 1;
    # when it's <= 0 (empty voxel in log-space), sigmoid <= 0.5. This is a
    # soft, smooth proxy for "is this voxel lit", sharing gradient signal
    # with the reconstruction loss. hit_logit_scale exists because
    # x_hat_log's own natural magnitude (tiny under the GeV/GeV units
    # convention) isn't by itself wide enough to express real confidence --
    # see the note on FactorizedCaloAE.hit_logit_scale.
    # ----------------------------------------------------------------------
    target_hit = (x_log > hit_threshold).float()
    hit_loss = F.binary_cross_entropy_with_logits(x_hat_log * hit_logit_scale, target_hit)

    # ----------------------------------------------------------------------
    # 3. Energy conservation in PHYSICAL space (per-shower normalized energy)
    # ----------------------------------------------------------------------
    x_phys = torch.clamp(torch.expm1(x_log), min=0.0)
    xhat_phys = torch.clamp(torch.expm1(x_hat_log), min=0.0)

    E_true = x_phys.sum(dim=(1, 2, 3))
    E_pred = xhat_phys.sum(dim=(1, 2, 3))
    energy_loss = (torch.abs(E_true - E_pred) / (E_true + 1e-8)).mean()
	
    # Add small physical-space term
    x_phys_target = torch.expm1(x_log)
    xhat_phys = torch.clamp(torch.expm1(x_hat_log), min=0.0)
    phys_loss = F.mse_loss(xhat_phys, x_phys_target)

    # ----------------------------------------------------------------------
    # 4. Per-shower relative error, made differentiable (rel_loss)
    #
    # Checked empirically before adding this (not assumed): binned real
    # showers by incident energy and measured phys_loss per bin -- it came
    # back essentially flat (~2.3-2.5e-5 across 1 GeV to 1000 GeV), so
    # phys_loss is NOT dominated by high-energy showers the way the original
    # units bug dominated voxel-space. What IS strongly energy-dependent is
    # rel_err itself: 322% at 1-3 GeV vs 1321% at 500-1000 GeV in that same
    # check, despite near-identical absolute per-voxel error. The reason:
    # a high-energy shower spreads its (fractional) deposit over far more
    # voxels, so each individual voxel's true value is smaller -- the same
    # constant absolute error is a much larger relative one there. Neither
    # recon_l1's brightness weighting (fixed, energy-independent) nor
    # phys_loss (unweighted) accounts for a shower's own characteristic
    # scale, which is exactly what rel_err (the metric actually reported
    # and cared about) measures. Rather than hope the existing proxies
    # keep correlating with it, put it in the objective directly -- same
    # formula as relative_error_phys(), just differentiable through
    # x_hat_log instead of computed under torch.no_grad().
    # ----------------------------------------------------------------------
    B = x_log.shape[0]
    true_flat = x_phys_target.reshape(B, -1)
    pred_flat = xhat_phys.reshape(B, -1)
    rel_loss = (torch.norm(pred_flat - true_flat, dim=1) /
                (torch.norm(true_flat, dim=1) + 1e-8)).mean()

    # ----------------------------------------------------------------------
    # Total
    # ----------------------------------------------------------------------
    # Previously hardcoded to 0.02/0.05/0.1 regardless of the hit_weight/
    # energy_weight arguments above -- those parameters were silently dead.
    # diagnose_training_plateau.py's loss_term_balance test measured the
    # gradient norm each term actually contributes on a real batch at those
    # old weights: recon_l1 ~0.25, energy_loss ~0.14, but hit_loss only
    # ~0.007 -- a 34x gap. hit_weight raised 0.02 -> 0.3 to close most of
    # that gap (hit_loss_floor test showed the model hadn't learned hit/miss
    # structure beyond the trivial ln(2) baseline). energy_weight nudged
    # 0.05 -> 0.1; phys_loss's weight is now the named phys_weight parameter
    # instead of a bare literal. rel_weight=0.05 was calibrated (not
    # guessed) against the current checkpoint via the same gradient-norm
    # methodology as diagnose_training_plateau.py's loss_term_balance test:
    # at weight=0.5 rel_loss's gradient norm was ~168, dwarfing every other
    # term (recon_l1 ~8.5, hit_loss ~10.1, energy_loss ~13.4, phys_loss
    # ~0.0002) by 13-1,000,000x -- the exact same class of imbalance that
    # starved hit_loss originally, just relocated. 0.05 puts rel_loss's
    # gradient (~16.8) just above energy_loss's, a deliberate modest edge
    # since it directly targets the metric that's been stuck, without
    # swamping the other terms the way an unexamined default would have.
    total = (recon_loss + hit_weight * hit_loss + energy_weight * energy_loss
             + phys_weight * phys_loss + rel_weight * rel_loss)
    assert torch.isfinite(total), (
        f"calorimeter_loss: total loss is non-finite "
        f"(recon={recon_loss.item()}, hit={hit_loss.item()}, "
        f"E={energy_loss.item()}, rel_loss={rel_loss.item()})"
    )

    return total, recon_loss, hit_loss, energy_loss


def calorimeter_loss_v4(v, h, x_log, x_scale, hit_weight=0.1, energy_weight=1.0):
    """
    v4 objective, for the two-head model. Replaces calorimeter_loss above,
    which is kept only for pre-v4 checkpoints/scripts.

    Why (all measured on the v3 checkpoint, val showers, before changing it):
      * v3 reconstructed WORSE than the energy-binned mean shower (rel 0.688
        vs 0.638; linear PCA with the same 96 numbers gets 0.539). The flat
        val curve was the objective, not capacity.
      * hit BCE shared the value output (fixed by the separate head h).
        It was 63% of the loss and stuck at 0.496, worse than a constant
        predictor (0.361).
      * the energy-sum L1 had the largest gradient (3x the rel term) and
        cosine 0.015 with it -- it moved every voxel together, carrying no
        shape information. phys_loss was 5.9e-7: inert.
      * log1p(E/E_inc) is nearly linear with values ~1e-4, so log-space
        MSE/L1 barely saw shower shape.

    Terms:
      rel2: per-shower squared relative L2 error in energy-fraction space --
            the square of the rel metric that is reported, so the objective
            and the metric agree. Scale-free, so tiny fractions don't matter.
      bce:  hit head vs "voxel nonzero" (the data is already cut at
            15.15 keV), so the gate reproduces the real sparsity.
      e2:   squared relative total-energy error, small weight; without it
            the energy error rose to 0.27 (v3: 0.12).
    Returns (total, rel2, bce, e2).
    """
    B = x_log.shape[0]
    true = torch.expm1(x_log.float()).reshape(B, -1)
    pred = torch.expm1(v.float() * x_scale).reshape(B, -1)
    rel2 = ((pred - true).pow(2).sum(1) / (true.pow(2).sum(1) + 1e-12)).mean()
    bce = F.binary_cross_entropy_with_logits(h.float(), (x_log > 0).float())
    e_true = true.sum(1)
    # Energy of the SOFT-gated prediction, value * P(hit): the reported output
    # zeros voxels where the hit head says empty, so an energy term on the
    # ungated values optimizes energy the gate then removes (measured: with
    # ungated e2 the gated energy error got worse in epoch 1, 0.29 -> 0.37).
    p_e = pred.clamp(min=0) * torch.sigmoid(h.float()).reshape(B, -1)
    e2 = ((p_e.sum(1) - e_true) / (e_true + 1e-12)).pow(2).mean()
    total = rel2 + hit_weight * bce + energy_weight * e2
    assert torch.isfinite(total), (
        f"calorimeter_loss_v4: non-finite loss (rel2={rel2.item()}, "
        f"bce={bce.item()}, e2={e2.item()})")
    return total, rel2, bce, e2


def calorimeter_loss_v5(v, h, x_log, x_scale, thr_log,
                        gamma=2.0, alpha_hit=0.25, lam=1.0, energy_weight=1.0):
    """
    v5: the hurdle objective with a FOCAL hit term, adapted from the NOvA
    sparse-CAE report (Prosper, Genesis Mission hackathon, Oct 2026) which
    solved the same problem on 99.4%-empty pixel maps.

    Why, measured on our own v4-architecture reconstruction (quick_metrics
    --paired): lit fraction 0.42 against a real 0.245, with recall 0.92 but
    precision 0.54 and 2.33x too many hits. That is OVER-PAINTING -- the model
    finds nearly every real hit and then draws a halo of false ones. The lit
    FRACTION alone could never show this, which is why it went unnoticed.

    v4's hit term is a plain BCE. Plain/weighted cross-entropy fixes class
    imbalance by DISTORTING the probabilities: for a hit weight w the optimal
    prediction crosses 0.5 at true probability q > 1/(w+1), so every voxel with
    even a small chance of being lit is declared lit. The NOvA report measured
    exactly this -- 2.68x too many hits with full-rarity weighting -- and that
    number is strikingly close to our 2.33x.

    Focal loss instead DOWN-WEIGHTS what is already easy:

        FL = -alpha_t * (1 - p_t)^gamma * log p_t

    where p_t is the probability assigned to the CORRECT answer. A voxel the
    model already gets right at p_t = 0.9 has its loss cut 100x, so the mass of
    obviously-empty voxels falls silent while the uncertain ones near the shower
    edge keep full weight. Crucially it does NOT shift an uncertain voxel's
    prediction toward "lit", which is the whole difference from re-weighting.

    Differences from the NOvA setup, and why:
      * They sum the hit term over all pixels and divide by the number of hits,
        so the scale is independent of how many empty pixels there are. Kept.
      * Their charge term is a squared error on log(1+y) over TRUE HITS ONLY --
        an empty pixel has no charge to predict, and including it drags the
        prediction toward zero (the sparsity problem again). v4's rel^2 runs
        over ALL voxels; here the charge term is restricted to true hits, which
        is the change their section 4.5 argues for.
      * Their class imbalance is 171:1; ours is only 3:1 (75% empty). So the
        motivation is weaker here and this must be A/B'd against v4, not
        assumed better. See docs/NOVA_TRANSFER.md.
      * alpha = 0.25 on the RARE class looks backwards but is what the original
        paper found best: once gamma has muted the easy majority, a small alpha
        on the rare class works better.

    A consequence to remember at read-out time: focal loss is NOT proper, so
    sigmoid(h) is not a calibrated occupancy probability. It is pulled toward
    0.5 and then shifted by alpha. The decision threshold must therefore be
    TUNED on validation rather than left at 0.5 -- the NOvA report found
    tau = 0.35-0.40. v4's decode() hard-codes h > 0, i.e. tau = 0.5.

    Returns (total, focal, charge, e2) so the three terms can be logged apart.
    """
    B = x_log.shape[0]
    hit = (x_log > thr_log).float()                       # true occupancy
    n_hit = hit.sum().clamp(min=1.0)

    # focal term, summed over ALL voxels, normalized by the number of hits
    logits = h.float()
    p = torch.sigmoid(logits)
    p_t = p * hit + (1.0 - p) * (1.0 - hit)
    alpha_t = alpha_hit * hit + (1.0 - alpha_hit) * (1.0 - hit)
    log_p_t = -F.binary_cross_entropy_with_logits(
        logits, hit, reduction="none")                     # = log p_t, stable
    focal = -(alpha_t * (1.0 - p_t).pow(gamma) * log_p_t).sum() / n_hit

    # Charge term on TRUE HITS ONLY, in the SCALED log space (both sides
    # divided by x_scale, so both are O(1)).
    #
    # Do NOT write this as (v*x_scale - x_log)^2: x_log is ~1e-4 here, so that
    # term is ~1e-8 and is invisible next to the focal term -- which is exactly
    # the v3 defect this file's history documents ("log1p(E/E_inc) is nearly
    # linear with values ~1e-4, so log-space MSE/L1 barely saw shower shape").
    # The NOvA report does not hit this because its charges are 1-165 ADC, so
    # log(1+y) is already O(1). Using log(1 + E/MeV) to get the same effect
    # would reintroduce the Jensen bias that made our absolute-log
    # representation undershoot the energy by 2.4x (see src/calods2/data/dataset.py),
    # so scaling is the right fix here, not changing the variable.
    target = x_log / x_scale
    charge = (((v.float() - target) ** 2) * hit).sum() / n_hit
    pred_log = v.float() * x_scale

    # energy closure on the soft-gated prediction, as in v4 (see its docstring:
    # an energy term on ungated values optimizes energy the gate then removes)
    true_lin = torch.expm1(x_log.float()).reshape(B, -1)
    pred_lin = torch.expm1(pred_log).clamp(min=0).reshape(B, -1)
    p_e = pred_lin * p.reshape(B, -1)
    e_true = true_lin.sum(1)
    e2 = ((p_e.sum(1) - e_true) / (e_true + 1e-12)).pow(2).mean()

    total = focal + lam * charge + energy_weight * e2
    assert torch.isfinite(total), (
        f"calorimeter_loss_v5: non-finite loss (focal={focal.item()}, "
        f"charge={charge.item()}, e2={e2.item()})")
    return total, focal, charge, e2


@torch.no_grad()
def tune_threshold(model, loader, device, thr_log, taus=None, max_batches=8):
    """
    Pick the occupancy threshold tau on VALIDATION data by maximizing F1.

    v4's decode() uses h > 0, i.e. tau = 0.5, which is the accuracy-optimal
    point for a calibrated probability and NOT the F1-optimal point -- and
    under v5's focal loss the probabilities are not calibrated anyway. The
    NOvA report found tau = 0.35-0.40 on its data and kept this choice strictly
    separate from training; a model-side cut is also what CaloDREAM applies
    (1 MeV) to fix sparsity.

    Returns (best_tau, table) where table is [(tau, precision, recall, f1)].
    """
    if taus is None:
        taus = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    model.eval()
    tp = torch.zeros(len(taus)); fp = torch.zeros(len(taus)); fn = torch.zeros(len(taus))
    for b, (x_raw, energy) in enumerate(loader):
        if b >= max_batches:
            break
        x_raw = x_raw.to(device); energy = energy.to(device)
        x_log = physical_to_model(x_raw, energy)
        hit = x_log > thr_log
        _, h, _ = model.forward_heads(x_log, energy)
        p = torch.sigmoid(h.float())
        for i, t in enumerate(taus):
            g = p > t
            tp[i] += (g & hit).sum().item()
            fp[i] += (g & ~hit).sum().item()
            fn[i] += (~g & hit).sum().item()
    table = []
    for i, t in enumerate(taus):
        pr = (tp[i] / (tp[i] + fp[i]).clamp(min=1)).item()
        rc = (tp[i] / (tp[i] + fn[i]).clamp(min=1)).item()
        table.append((t, pr, rc, 2 * pr * rc / max(pr + rc, 1e-9)))
    best = max(table, key=lambda r: r[3])
    return best[0], table


@torch.no_grad()
def relative_error_phys(x_log, x_hat_log):
    """L2 relative error in physical-normalized space."""
    assert x_hat_log.shape == x_log.shape, (
        f"relative_error_phys: shape mismatch "
        f"{tuple(x_hat_log.shape)} vs {tuple(x_log.shape)}"
    )
    x = torch.clamp(torch.expm1(x_log), min=0.0)
    x_hat = torch.clamp(torch.expm1(x_hat_log), min=0.0)
    B = x.shape[0]
    x = x.reshape(B, -1)
    x_hat = x_hat.reshape(B, -1)
    err = torch.norm(x - x_hat, dim=1)
    norm = torch.norm(x, dim=1)
    return (err / (norm + 1e-8)).mean().item()


@torch.no_grad()
def energy_error_phys(x_log, x_hat_log):
    """Total deposited energy fractional error in physical-normalized space."""
    assert x_hat_log.shape == x_log.shape, (
        f"energy_error_phys: shape mismatch "
        f"{tuple(x_hat_log.shape)} vs {tuple(x_log.shape)}"
    )
    x = torch.clamp(torch.expm1(x_log), min=0.0)
    x_hat = torch.clamp(torch.expm1(x_hat_log), min=0.0)
    E_true = x.sum(dim=(1, 2, 3))
    E_pred = x_hat.sum(dim=(1, 2, 3))
    return (torch.abs(E_true - E_pred) / (E_true + 1e-8)).mean().item()
