"""
suite.py — the evaluation suite.  ONE source of truth.
=======================================================

Used by ae_core.evaluate() (models trained with v17) and by reeval.py
(models trained earlier).  Both write the same keys, so every table in the
paper is computed by exactly the same code.

EVERY METRIC IS SCORED THE SAME WAY
-----------------------------------
For each test jet, compute the quantity on the real jet and on its
reconstruction.  Then

    paired relative error = mean_over_jets |recon - real|  /  std(real)

0 = perfect.  Dividing by the real spread makes different quantities
comparable.

THE SUITE
---------
PRIMARY -- need particles to be compared in pairs; a zero-reach model
           cannot compute them from per-particle sums:
    ecf2_b1    ECF(2, beta=1) = sum_{i<j} w_i w_j dR_ij
               Standard energy correlation function (Larkoski, Salam,
               Thaler 2013).  The square root inside dR does not factorise.
    eec_wide   sum_{i<j} w_i w_j [dR_ij > 0.2]
               Wide-angle part of the energy-energy correlator.  A threshold
               on a PAIR cannot be rewritten as a sum over single particles.
DIAGNOSTIC -- computed and saved, never averaged into a group:
    log_d2     log D2,  D2 = ECF(3, beta=1) / ECF(2, beta=1)^3
               Two-prong vs one-prong discriminant (Larkoski, Moult, Neill
               2014).  Genuine 3-point correlator.  Scored on the LOG: D2
               divides by ECF(2)^3, which is near zero for small, narrow
               jets, so raw D2 has enormous outliers.  Scored raw, a handful
               of such jets dominated the average and the score came out
               BETTER for WORSE-trained models (corr -0.93 with training loss
               across the 9 reach runs).  On the log, an error means "off by
               a factor", which is the physically meaningful scale.  Jets
               with fewer than 3 particles (D2 = 0) are left out.
               The log cut the tail share from ~0.12 to ~0.06 but log D2
               STILL correlated -0.77 with training loss on the real runs,
               so it was moved out of PRIMARY (the rule fixed in advance).
    d2         raw D2, kept for comparison

CONTROL -- no geometry at all; nothing about reach can change it:
    ptD        sqrt(sum_i w_i^2)

HOLISTIC -- one overall score, and the only metric Hariri et al. 2020 used:
    emd        Energy-flow EMD (Komiske, Metodiev, Thaler 2019): transport
               the reconstruction's pT onto the real jet's pT, paying the
               angular distance dR/R, with R = 0.8.  Computed with a Sinkhorn
               approximation (blur 0.01), which converges to the exact EMD.

SUPPORT -- these FACTORISE exactly into sums over single particles, so a
           ZERO-reach model computes them perfectly.  They show that models
           reconstruct well; they are never evidence about reach:
    girth      sum_i w_i r_i
    mass       relative jet mass from summed four-momenta
    ecf2_b2    ECF(2, beta=2) = sum_{i<j} w_i w_j dR_ij^2
               -- squared distance expands into a polynomial and factorises.
               Whether an ECF is testable depends on beta: even-integer beta
               factorises, non-integer or odd beta does not.
    mult       particle count

w_i = pT_i / sum(pT) over real (mask = 1) particles.
dR_ij = sqrt(d_eta^2 + d_phi^2), with d_phi wrapped into (-pi, pi].
"""

import numpy as np

WIDE_DR  = 0.20   # eec_wide threshold
JET_R    = 0.80   # JetNet jets are anti-kT R = 0.8; EMD distance unit

PRIMARY  = ["ecf2_b1", "eec_wide"]
CONTROL  = ["ptD"]
SUPPORT  = ["girth", "mass", "ecf2_b2"]
# D2 is reported but NOT in a group: raw D2 correlated -0.93 with training
# loss across the 9 reach runs, log D2 still -0.77 -- both reward
# worse-trained models, so neither can carry a claim.
DIAGNOSTIC = ["log_d2", "d2"]
PER_JET  = PRIMARY + CONTROL + SUPPORT + DIAGNOSTIC   # computed per jet
ALL_KEYS = PER_JET + ["mult"]

LABELS = {
    "ecf2_b1": "ECF(2, β=1)", "eec_wide": "EEC wide (ΔR>0.2)",
    "log_d2": "log D₂", "d2": "D₂ raw (diagnostic)",
    "ptD": "ptD (control)", "girth": "girth", "mass": "jet mass",
    "ecf2_b2": "ECF(2, β=2)", "mult": "multiplicity", "emd": "EMD",
}


# ══════════════════════════════════════════════════════════════════════════
# Per-jet observables
# ══════════════════════════════════════════════════════════════════════════
def observables(jet):
    """jet: (N, 4) array of (eta, phi, pt, mask).  Returns a dict, or None if
    the jet has fewer than 2 real particles."""
    m = jet[:, 3] > 0.5
    eta, phi, pt = jet[m, 0], jet[m, 1], jet[m, 2]
    n = len(pt)
    if n < 2 or pt.sum() <= 0:
        return None
    w = pt / pt.sum()

    d_eta = eta[:, None] - eta[None, :]
    d_phi = (phi[:, None] - phi[None, :] + np.pi) % (2 * np.pi) - np.pi
    dR = np.sqrt(d_eta ** 2 + d_phi ** 2)
    iu = np.triu_indices(n, k=1)
    wij, dRij = (w[:, None] * w[None, :])[iu], dR[iu]

    # --- primary ----------------------------------------------------------
    ecf2_b1 = float((wij * dRij).sum())
    eec_wide = float((wij * (dRij > WIDE_DR)).sum())
    # ECF(3, beta=1) = sum_{i<j<k} w_i w_j w_k dR_ij dR_ik dR_jk.
    # Exact identity: with M = diag(w) @ dR (dR has a zero diagonal, so any
    # repeated index vanishes), trace(M^3) sums every ORDERED distinct triple
    # once, i.e. 6 * ECF(3).  Two matmuls instead of an n^3 array.
    M = w[:, None] * dR
    ecf3_b1 = float(((M @ M) * M.T).sum() / 6.0)
    d2 = float(ecf3_b1 / ecf2_b1 ** 3) if ecf2_b1 > 1e-12 else 0.0
    # NaN (not 0) when undefined, so _relerr drops the jet instead of
    # scoring log(0).  Fewer than 3 particles -> ECF(3) = 0 -> undefined.
    log_d2 = float(np.log(d2)) if (d2 > 0 and n >= 3) else float("nan")

    # --- control ----------------------------------------------------------
    ptD = float(np.sqrt((w ** 2).sum()))

    # --- support ----------------------------------------------------------
    girth = float((w * np.sqrt(eta ** 2 + phi ** 2)).sum())
    ecf2_b2 = float((wij * dRij ** 2).sum())
    eta_c = np.clip(eta, -5, 5)
    px = (pt * np.cos(phi)).sum(); py = (pt * np.sin(phi)).sum()
    pz = (pt * np.sinh(eta_c)).sum(); E = (pt * np.cosh(eta_c)).sum()
    mass = float(np.sqrt(max(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0)))

    return {"ecf2_b1": ecf2_b1, "eec_wide": eec_wide, "log_d2": log_d2,
            "d2": d2, "ptD": ptD,
            "girth": girth, "mass": mass, "ecf2_b2": ecf2_b2, "mult": float(n)}


def observables_batch(arr, progress=""):
    out = {k: np.full(len(arr), np.nan) for k in ALL_KEYS}
    ok = np.zeros(len(arr), dtype=bool)
    for i in range(len(arr)):
        o = observables(arr[i])
        if o is None:
            continue
        ok[i] = True
        for k in ALL_KEYS:
            out[k][i] = o[k]
        if progress and (i + 1) % 20000 == 0:
            print(f"      {progress}: {i+1}/{len(arr)}", flush=True)
    return out, ok


# ══════════════════════════════════════════════════════════════════════════
# Energy-flow EMD (Komiske, Metodiev, Thaler 2019) -- the Hariri et al. metric
# ══════════════════════════════════════════════════════════════════════════
def emd_energy_flow(recon, real, device, batch_size=512, blur=0.01):
    """Per-jet EMD between real and reconstructed jets.

    Transport cost = dR / R on the (eta, phi) plane; transported mass = each
    particle's pT fraction.  Both jets are normalised to unit total pT, so the
    |sum E - sum E'| term of the EMD definition vanishes.

    NOTE: training uses a DIFFERENT distance (uniform weights, pT as a third
    coordinate).  That one is a training loss, not this metric.
    """
    import torch
    from geomloss import SamplesLoss
    fn = SamplesLoss(loss="sinkhorn", p=1, blur=blur, scaling=0.9,
                     debias=True)
    vals = []
    for i in range(0, len(recon), batch_size):
        p = torch.as_tensor(recon[i:i + batch_size], dtype=torch.float32,
                            device=device)
        t = torch.as_tensor(real[i:i + batch_size], dtype=torch.float32,
                            device=device)
        # pT weights on real particles only; clamp keeps log-domain Sinkhorn
        # finite on padding without giving padding any real mass
        pw = (p[:, :, 2] * (p[:, :, 3] > 0.5)).clamp(min=1e-12)
        tw = (t[:, :, 2] * (t[:, :, 3] > 0.5)).clamp(min=1e-12)
        pw = pw / pw.sum(1, keepdim=True)
        tw = tw / tw.sum(1, keepdim=True)
        px = (p[:, :, :2] / JET_R).contiguous()
        tx = (t[:, :, :2] / JET_R).contiguous()
        vals.append(fn(pw, px, tw, tx).float().cpu().numpy())
    return np.concatenate(vals)


# ══════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════
def _relerr(real, recon, sel):
    good = sel & np.isfinite(real) & np.isfinite(recon)
    if good.sum() < 20:
        return None
    r, g = real[good], recon[good]
    return float(np.mean(np.abs(g - r)) / (r.std() + 1e-12))


def tail_share(real, recon, sel, top=0.01):
    """Fraction of the total |recon - real| that comes from the worst `top`
    fraction of jets.  ~0.01-0.1 is healthy; near 1 means a handful of jets
    decide the score (what went wrong with raw D2)."""
    good = sel & np.isfinite(real) & np.isfinite(recon)
    if good.sum() < 100:
        return None
    e = np.sort(np.abs(recon[good] - real[good]))[::-1]
    k = max(1, int(round(top * len(e))))
    return float(e[:k].sum() / (e.sum() + 1e-12))


def score(real_arr, recon_arr, latent_dim, jet_types, device,
          batch_size=512, log=print):
    """Returns the full suite as a flat dict of floats plus nested splits."""
    log("      observables: real jets ...")
    o_r, ok_r = observables_batch(real_arr, "real")
    log("      observables: reconstructions ...")
    o_g, ok_g = observables_batch(recon_arr, "recon")
    ok = ok_r & ok_g

    log("      energy-flow EMD ...")
    emd = emd_energy_flow(recon_arr, real_arr, device, batch_size)

    s = {"n_jets_scored": int(ok.sum())}
    for k in PER_JET:
        s[k] = _relerr(o_r[k], o_g[k], ok)
    # how much of each error the worst 1% of jets carry
    s["tail_share_top1pct"] = {k: tail_share(o_r[k], o_g[k], ok)
                               for k in PER_JET}
    s["emd"] = float(np.mean(emd[ok]))
    s["emd_median"] = float(np.median(emd[ok]))

    mr, mg = o_r["mult"][ok], o_g["mult"][ok]
    s["mult_mae"] = float(np.mean(np.abs(mg - mr)))
    s["mult_exact_frac"] = float(np.mean(mg == mr))
    s["mult_mae_mean_baseline"] = float(np.mean(np.abs(mr - mr.mean())))

    def mean_of(keys, d):
        v = [d[k] for k in keys if d.get(k) is not None]
        return float(np.mean(v)) if v else None

    s["primary_mean"] = mean_of(PRIMARY, s)
    s["support_mean"] = mean_of(SUPPORT, s)
    s["control_ptD"] = s["ptD"]

    # --- split by jet size relative to the latent --------------------------
    # A compression-driven reach effect should concentrate in jets carrying
    # more numbers than the latent can hold (3 per particle).
    n_real = o_r["mult"]
    thresh = latent_dim / 3.0
    split = {}
    for name, sel in [("small", ok & (n_real <= thresh)),
                      ("large", ok & (n_real > thresh))]:
        d = {k: _relerr(o_r[k], o_g[k], sel) for k in PER_JET}
        d["n_jets"] = int(sel.sum())
        d["primary_mean"] = mean_of(PRIMARY, d)
        d["support_mean"] = mean_of(SUPPORT, d)
        split[name] = d
    s["split_by_size"] = split
    s["split_threshold_particles"] = thresh

    # --- per jet type --------------------------------------------------------
    per_type = {}
    for tid, tname in enumerate(["gluon", "quark", "top"]):
        sel = ok & (jet_types == tid)
        if sel.sum() < 50:
            continue
        d = {k: _relerr(o_r[k], o_g[k], sel) for k in PER_JET}
        d["n_jets"] = int(sel.sum())
        d["primary_mean"] = mean_of(PRIMARY, d)
        d["emd"] = float(np.mean(emd[sel]))
        per_type[tname] = d
    s["per_jet_type"] = per_type

    return s, (o_r, o_g, ok)
