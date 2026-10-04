#!/usr/bin/env python
"""
diagnostic.py — v17_final.  Four checks.  No autoencoder is retrained.
=====================================================================

Experiment 3 showed that reach does not help.  These checks ask WHY, and
tidy up two loose ends in Experiments 1 and 3.

  STEP 1  rescore   Re-score every trained model with the fixed suite
                    (D2 is now scored on log D2 -- see suite.py).  Rewrites
                    each run's suite.json; the old one is kept as
                    suite_pre_d2fix.json.  Then re-run aggregate.py.

  STEP 2  factor    TEST A -- does ECF(2, beta=1) nearly factorise on REAL
                    jets?  No model involved.
          learned     A1 (factor):  compute per-particle sums (the only thing
                                    a zero-reach encoder can build), fit
                                    gradient-boosted trees sums -> observable,
                                    score on the test split.
                      A2 (learned): train three encoders to PREDICT the
                                    observables directly, no decoder:
                                    reach 0 (no message passing), reach 2
                                    (ChebNet K=2), reach 10 (ChebNet K=6).
                    If sums / reach 0 already predict ECF(2, beta=1) well,
                    reach has nothing left to add (hypothesis H1).

  STEP 3  probe     TEST B -- what is IN the latent?  Freeze each trained
                    encoder, fit a small network latent -> true observable,
                    compare with what the decoder actually reconstructs.
                        probe << reconstruction  : the latent holds it and
                                                   the decoder loses it (H2)
                        probe ~= reconstruction  : the latent lacks it
                    Fitted on one half of the test split, scored on the
                    other half; the autoencoder never saw either.

  STEP 4  count     Experiment 1 mechanism.  Re-compare StatsPool vs Set2Set
                    on ONLY the jets where BOTH got the particle count
                    right.  If the gap mostly vanishes, the count failure
                    (Proposition 2) explains it.

  STEP 5  smooth    Over-smoothing.  How alike do a jet's particles look
                    after message passing?  Mean cosine similarity between
                    the particle embeddings of one jet, before and after the
                    conv layers.  If more hops make particles look more alike,
                    that would explain why K=6 decodes slightly worse.

  report            Writes diagnostics.md + figures from whatever steps have
                    finished.

Every step caches its output in --out, so a job that hits its time limit
can simply be resubmitted and carries on.

    python v17_final/diagnostic.py \\
        --pooling     $SCRATCH/gsoc_rushil/v16_ablation/results \\
        --compression $SCRATCH/gsoc_rushil/v16_ablation/results_scout \\
        --reach       $SCRATCH/gsoc_rushil/v17_final/results \\
        --out         $SCRATCH/gsoc_rushil/v17_final/diagnostics

    --steps rescore,probe,count,smooth,factor,learned,report   (default: all)
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ae_core as A   # noqa: E402
import suite as S     # noqa: E402
import reeval as R    # noqa: E402

import torch                                   # noqa: E402
import torch.nn as nn                          # noqa: E402
import torch.nn.functional as F                # noqa: E402
from torch.utils.data import DataLoader        # noqa: E402
from torch_geometric.data import Batch         # noqa: E402

import matplotlib                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402
from scipy import stats                        # noqa: E402

ALL_STEPS = ["rescore", "probe", "count", "smooth", "factor", "learned",
             "report"]

# observables the probes and Test A try to recover
PROBE_TARGETS = ["ecf2_b1", "eec_wide", "log_d2", "ecf2_b2", "girth", "mass",
                 "ptD", "mult"]
FACTOR_TARGETS = ["ecf2_b1", "eec_wide", "log_d2", "ecf2_b2"]
REACH_ORDER = ["GraphSAGE", "ChebNet K=2", "ChebNet K=4", "ChebNet K=6",
               "GraphSAGE + Set2Set"]
COLOURS = {"GraphSAGE": "#8a8797", "ChebNet K=2": "#6A4FC8",
           "ChebNet K=4": "#3B82B0", "ChebNet K=6": "#1B7C77",
           "GraphSAGE + Set2Set": "#8F6410"}


def log(msg=""):
    print(msg, flush=True)


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def jload(path):
    with open(path) as f:
        return json.load(f)


def relerr(real, pred, sel=None):
    """Paired relative error -- the suite's one scoring rule."""
    sel = np.ones(len(real), bool) if sel is None else sel
    return S._relerr(np.asarray(real, float), np.asarray(pred, float), sel)


def model_name(cfg):
    base = "GraphSAGE" if cfg["conv"] == "sage" else f"ChebNet K={cfg['K']}"
    return base + (" + Set2Set" if cfg.get("pool") == "set2set" else "")


# ══════════════════════════════════════════════════════════════════════════
# FINDING RUNS
# ══════════════════════════════════════════════════════════════════════════
def find_runs(root, group):
    """Run folders in `root` with a trained model.  Old (pre-v17) folders are
    GraphSAGE only: their ChebNet runs were trained on directed graphs."""
    runs = []
    if not root:
        return runs
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        meta = os.path.join(d, "metrics.json")
        if name.startswith("_") or not os.path.exists(meta):
            continue
        if not all(os.path.exists(os.path.join(d, f))
                   for f in ("encoder.pth", "decoder.pth")):
            log(f"  [skip] {name}: encoder.pth / decoder.pth missing")
            continue
        cfg = R.run_config(jload(meta))
        if group in ("pooling", "compression") and cfg["conv"] != "sage":
            continue
        if group == "pooling" and cfg["latent_dim"] != 512:
            continue
        if group == "compression" and cfg["pool"] != "stats":
            continue
        if group == "reach" and cfg["latent_dim"] != 64:
            continue
        runs.append((group, name, d, cfg))
    return runs


_TEST = {}


def test_split(cfg):
    """JetNet test split as graphs, built exactly the way the run was
    trained (directed for pre-v17 runs, undirected for v17)."""
    key = (cfg["total_particles"], cfg["knn"], bool(cfg["graph_symmetric"]))
    if key not in _TEST:
        path = A.graph_cache_path("test", key[0], key[1], None, key[2])
        cached = A.load_graph_cache(path, log)
        if cached is None:
            pdata, _, ptypes = A.load_jetnet_data(key[0], None, split="test",
                                                  log=log)
            cached = A.build_graphs(pdata, ptypes, key[1], key[0], path,
                                    log=log, symmetric=key[2])
            del pdata
        _TEST[key] = cached
    return _TEST[key], key


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 (+ the per-run pass that STEPS 3 and 4 also use)
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def encode_reconstruct(enc, dec, graphs, targets, bs):
    zs, recon, real = [], [], []
    for i in range(0, len(graphs), bs):
        bg = Batch.from_data_list(graphs[i:i + bs]).to(A.device)
        z = enc(bg.x, bg.edge_index, bg.batch)
        zs.append(z.float().cpu().numpy())
        recon.append(dec(z).cpu().numpy())
        real.append(torch.stack(targets[i:i + bs]).numpy())
    return np.concatenate(zs), np.concatenate(recon), np.concatenate(real)


def write_suite(run_dir, cfg, new):
    """Merge the re-scored suite into suite.json, keeping every training-time
    key (params, train loss, graph stats ...).  The old file is backed up
    once, so re-running never overwrites the original."""
    path = os.path.join(run_dir, "suite.json")
    backup = os.path.join(run_dir, "suite_pre_d2fix.json")
    if os.path.exists(path):
        base = jload(path)
        if not os.path.exists(backup):
            jdump(base, backup)
    else:
        base = {k: cfg.get(k) for k in R.META_KEYS}
    base.update(new)
    base["tag"] = os.path.basename(run_dir)
    base["scored_by"] = "diagnostic.py (suite with log D2)"
    jdump(base, path)


def per_run_pass(run, steps, out, args):
    """One pass through the test split per model: re-score (step 1), save the
    per-jet observables (step 4), probe the latent (step 3)."""
    group, name, run_dir, cfg = run
    obs_path = os.path.join(out, "_obs", f"{group}__{name}.npz")
    probe_path = os.path.join(out, "_probe", f"{group}__{name}.json")
    smooth_path = os.path.join(out, "_smooth", f"{group}__{name}.json")
    done_flag = os.path.join(out, "_obs", f"{group}__{name}.rescored")

    need_rescore = "rescore" in steps and (args.force
                                           or not os.path.exists(done_flag))
    need_obs = "count" in steps and (args.force or not os.path.exists(obs_path))
    need_probe = "probe" in steps and (args.force
                                       or not os.path.exists(probe_path))
    need_smooth = "smooth" in steps and (args.force
                                         or not os.path.exists(smooth_path))
    if not (need_rescore or need_obs or need_probe or need_smooth):
        log(f"  [{name}] all requested outputs exist -- skipping")
        return

    (graphs, targets, types), key = test_split(cfg)
    n = len(graphs) if not args.n_eval else min(args.n_eval, len(graphs))
    log(f"\n  [{group}/{name}]  {model_name(cfg)}  pool={cfg['pool']}  "
        f"latent={cfg['latent_dim']}  graph="
        f"{'undirected' if key[2] else 'directed'}  jets={n}")

    t0 = time.time()
    enc, dec = R.load_model(run_dir, cfg)
    if need_smooth:
        ns = min(n, args.smooth_jets)
        before, after = node_similarity(enc, graphs[:ns], args.batch_size)
        jdump({"group": group, "tag": name, "model": model_name(cfg),
               "hops": cfg.get("receptive_field_hops"),
               "seed": cfg.get("seed", 0), "n_jets": int(ns),
               "cos_before": before, "cos_after": after}, smooth_path)
        log(f"      over-smoothing: particle-embedding similarity "
            f"{before:.3f} (before convs) -> {after:.3f} (after)")
    if not (need_rescore or need_obs or need_probe):
        log(f"      done in {time.time() - t0:.0f}s")
        return
    z, recon, real = encode_reconstruct(enc, dec, graphs[:n], targets[:n],
                                        args.batch_size)
    del enc, dec
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    suite, (o_r, o_g, ok) = S.score(real, recon, cfg["latent_dim"],
                                    types[:n], A.device, args.batch_size,
                                    log=log)
    del recon, real

    if need_rescore:
        write_suite(run_dir, cfg, suite)
        open(done_flag, "w").write("ok\n")
        log(f"      rescored: primary={suite['primary_mean']:.4f} "
            f"(log D2 {suite['log_d2']:.4f}, raw D2 {suite['d2']:.4f})  "
            f"support={suite['support_mean']:.4f}  ptD={suite['ptD']:.4f}")

    # per-jet observables: step 4 compares models jet by jet
    np.savez_compressed(
        obs_path, ok=ok, types=types[:n],
        **{f"r_{k}": o_r[k] for k in S.ALL_KEYS},
        **{f"g_{k}": o_g[k] for k in S.ALL_KEYS})

    if need_probe:
        res = probe_latent(z, o_r, o_g, ok, args)
        res.update({"group": group, "tag": name, "model": model_name(cfg),
                    "pool": cfg["pool"], "latent_dim": cfg["latent_dim"],
                    "seed": cfg.get("seed", 0),
                    "hops": cfg.get("receptive_field_hops")})
        jdump(res, probe_path)
        log("      probe (held-out half):  target   recon -> probe(mlp)")
        for k in PROBE_TARGETS:
            r = res["targets"][k]
            log(f"        {k:9s} {r['recon']:.4f} -> {r['probe_mlp']:.4f}")
    log(f"      done in {time.time() - t0:.0f}s")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — OVER-SMOOTHING
# ══════════════════════════════════════════════════════════════════════════
def _mean_cos(h, batch, n_graphs):
    """Per jet: mean cosine similarity over all PAIRS of distinct particles.
    With unit vectors u_i, |sum_i u_i|^2 = sum_ij cos_ij = n + sum_{i!=j}."""
    u = F.normalize(h, dim=1)
    s = torch.zeros(n_graphs, u.size(1), device=u.device).index_add_(0, batch, u)
    n = torch.bincount(batch, minlength=n_graphs).float()
    ok = n > 1
    val = ((s ** 2).sum(1) - n)[ok] / (n * (n - 1))[ok]
    return val.cpu().numpy()


@torch.no_grad()
def node_similarity(enc, graphs, bs):
    """Particle embeddings right before and right after message passing --
    exactly the path JetEncoder.forward takes up to its out_norm."""
    enc.eval()
    before, after = [], []
    for i in range(0, len(graphs), bs):
        bg = Batch.from_data_list(graphs[i:i + bs]).to(A.device)
        ng = int(bg.batch.max()) + 1
        h0 = enc.input_proj(bg.x)
        h = h0
        for j, cv in enumerate(enc.convs):
            h = cv(h, bg.edge_index)
            if j < len(enc.bns):
                h = F.leaky_relu(enc.bns[j](h))
        h = enc.out_norm(h)
        before.append(_mean_cos(enc.out_norm(h0), bg.batch, ng))
        after.append(_mean_cos(h, bg.batch, ng))
    return (float(np.concatenate(before).mean()),
            float(np.concatenate(after).mean()))


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — LATENT PROBE
# ══════════════════════════════════════════════════════════════════════════
def _standardise(Y, M):
    mu = np.array([Y[M[:, j], j].mean() for j in range(Y.shape[1])])
    sd = np.array([Y[M[:, j], j].std() + 1e-12 for j in range(Y.shape[1])])
    return mu, sd


def fit_mlp(X, Y, M, X_eval, epochs, seed, hidden=256, log_every=0):
    """Small MLP X -> Y with a masked MSE (targets can be NaN, e.g. log D2 of
    a 2-particle jet).  Early stopping on 10% of the fitting data."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = _standardise(Y, M)
    Xn = (X - xm) / xs
    Yn = np.where(M, (Y - ym) / ys, 0.0)

    idx = rng.permutation(len(X))
    n_val = max(1, len(X) // 10)
    va, tr = idx[:n_val], idx[n_val:]
    dev = A.device
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa
    Xt, Yt, Mt = t(Xn), t(Yn), t(M.astype(np.float32))

    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, Y.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def loss_on(ii):
        p = net(Xt[ii])
        return ((p - Yt[ii]) ** 2 * Mt[ii]).sum() / Mt[ii].sum().clamp(min=1)

    best, best_state = float("inf"), None
    va_t = torch.as_tensor(va, device=dev)
    for ep in range(epochs):
        net.train()
        perm = torch.as_tensor(rng.permutation(tr), device=dev)
        for i in range(0, len(perm), 1024):
            opt.zero_grad(set_to_none=True)
            loss_on(perm[i:i + 1024]).backward()
            opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_on(va_t))
        if v < best:
            best = v
            best_state = {k: x.detach().clone()
                          for k, x in net.state_dict().items()}
        if log_every and (ep + 1) % log_every == 0:
            log(f"        probe epoch {ep+1}/{epochs}  val={v:.4f}")
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        P = net(t((X_eval - xm) / xs)).cpu().numpy()
    return P * ys + ym


def fit_ridge(X, Y, M, X_eval, lam=1e-3):
    """Linear probe, one target at a time (each has its own NaN mask)."""
    xm, xs = X.mean(0), X.std(0) + 1e-6
    Xn = np.c_[(X - xm) / xs, np.ones(len(X))]
    Xe = np.c_[(X_eval - xm) / xs, np.ones(len(X_eval))]
    out = np.zeros((len(X_eval), Y.shape[1]))
    for j in range(Y.shape[1]):
        m = M[:, j]
        G = Xn[m].T @ Xn[m] + lam * len(Xn[m]) * np.eye(Xn.shape[1])
        out[:, j] = Xe @ np.linalg.solve(G, Xn[m].T @ Y[m, j])
    return out


def probe_latent(z, o_r, o_g, ok, args):
    """Fit on one half of the test split, score on the other half.  The
    same held-out half is used for the reconstruction error, so the two
    numbers are directly comparable."""
    rng = np.random.default_rng(12345)          # same halves for every run
    idx = rng.permutation(len(z))
    fit, ev = idx[: len(z) // 2], idx[len(z) // 2:]
    fit = fit[ok[fit]]
    ev = ev[ok[ev]]

    Y = np.stack([o_r[k] for k in PROBE_TARGETS], 1)
    M = np.isfinite(Y)
    Y = np.nan_to_num(Y)

    P_mlp = fit_mlp(z[fit], Y[fit], M[fit], z[ev], args.probe_epochs,
                    seed=args.seed)
    P_lin = fit_ridge(z[fit], Y[fit], M[fit], z[ev])

    res = {"n_fit": int(len(fit)), "n_eval": int(len(ev)), "targets": {}}
    for j, k in enumerate(PROBE_TARGETS):
        real = np.where(M[ev, j], Y[ev, j], np.nan)
        res["targets"][k] = {
            "recon": relerr(o_r[k][ev], o_g[k][ev]),
            "probe_mlp": relerr(real, P_mlp[:, j]),
            "probe_linear": relerr(real, P_lin[:, j]),
            "predict_mean": relerr(real, np.full(len(ev),
                                                 np.nanmean(Y[fit, j]))),
        }
    mr = o_r["mult"][ev]
    res["mult_exact_frac"] = {
        "recon": float(np.mean(o_g["mult"][ev] == mr)),
        "probe_mlp": float(np.mean(np.round(P_mlp[:, -1]) == mr)),
    }
    return res


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 / A1 — FACTORISATION TEST (no model)
# ══════════════════════════════════════════════════════════════════════════
def moment_features(P, rich=True):
    """Per-particle sums -- EVERYTHING a zero-reach encoder can build.
    Each feature is sum_i f(particle i): no pair of particles is ever
    compared.  `rich=False` keeps orders <= 2."""
    m = P[:, :, 3] > 0.5
    eta = np.where(m, P[:, :, 0], 0.0)
    phi = np.where(m, P[:, :, 1], 0.0)
    pt = np.where(m, P[:, :, 2], 0.0)
    w = pt / np.clip(pt.sum(1, keepdims=True), 1e-12, None)
    r = np.sqrt(eta ** 2 + phi ** 2)
    S_ = lambda a: (np.where(m, a, 0.0)).sum(1)          # noqa: E731

    f = {"n": m.sum(1).astype(float),
         "w2": S_(w ** 2),
         "w_eta": S_(w * eta), "w_phi": S_(w * phi),
         "w_eta2": S_(w * eta ** 2), "w_phi2": S_(w * phi ** 2),
         "w_etaphi": S_(w * eta * phi),
         "w_r": S_(w * r), "w_r2": S_(w * r ** 2)}
    if rich:
        f.update({
            "w3": S_(w ** 3), "w4": S_(w ** 4),
            "w_eta3": S_(w * eta ** 3), "w_phi3": S_(w * phi ** 3),
            "w_eta2phi": S_(w * eta ** 2 * phi),
            "w_etaphi2": S_(w * eta * phi ** 2),
            "w_eta4": S_(w * eta ** 4), "w_phi4": S_(w * phi ** 4),
            "w_eta2phi2": S_(w * eta ** 2 * phi ** 2),
            "w_r3": S_(w * r ** 3), "w_r4": S_(w * r ** 4),
            "w_sqrtr": S_(w * np.sqrt(r)),
            "w2_r": S_(w ** 2 * r), "w2_r2": S_(w ** 2 * r ** 2),
            "w2_eta": S_(w ** 2 * eta), "w2_phi": S_(w ** 2 * phi),
            "r_mean": S_(r) / np.clip(m.sum(1), 1, None),
            "r2_mean": S_(r ** 2) / np.clip(m.sum(1), 1, None)})
    names = list(f)
    return np.stack([f[k] for k in names], 1), names


def observables_of(P, label):
    o, ok = S.observables_batch(P, label)
    return o, ok


def step_factor(out, args):
    path = os.path.join(out, "testA_factor.json")
    if os.path.exists(path) and not args.force:
        log("  [factor] testA_factor.json exists -- skipping")
        return
    from sklearn.ensemble import HistGradientBoostingRegressor

    log("\n=== STEP 2 / A1: does ECF(2, beta=1) nearly factorise? ===")
    Ptr, _, _ = A.load_jetnet_data(150, None, split="train", log=log)
    Pte, _, _ = A.load_jetnet_data(150, None, split="test", log=log)
    rng = np.random.default_rng(0)
    if args.factor_train_jets and len(Ptr) > args.factor_train_jets:
        Ptr = Ptr[np.sort(rng.choice(len(Ptr), args.factor_train_jets,
                                     replace=False))]
    if args.n_eval:
        Pte = Pte[:args.n_eval]
    log(f"  fit on {len(Ptr)} train jets, score on {len(Pte)} test jets")

    o_tr, ok_tr = observables_of(Ptr, "train obs")
    o_te, ok_te = observables_of(Pte, "test obs")

    res = {"n_fit": int(ok_tr.sum()), "n_eval": int(ok_te.sum()),
           "targets": {}, "features": {}}

    # exact identity: ECF(2, beta=2) = sum w r^2 - |sum w x|^2
    F_te, names = moment_features(Pte, rich=False)
    fi = {k: i for i, k in enumerate(names)}
    exact = (F_te[:, fi["w_r2"]] - F_te[:, fi["w_eta"]] ** 2
             - F_te[:, fi["w_phi"]] ** 2)
    res["ecf2_b2_exact_from_sums"] = relerr(o_te["ecf2_b2"], exact, ok_te)
    log(f"  sanity: ECF(2,beta=2) from 3 sums, exact formula -> error "
        f"{res['ecf2_b2_exact_from_sums']:.2e}  (should be ~0)")

    for fs in ["low", "rich"]:
        Ftr, names = moment_features(Ptr, rich=(fs == "rich"))
        Fte, _ = moment_features(Pte, rich=(fs == "rich"))
        res["features"][fs] = names
        for k in FACTOR_TARGETS:
            y = o_tr[k]
            m = ok_tr & np.isfinite(y)
            gb = HistGradientBoostingRegressor(
                max_iter=args.gbdt_iters, learning_rate=0.1,
                max_leaf_nodes=63, early_stopping=True,
                validation_fraction=0.1, random_state=0)
            gb.fit(Ftr[m], y[m])
            pred = gb.predict(Fte)
            e = relerr(o_te[k], pred, ok_te)
            res["targets"].setdefault(k, {})[f"gbdt_{fs}"] = e
            log(f"  {fs:4s} sums ({len(names):2d})  {k:9s} -> error {e:.4f}")
    for k in FACTOR_TARGETS:
        y = o_tr[k]
        res["targets"][k]["predict_mean"] = relerr(
            o_te[k], np.full(len(Pte), np.nanmean(y[ok_tr])), ok_te)
    jdump(res, path)


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 / A2 — SUPERVISED REACH TEST (no decoder)
# ══════════════════════════════════════════════════════════════════════════
LEARNED_ARMS = [
    # name,              encoder kwargs,                                 hops
    ("reach 0 (no MP)",  dict(conv="sage", K=1, layers=0), 0),
    ("reach 2 (K=2)",    dict(conv="cheb", K=2, layers=2), 2),
    ("reach 10 (K=6)",   dict(conv="cheb", K=6, layers=2), 10),
]


class Regressor(nn.Module):
    """The SAME encoder as the autoencoder (StatsPool into a latent of 64),
    with a small head that predicts the observables instead of a decoder."""
    def __init__(self, enc_kw, n_out, latent=64):
        super().__init__()
        self.enc = A.JetEncoder(pool="stats", term_norm="sqrt",
                                latent_dim=latent, **enc_kw)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(latent, 128),
                                  nn.GELU(), nn.Linear(128, n_out))

    def forward(self, bg):
        return self.head(self.enc(bg.x, bg.edge_index, bg.batch))


def _load_sym_split(split, max_per_type=None):
    path = A.graph_cache_path(split, 150, 6, max_per_type, True)
    cached = A.load_graph_cache(path, log)
    if cached is None:
        pdata, _, ptypes = A.load_jetnet_data(150, max_per_type, split=split,
                                              log=log)
        cached = A.build_graphs(pdata, ptypes, 6, 150, path, log=log,
                                symmetric=True)
    return cached


def step_learned(out, args):
    path = os.path.join(out, "testA_learned.json")
    res = jload(path) if os.path.exists(path) and not args.force else {}
    todo = [(a, s) for a in LEARNED_ARMS for s in range(args.learned_seeds)
            if f"{a[0]}|{s}" not in res]
    if not todo:
        log("  [learned] testA_learned.json complete -- skipping")
        return

    log("\n=== STEP 2 / A2: can an encoder PREDICT the observables, "
        "and does reach help? ===")
    tr_g, tr_t, _ = _load_sym_split("train")
    te_g, te_t, _ = _load_sym_split("test")
    rng = np.random.default_rng(0)
    sub = np.arange(len(tr_g))
    if args.learned_train_jets and len(sub) > args.learned_train_jets:
        sub = np.sort(rng.choice(len(sub), args.learned_train_jets,
                                 replace=False))
    tr_g = [tr_g[i] for i in sub]
    tr_P = torch.stack([tr_t[i] for i in sub]).numpy()
    te_n = len(te_g) if not args.n_eval else min(args.n_eval, len(te_g))
    te_g = te_g[:te_n]
    te_P = torch.stack(te_t[:te_n]).numpy()
    del tr_t, te_t

    o_tr, ok_tr = observables_of(tr_P, "train obs")
    o_te, ok_te = observables_of(te_P, "test obs")
    Y = np.stack([o_tr[k] for k in FACTOR_TARGETS], 1)
    M = np.isfinite(Y) & ok_tr[:, None]
    ym, ys = _standardise(np.nan_to_num(Y), M)
    Yn = torch.as_tensor(np.where(M, (np.nan_to_num(Y) - ym) / ys, 0.0),
                         dtype=torch.float32)
    Mt = torch.as_tensor(M, dtype=torch.float32)
    YM = torch.cat([Yn, Mt], 1)                  # target | mask, per jet
    T = len(FACTOR_TARGETS)
    log(f"  train jets={len(tr_g)}  test jets={te_n}  epochs="
        f"{args.learned_epochs}")

    for (arm, enc_kw, hops), seed in todo:
        torch.manual_seed(seed); np.random.seed(seed)
        net = Regressor(enc_kw, T).to(A.device)
        n_par = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.learned_epochs, eta_min=1e-6)
        loader = DataLoader(A.JetGraphDataset(tr_g, YM),
                            batch_size=args.batch_size, shuffle=True,
                            collate_fn=A.collate_fn, drop_last=True)
        log(f"\n  [{arm}] seed {seed}  params={n_par:,}")
        t0 = time.time()
        for ep in range(args.learned_epochs):
            net.train(); tot = 0.0
            for bg, ym_b in loader:
                bg = bg.to(A.device); ym_b = ym_b.to(A.device)
                y, m = ym_b[:, :T], ym_b[:, T:]
                loss = ((net(bg) - y) ** 2 * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); tot += loss.item()
            sch.step()
            log(f"    epoch {ep+1:3d}/{args.learned_epochs}  "
                f"mse={tot / max(len(loader), 1):.4f}  "
                f"({time.time() - t0:.0f}s)")

        net.eval(); preds = []
        with torch.no_grad():
            for i in range(0, te_n, args.batch_size):
                bg = Batch.from_data_list(te_g[i:i + args.batch_size])
                preds.append(net(bg.to(A.device)).cpu().numpy())
        P = np.concatenate(preds) * ys + ym
        errs = {k: relerr(o_te[k], P[:, j], ok_te)
                for j, k in enumerate(FACTOR_TARGETS)}
        log("    test error: " + "  ".join(f"{k}={v:.4f}"
                                           for k, v in errs.items()))
        res[f"{arm}|{seed}"] = {"arm": arm, "hops": hops, "seed": seed,
                                "params": n_par, "errors": errs}
        jdump(res, path)                          # save after every arm
        del net, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════
def f4(v):
    return "—" if v is None else f"{v:.4f}"


def msd(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0:
        return None, None
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def pm(vals):
    mu, sd = msd(vals)
    return "—" if mu is None else f"{mu:.4f} ± {sd:.4f}"


def welch(a, b):
    a = np.array([x for x in a if x is not None], float)
    b = np.array([x for x in b if x is not None], float)
    if len(a) < 2 or len(b) < 2:
        return None, None
    return (float(b.mean() - a.mean()),
            float(stats.ttest_ind(a, b, equal_var=False).pvalue))


def report_d2(runs, md):
    rows = []
    for group, name, d, cfg in runs:
        if group == "reach" and cfg["pool"] != "stats":
            continue
        p = os.path.join(d, "suite.json")
        if not os.path.exists(p):
            continue
        s = jload(p)
        if s.get("log_d2") is None:
            continue
        ts = s.get("tail_share_top1pct", {})
        rows.append((group, name, s, ts))
    if not rows:
        md += ["## 1. D₂ fix", "", "_No re-scored runs yet._", ""]
        return
    md += ["## 1. D₂ fix — does log D₂ behave?", "",
           "Raw D₂ divides by ECF(2)³, which is near zero for small, narrow "
           "jets. **Tail share** = fraction of the total error carried by the "
           "worst 1% of jets: about 0.01–0.1 is healthy, close to 1 means a "
           "handful of jets decide the score.", "",
           "| group | run | raw D₂ error | raw tail share | log D₂ error | "
           "log tail share | support | train loss |",
           "|---|---|---|---|---|---|---|---|"]
    for group, name, s, ts in rows:
        md.append(f"| {group} | {name} | {f4(s.get('d2'))} | "
                  f"{f4(ts.get('d2'))} | {f4(s.get('log_d2'))} | "
                  f"{f4(ts.get('log_d2'))} | {f4(s.get('support_mean'))} | "
                  f"{f4(s.get('final_train_loss'))} |")
    reach = [s for g, _, s, _ in rows
             if g == "reach" and s.get("final_train_loss") is not None]
    if len(reach) >= 4:
        loss = np.array([s["final_train_loss"] for s in reach])
        sup = np.array([s["support_mean"] for s in reach])
        md += ["", "Across the reach runs, a sound metric should get WORSE "
               "(higher) as training gets worse, i.e. correlate positively "
               "with training loss and with the support error:", "",
               "| metric | corr. with train loss | corr. with support error |",
               "|---|---|---|"]
        for k, lab in [("d2", "raw D₂ (old)"), ("log_d2", "log D₂ (new)"),
                       ("ecf2_b1", "ECF(2, β=1)"),
                       ("eec_wide", "EEC wide")]:
            v = np.array([s[k] for s in reach])
            md.append(f"| {lab} | {np.corrcoef(v, loss)[0, 1]:+.2f} | "
                      f"{np.corrcoef(v, sup)[0, 1]:+.2f} |")
        md += ["", f"_{len(reach)} runs from three models: read the SIGN, "
               "not the decimals.  If log D₂ is still strongly negative, drop D₂ "
               "from the primary group rather than keep a metric that "
               "rewards worse models._"]
    md.append("")


def report_factor(out, reach_suites, md):
    p = os.path.join(out, "testA_factor.json")
    md += ["## 2. Test A — does ECF(2, β=1) nearly factorise on real jets?",
           ""]
    if not os.path.exists(p):
        md += ["_Not run yet._", ""]
        return None
    r = jload(p)
    best_recon = {}
    for k in FACTOR_TARGETS:
        best_recon[k] = msd([s.get(k) for s in reach_suites])[0]
    md += ["**A1 — per-particle sums only, no model.** Every input is a sum "
           "over single particles (the only thing a zero-reach encoder can "
           "build), fed to gradient-boosted trees.  "
           f"Fitted on {r['n_fit']:,} train jets, scored on "
           f"{r['n_eval']:,} test jets.", "",
           f"Sanity check: ECF(2, β=2) from three sums with the exact "
           f"formula → error {r['ecf2_b2_exact_from_sums']:.1e} "
           "(Proposition 1: it IS a function of sums).", "",
           "| observable | predict the mean | trees on order-≤2 sums "
           f"({len(r['features']['low'])}) | trees on rich sums "
           f"({len(r['features']['rich'])}) | autoencoder reconstruction "
           "(latent 64, mean of reach runs) |",
           "|---|---|---|---|---|"]
    for k in FACTOR_TARGETS:
        t = r["targets"][k]
        md.append(f"| {S.LABELS.get(k, k)} | {f4(t.get('predict_mean'))} | "
                  f"{f4(t.get('gbdt_low'))} | {f4(t.get('gbdt_rich'))} | "
                  f"{f4(best_recon[k])} |")
    md += ["", "**How to read it.** ECF(2, β=2) is an exact function of the "
           "sums, so its tree error is the *regressor's own floor*.  If "
           "ECF(2, β=1) comes out close to that floor, it is a function of "
           "per-particle sums *in practice* and reach has nothing to add "
           "(H1).  If it stays well above the floor, pairs genuinely carry "
           "information the sums miss.", ""]
    return r


def report_learned(out, md):
    p = os.path.join(out, "testA_learned.json")
    md += ["**A2 — encoders trained to predict the observables directly** "
           "(same encoder and StatsPool into 64 numbers as the autoencoder, "
           "a small head instead of the decoder).  Reach 0 = no message "
           "passing at all.", ""]
    if not os.path.exists(p):
        md += ["_Not run yet._", ""]
        return
    r = jload(p)
    arms = [a[0] for a in LEARNED_ARMS if any(v["arm"] == a[0]
                                               for v in r.values())]
    md += ["| encoder | hops | seeds | params | " +
           " | ".join(S.LABELS.get(k, k) for k in FACTOR_TARGETS) + " |",
           "|" + "---|" * (len(FACTOR_TARGETS) + 4)]
    for a in arms:
        vs = [v for v in r.values() if v["arm"] == a]
        md.append(f"| {a} | {vs[0]['hops']} | {len(vs)} | "
                  f"{vs[0]['params']:,} | " +
                  " | ".join(pm([v["errors"][k] for v in vs])
                             for k in FACTOR_TARGETS) + " |")
    md += ["", "If reach 0 ≈ reach 10 on ECF(2, β=1), reach is not needed "
           "even to *know* the quantity, with no decoder in the way.  If "
           "reach 10 is clearly better here but not in the autoencoder, the "
           "decoder or the bottleneck hides the benefit.", ""]


def load_probes(out):
    return [jload(p) for p in sorted(glob.glob(os.path.join(out, "_probe",
                                                            "*.json")))]


def report_probe(out, md):
    probes = load_probes(out)
    md += ["## 3. Test B — what is inside the latent?", "",
           "Each trained encoder is frozen.  A small network is fitted to "
           "read the true observable off the latent (on half the test "
           "split) and scored on the other half, next to what the decoder "
           "actually reconstructs for the same jets.", "",
           "- **probe ≪ reconstruction**: the latent holds it, the decoder "
           "loses it.", "- **probe ≈ reconstruction**: the latent itself "
           "lacks it.", ""]
    if not probes:
        md += ["_Not run yet._", ""]
        return probes
    reach = [p for p in probes if p["group"] == "reach"]
    if reach:
        models = [m for m in REACH_ORDER if any(p["model"] == m
                                                for p in reach)]
        md += ["### Latent-64 runs (mean over seeds)", "",
               "| observable | " + " | ".join(
                   f"{m}: recon → probe" for m in models) + " |",
               "|" + "---|" * (len(models) + 1)]
        for k in PROBE_TARGETS:
            cells = []
            for m in models:
                ps = [p for p in reach if p["model"] == m]
                rc = msd([p["targets"][k]["recon"] for p in ps])[0]
                pr = msd([p["targets"][k]["probe_mlp"] for p in ps])[0]
                cells.append(f"{f4(rc)} → {f4(pr)}")
            md.append(f"| {S.LABELS.get(k, k)} | " + " | ".join(cells) + " |")
        md += ["", "**Does reach put more into the latent?**  Probe error, "
               "Welch t-test across seeds (Δ > 0 = the second model's latent "
               "holds more).", "",
               "| observable | K=2 → K=4 Δ (p) | K=2 → K=6 Δ (p) | "
               "GraphSAGE → K=6 Δ (p) | StatsPool → Set2Set Δ (p) |",
               "|---|---|---|---|---|"]
        g = lambda m, k: [p["targets"][k]["probe_mlp"]       # noqa: E731
                          for p in reach if p["model"] == m]
        for k in PROBE_TARGETS:
            cells = []
            for a, b in [("ChebNet K=2", "ChebNet K=4"),
                         ("ChebNet K=2", "ChebNet K=6"),
                         ("GraphSAGE", "ChebNet K=6"),
                         ("GraphSAGE", "GraphSAGE + Set2Set")]:
                d, pv = welch(g(b, k), g(a, k))
                cells.append("—" if d is None else f"{d:+.4f} ({pv:.3f})")
            md.append(f"| {S.LABELS.get(k, k)} | " + " | ".join(cells) + " |")
        md.append("")
    other = [p for p in probes if p["group"] != "reach"]
    if other:
        md += ["### Pooling and compression runs (one seed each)", "",
               "| run | pool | latent | ECF(2, β=1) recon → probe | "
               "log D₂ recon → probe | count exact: recon / probe |",
               "|---|---|---|---|---|---|"]
        for p in other:
            t = p["targets"]
            md.append(
                f"| {p['tag']} | {p['pool']} | {p['latent_dim']} | "
                f"{f4(t['ecf2_b1']['recon'])} → {f4(t['ecf2_b1']['probe_mlp'])}"
                f" | {f4(t['log_d2']['recon'])} → "
                f"{f4(t['log_d2']['probe_mlp'])} | "
                f"{100 * p['mult_exact_frac']['recon']:.1f}% / "
                f"{100 * p['mult_exact_frac']['probe_mlp']:.1f}% |")
        md += ["", "_For Set2Set, a low count accuracy **from the probe** "
               "means the count is missing from the latent itself — "
               "Proposition 2, measured directly._", ""]
    return probes


def _obs_runs(out, group, root):
    """{(pool, seed): (tag, npz)} for the GraphSAGE runs of one group."""
    runs = {}
    for f in glob.glob(os.path.join(out, "_obs", f"{group}__*.npz")):
        tag = os.path.basename(f)[len(group) + 2:-4]
        mp = os.path.join(root or "", tag, "metrics.json")
        if not os.path.exists(mp):
            continue
        cfg = R.run_config(jload(mp))
        if cfg["conv"] != "sage":
            continue
        if group == "reach" and cfg["latent_dim"] != 64:
            continue
        runs[(cfg["pool"], cfg.get("seed", 0))] = (tag, np.load(f))
    return runs


def _count_rows(pairs):
    """Errors on all jets and on jets where BOTH models counted right,
    averaged over the (StatsPool, Set2Set) pairs given."""
    keys = S.PRIMARY + S.CONTROL + S.SUPPORT
    acc = {k: [] for k in keys}
    fracs = []
    for a, b in pairs:
        if len(a["ok"]) != len(b["ok"]) or not np.allclose(
                a["r_mult"], b["r_mult"], equal_nan=True):
            continue
        both = a["ok"] & b["ok"]
        mr = a["r_mult"]
        cc = both & (a["g_mult"] == mr) & (b["g_mult"] == mr)
        fracs.append(cc.sum() / both.sum())
        for k in keys:
            acc[k].append([relerr(x[f"r_{k}"], x[f"g_{k}"], sel)
                           for x, sel in [(a, both), (b, both),
                                          (a, cc), (b, cc)]])
    return acc, fracs


def _count_table(title, pairs, md):
    acc, fracs = _count_rows(pairs)
    if not fracs:
        md += [f"### {title}", "", "_The paired runs were scored on "
               "different jet lists — cannot compare._", ""]
        return
    md += [f"### {title}", "",
           f"{len(fracs)} StatsPool/Set2Set pair(s).  Count-correct = jets "
           f"where BOTH models got exactly the right number of particles "
           f"({100 * np.mean(fracs):.1f}% of jets).", "",
           "| metric | StatsPool, all | Set2Set, all | gap, all | "
           "StatsPool, count-correct | Set2Set, count-correct | "
           "gap, count-correct | share of gap explained by count |",
           "|---|---|---|---|---|---|---|---|"]
    for k, vals in acc.items():
        v = np.array([[np.nan if x is None else x for x in r] for r in vals])
        ea, eb, ca, cb = np.nanmean(v, 0)
        g_all, g_cc = eb - ea, cb - ca
        share = ("—" if abs(g_all) < 1e-9 or np.isnan(g_all + g_cc)
                 else f"{100 * (1 - g_cc / g_all):.0f}%")
        c = lambda x, sign="": ("—" if np.isnan(x)             # noqa: E731
                                else f"{x:{sign}.4f}")
        md.append(f"| {S.LABELS.get(k, k)} | {c(ea)} | {c(eb)} | "
                  f"{c(g_all, '+')} | {c(ca)} | {c(cb)} | {c(g_cc, '+')} | "
                  f"{share} |")
    md.append("")


def report_count(out, md):
    md += ["## 4. Attention — is it all the particle count?", "",
           "_Share near 100%: getting the count wrong explains the gap.  Near "
           "0%: Set2Set is worse even when it counts right, so the gap has "
           "another cause._", ""]
    found = False
    r512 = _obs_runs(out, "pooling", ARGS.pooling)
    st = [v for (p, _), v in r512.items() if p == "stats"]
    s2 = [v for (p, _), v in r512.items() if p == "set2set"]
    if st and s2:
        found = True
        _count_table("Latent 512 (round 1, one seed)",
                     [(st[0][1], s2[0][1])], md)
    r64 = _obs_runs(out, "reach", ARGS.reach)
    pairs = [(r64[("stats", sd)][1], r64[("set2set", sd)][1])
             for (p, sd) in sorted(r64) if p == "set2set"
             and ("stats", sd) in r64]
    if pairs:
        found = True
        _count_table("Latent 64 (paired by seed)", pairs, md)
    if not found:
        md += ["_Needs GraphSAGE StatsPool and Set2Set runs._", ""]


def report_smooth(out, md):
    rs = [jload(p) for p in sorted(glob.glob(os.path.join(out, "_smooth",
                                                           "*.json")))]
    md += ["## 5. Over-smoothing — do more hops make particles look alike?",
           ""]
    rs = [r for r in rs if r["group"] == "reach"]
    if not rs:
        md += ["_Not run yet._", ""]
        return
    md += ["Mean cosine similarity between the embeddings of two particles "
           "of the same jet (1 = identical, 0 = unrelated), right before and "
           "right after message passing.  Mean ± sd over seeds.", "",
           "| model | hops | before | after | change |", "|---|---|---|---|---|"]
    for m in REACH_ORDER:
        ms_ = [r for r in rs if r["model"] == m]
        if not ms_:
            continue
        b = [r["cos_before"] for r in ms_]; a = [r["cos_after"] for r in ms_]
        d = [x - y for x, y in zip(a, b)]
        md.append(f"| {m} | {ms_[0]['hops']} | {pm(b)} | {pm(a)} | {pm(d)} |")
    md += ["", "_If 'after' rises with hops (K=2 < K=4 < K=6), longer reach "
           "blurs particles together — a known cost of deep message passing "
           "that fits K=6 decoding slightly worse._", ""]


def fig_probe(probes, out):
    reach = [p for p in probes if p["group"] == "reach"]
    if not reach:
        return
    models = [m for m in REACH_ORDER if any(p["model"] == m for p in reach)]
    keys = ["ecf2_b1", "eec_wide", "log_d2", "ecf2_b2", "girth", "ptD"]
    colours = COLOURS
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    x = np.arange(len(keys)); w = 0.8 / len(models)
    for ax, field, title in [(axes[0], "recon", "decoder reconstruction"),
                             (axes[1], "probe_mlp", "probe on the latent")]:
        for i, m in enumerate(models):
            ps = [p for p in reach if p["model"] == m]
            mu = [msd([p["targets"][k][field] for p in ps])[0] or 0
                  for k in keys]
            sd = [msd([p["targets"][k][field] for p in ps])[1] or 0
                  for k in keys]
            ax.bar(x + (i - (len(models) - 1) / 2) * w, mu, w, yerr=sd,
                   capsize=3, color=colours.get(m), label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([S.LABELS.get(k, k) for k in keys], rotation=20,
                           ha="right")
        ax.set_title(title); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("paired relative error (lower = better)")
    axes[0].legend()
    fig.suptitle("Test B — what the latent holds vs what comes back out "
                 "(latent 64, held-out half of the test split)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_probe.png"), dpi=140)
    plt.close()


def step_report(runs, out):
    md = ["# Diagnostics — why reach doesn't help", "",
          "Every error is the suite's paired relative error on the JetNet "
          "test split (0 = perfect, lower = better).", ""]
    report_d2(runs, md)
    reach_suites = [jload(os.path.join(d, "suite.json"))
                    for g, _, d, c in runs if g == "reach"
                    and c["pool"] == "stats"
                    and os.path.exists(os.path.join(d, "suite.json"))]
    report_factor(out, reach_suites, md)
    report_learned(out, md)
    probes = report_probe(out, md)
    report_count(out, md)
    report_smooth(out, md)
    fig_probe(probes, out)
    path = os.path.join(out, "diagnostics.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    log("\n".join(md))
    log(f"\nwritten -> {path}")


# ══════════════════════════════════════════════════════════════════════════
ARGS = None


def main():
    global ARGS
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pooling", help="round-1 folder (Experiment 1)")
    ap.add_argument("--compression", help="scout folder (Experiment 2)")
    ap.add_argument("--reach", help="v17 results folder (Experiment 3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default=",".join(ALL_STEPS),
                    help="comma list from: " + ",".join(ALL_STEPS))
    ap.add_argument("--n-eval", type=int, default=0,
                    help="test jets; 0 = the whole test split")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--probe-epochs", type=int, default=60)
    ap.add_argument("--smooth-jets", type=int, default=5000,
                    help="test jets used for the over-smoothing measure")
    ap.add_argument("--factor-train-jets", type=int, default=200000)
    ap.add_argument("--gbdt-iters", type=int, default=500)
    ap.add_argument("--learned-train-jets", type=int, default=200000)
    ap.add_argument("--learned-epochs", type=int, default=15)
    ap.add_argument("--learned-seeds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="redo steps whose outputs already exist")
    ARGS = args = ap.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = set(steps) - set(ALL_STEPS)
    if bad:
        ap.error(f"unknown steps: {sorted(bad)}")
    out = args.out
    for sub in ["", "_obs", "_probe", "_smooth"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    log("=" * 76)
    log(f"DIAGNOSTICS  |  steps={','.join(steps)}  |  device={A.device}")
    log("=" * 76)

    runs = (find_runs(args.pooling, "pooling")
            + find_runs(args.compression, "compression")
            + find_runs(args.reach, "reach"))
    log(f"  runs found: {len(runs)}")
    for g, n, _, c in runs:
        log(f"    {g:12s} {n:32s} {model_name(c):20s} pool={c['pool']:8s} "
            f"latent={c['latent_dim']}")

    if {"rescore", "probe", "count", "smooth"} & set(steps):
        log("\n=== STEPS 1/3/4/5: one pass per trained model ===")
        for run in runs:
            per_run_pass(run, steps, out, args)
    if "factor" in steps:
        step_factor(out, args)
    if "learned" in steps:
        step_learned(out, args)
    if "report" in steps:
        step_report(runs, out)


if __name__ == "__main__":
    main()
