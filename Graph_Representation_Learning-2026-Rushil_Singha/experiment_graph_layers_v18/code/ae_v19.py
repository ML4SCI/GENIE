#!/usr/bin/env python
"""
ae_v19.py — v19 autoencoder: graph decoder with a count head.
=============================================================

One script: train -> pick the best epoch on the VALIDATION split -> score the
TEST split with the v17 metric suite and the JetNet ceiling vs EPiC-GAN.

WHAT CHANGED FROM v17 (and why)
-------------------------------
  [1] Latent 128 (was 64).  Still smaller than a typical jet (~169 numbers),
      reconstructs better, and is a comfortable size for a latent flow later.
  [2] Mean pooling (was StatsPool sum|mean|max|std).  StatsPool handed every
      model jet-wide statistics for free, which could hide a benefit of reach;
      with a plain mean, jet-wide information has to come from message
      passing.  A mean cannot see the particle count, so log N is appended to
      the pooled vector BEFORE it is squeezed into z -- z carries N, and the
      decoder needs nothing but z at generation time.
  [3] Two-headed decoder:
        count head    z -> p(N | z)               (cross-entropy, N = 1..150)
        particle head z, N -> N particles:
            h_i = f(z, q_i) for the first N of 150 learned slots q_i
            -> kNN graph (k=6, undirected) on the decoded positions
            -> the SAME graph layer as the encoder (ChebNet / GraphSAGE)
            -> final (eta, phi, pT)
      Training uses the TRUE N; generation samples N from the count head.
      The graph layer lets decoded particles adjust each other -- the v17
      slot decoder made every particle independently, the main suspect for
      the jet-mass gap to EPiC-GAN.
  [4] Loss:
        L = L_sinkhorn + l_emd * L_emd + l_count * L_count
        L_sinkhorn  particles matched on (eta, phi, 3*pT), equal weights,
                    blur 0.02 (v17: 10*pT, blur 0.05 -- matching was dominated
                    by pT for hard particles, angles barely counted)
        L_emd       energy-flow EMD: particles weighted by pT share, distance
                    dR/0.8 -- the hard particles that set mass and EFPs count
                    most.  A general distance, not one observable.
        L_count     cross-entropy of the count head; it replaces v17's mask and
                    count terms, since the decoder now makes exactly N particles.
      No jet-mass term: it is an evaluation observable.
  [5] Total pT computed the way JetNet defines it, with no head and no loss.
      pT_rel = particle pT / jet pT, and jet pT is the length of the VECTOR
      sum of the particles' transverse momenta, so the plain sum of pT_rel is
      a little above 1 (1.00-1.02) and grows with the jet's width.  The
      decoder outputs pT shares and rescales them so that the decoded jet's
      vector-summed pT is exactly 1 -- the total then follows from the decoded
      angles instead of being forced to 1.  Each run logs a check of this on
      the real training jets.
  [6] The layer comparison.  Everything above stays fixed; only the graph
      layer (in the encoder AND the decoder) changes:
        --conv deepsets            DeepSets: each particle alone, 0 hops
        --conv sage --layers 2     GraphSAGE, 2 hops
        --conv cheb --K 2          ChebNet K=2, 2 hops      (the pilot)
        --conv sage --layers 6     GraphSAGE deep, 6 hops
        --conv cheb --K 4          ChebNet K=4, 6 hops
      DeepSets uses the same Euler residual layer with the neighbour term
      removed (exactly ChebNet with K=1), and its decoder skips the kNN graph.
      Training time is recorded for every run (timing.json).

Reuses from v17: the stable ChebNet / GraphSAGE layers, FiLM blocks, JetNet
loading, the undirected graph caches, load_state and the metric suite.
Reuses from v18: the JetNet ceiling scoring (ceiling_eval.py).

    python ae_v19.py --tag v19_chebK2_lat128_s0 --conv cheb --K 2
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np

warnings.filterwarnings("ignore")

SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/r/rushil13")
V17_DIR = os.environ.get("V17_DIR", f"{SCRATCH}/gsoc_rushil/v17_final")
V18_DIR = os.environ.get("V18_DIR", f"{SCRATCH}/gsoc_rushil/v18_generative_modelling")
SAVE_ROOT = os.environ.get("V19_SAVE_ROOT", f"{SCRATCH}/gsoc_rushil/v19/results")
sys.path.insert(0, V17_DIR)

import ae_core as A                                 # noqa: E402
import suite as S                                   # noqa: E402

import torch                                        # noqa: E402
import torch.nn as nn                               # noqa: E402
import torch.nn.functional as F                     # noqa: E402
import torch.optim as optim                         # noqa: E402
from torch.utils.data import DataLoader             # noqa: E402
from torch_geometric.data import Batch              # noqa: E402
from torch_geometric.utils import softmax as pyg_softmax, to_undirected  # noqa: E402
from geomloss import SamplesLoss                    # noqa: E402

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

device = A.device
N_MAX = 150
JET_R = 0.8

# The v17 ceiling for ChebNet K=2, latent 64 (mean of 3 seeds, v18 report):
# W1M, W1P in 1e-3, W1EFP in 1e-5.
OLD_CEILING = {"gluon": (7.16, 2.39, 9.30), "quark": (6.53, 3.15, 4.73),
               "top": (10.11, 2.50, 35.19)}


# ══════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════
class EulerPointLayer(nn.Module):
    """DeepSets layer: the Euler ChebConv with K=1, i.e. only the T_0 term,
        out = x + eps * scale * (W x + b),   W anti-symmetric,
    so each particle is updated on its own (no message passing).  It reuses
    Euler_ChebConv's weights and constants and only skips the Laplacian,
    which K=1 never uses."""
    def __init__(self, hidden, term_norm):
        super().__init__()
        self.c = A.Euler_ChebConv(hidden, hidden, 1, term_norm=term_norm)

    def forward(self, x, edge_index=None):
        c = self.c
        out = c.lins[0](x)
        if c.bias is not None:
            out = out + c.bias
        return x + c.e * c._scale * out


def make_conv(conv, hidden, K, term_norm):
    if conv == "deepsets":
        return EulerPointLayer(hidden, term_norm)
    if conv == "cheb":
        return A.Euler_ChebConv(hidden, hidden, K, term_norm=term_norm)
    if conv == "sage":
        return A.EulerSAGEConv(hidden, hidden, term_norm=term_norm)
    raise ValueError(conv)


class GraphStack(nn.Module):
    """L graph layers, BatchNorm + LeakyReLU between them (as in v17)."""
    def __init__(self, conv, hidden, K, layers, term_norm):
        super().__init__()
        self.convs = nn.ModuleList([make_conv(conv, hidden, K, term_norm)
                                    for _ in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden)
                                  for _ in range(layers - 1)])

    def forward(self, h, edge_index):
        for i, cv in enumerate(self.convs):
            h = cv(h, edge_index)
            if i < len(self.bns):
                h = F.leaky_relu(self.bns[i](h))
        return h


def mean_pool(h, batch, n_graphs):
    s = h.new_zeros(n_graphs, h.size(1)).index_add_(0, batch, h)
    n = torch.bincount(batch, minlength=n_graphs).clamp(min=1).to(h.dtype)
    return s / n[:, None], n


class EncoderV19(nn.Module):
    """kNN graph -> graph layers -> MEAN pool, + log N -> latent z."""
    def __init__(self, conv, K, layers, hidden=256, latent=128, pos_dim=64,
                 term_norm="sqrt"):
        super().__init__()
        self.input_proj = nn.Linear(3, hidden, bias=False)
        self.graph = GraphStack(conv, hidden, K, layers, term_norm)
        self.out_norm = nn.LayerNorm(hidden)
        self.pos_mlp = nn.Sequential(nn.Linear(2, pos_dim), nn.GELU(),
                                     nn.Linear(pos_dim, pos_dim))
        self.fuse = nn.Linear(hidden + pos_dim, hidden)
        self.head = nn.Sequential(nn.LayerNorm(hidden + 1),
                                  nn.Linear(hidden + 1, hidden), nn.GELU(),
                                  nn.Linear(hidden, latent))

    def forward(self, x, edge_index, batch, n_graphs):
        h = self.graph(self.input_proj(x), edge_index)
        h = self.fuse(torch.cat([self.out_norm(h), self.pos_mlp(x[:, :2])], -1))
        pooled, n = mean_pool(h, batch, n_graphs)
        return self.head(torch.cat([pooled, torch.log(n)[:, None]], -1))


def knn_edges(pos, active, n_per_jet, k):
    """Undirected kNN graph on decoded (eta, phi), batched, no extra package.
    pos [B, P, 2], active [B, P] (the first N slots of each jet).  Returns
    edge_index in the flattened numbering of the active particles."""
    B, P, _ = pos.shape
    with torch.no_grad():
        d = torch.cdist(pos, pos)
        pair_ok = active[:, :, None] & active[:, None, :]
        pair_ok &= ~torch.eye(P, dtype=torch.bool, device=pos.device)[None]
        d = d.masked_fill(~pair_ok, float("inf"))
        kk = min(k, P - 1)
        vals, nbr = d.topk(kk, dim=-1, largest=False)
        ok = active[:, :, None] & torch.isfinite(vals)
        offset = (torch.cumsum(n_per_jet, 0) - n_per_jet)[:, None, None]
        src = (offset + torch.arange(P, device=pos.device)[None, :, None]
               ).expand(B, P, kk)[ok]
        dst = (offset + nbr)[ok]
        ei = torch.stack([src, dst])
        return to_undirected(ei, num_nodes=int(n_per_jet.sum()))


class DecoderV19(nn.Module):
    """Count head p(N|z) + particle head: slots -> kNN graph -> graph layers."""
    def __init__(self, conv, K, layers, latent=128, hidden=256, slot_dim=64,
                 n_res=3, knn=6, pos_dim=64, term_norm="sqrt"):
        super().__init__()
        self.knn = knn
        self.use_graph = conv != "deepsets"      # DeepSets needs no kNN graph
        self.count_head = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                        nn.Linear(hidden, N_MAX))
        self.slot_emb = nn.Embedding(N_MAX, slot_dim)
        self.z_proj = nn.Linear(latent, slot_dim)
        self.stem = nn.Linear(2 * slot_dim, hidden)
        self.blocks = nn.ModuleList([A.FiLMResBlock(hidden, latent)
                                     for _ in range(n_res)])
        self.pos_head = nn.Linear(hidden, 2)
        self.pos_mlp = nn.Sequential(nn.Linear(2, pos_dim), nn.GELU(),
                                     nn.Linear(pos_dim, pos_dim))
        self.fuse = nn.Linear(hidden + pos_dim, hidden)
        self.graph = GraphStack(conv, hidden, K, layers, term_norm)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, 3))

    def count_logits(self, z):
        return self.count_head(z)                       # class i <-> N = i+1

    def forward(self, z, n):
        """z [B, latent], n [B] (long, 1..150) -> padded jets [B, 150, 4]."""
        B = z.size(0)
        slots = torch.cat([self.z_proj(z)[:, None].expand(B, N_MAX, -1),
                           self.slot_emb.weight[None].expand(B, -1, -1)], -1)
        h = self.stem(slots)
        for blk in self.blocks:
            h = blk(h, z)
        pos0 = torch.tanh(self.pos_head(h)) * 2.0                  # [B, P, 2]

        active = torch.arange(N_MAX, device=z.device)[None] < n[:, None]
        batch = torch.repeat_interleave(torch.arange(B, device=z.device), n)
        ei = knn_edges(pos0, active, n, self.knn) if self.use_graph else None

        hb, pb = h[active], pos0[active]
        x = self.fuse(torch.cat([hb, self.pos_mlp(pb)], -1))
        x = self.graph(x, ei)
        o = self.out(x)
        pos = pb + 0.5 * torch.tanh(o[:, :2])          # refine around pos0
        share = pyg_softmax(o[:, 2], batch)            # sum to 1 per jet
        # JetNet's pT_rel: divide by the jet's VECTOR-summed pT, so the plain
        # sum lands at 1.00-1.02 depending on how spread out the jet is
        vx = z.new_zeros(B).index_add_(0, batch, share * torch.cos(pos[:, 1]))
        vy = z.new_zeros(B).index_add_(0, batch, share * torch.sin(pos[:, 1]))
        pt = share / torch.sqrt(vx ** 2 + vy ** 2).clamp(min=1e-6)[batch]

        vals = torch.cat([pos, pt[:, None], torch.ones_like(pt)[:, None]], -1)
        out = z.new_zeros(B, N_MAX, 4)
        out[active] = vals
        return out


# ══════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════
class V19Loss:
    TERMS = ["sinkhorn", "emd", "count"]

    def __init__(self, args):
        self.a = args
        self.sink = SamplesLoss("sinkhorn", p=1, blur=args.blur, debias=True)
        self.emd = SamplesLoss("sinkhorn", p=1, blur=args.emd_blur,
                               scaling=0.9, debias=True)

    @staticmethod
    def _norm_w(w):
        w = w.clamp(min=1e-9)
        return w / w.sum(1, keepdim=True)

    def __call__(self, pred, tgt, logits, n):
        a = self.a
        pm = (pred[:, :, 3] > 0.5).to(pred.dtype)
        tm = (tgt[:, :, 3] > 0.5).to(tgt.dtype)
        # every particle weighted equally, pT as a scaled coordinate
        sc = torch.tensor([1.0, 1.0, a.pt_scale], device=pred.device)
        l_sink = self.sink(self._norm_w(pm), (pred[:, :, :3] * sc).contiguous(),
                           self._norm_w(tm), (tgt[:, :, :3] * sc).contiguous()
                           ).mean()
        # energy-flow EMD: particles weighted by pT share, distance dR/R
        if a.lam_emd > 0:
            l_emd = self.emd(self._norm_w(pred[:, :, 2] * pm),
                             (pred[:, :, :2] / JET_R).contiguous(),
                             self._norm_w(tgt[:, :, 2] * tm),
                             (tgt[:, :, :2] / JET_R).contiguous()).mean()
        else:
            l_emd = l_sink.new_zeros(())
        l_count = F.cross_entropy(logits, n - 1)
        total = l_sink + a.lam_emd * l_emd + a.lam_count * l_count
        return total, {"sinkhorn": l_sink.item(), "emd": l_emd.item(),
                       "count": l_count.item(), "total": total.item()}


# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════
def get_split(split, knn, log, max_jets=None):
    """Undirected kNN graphs for a JetNet split, from the v17 cache when there
    (train / test), built and cached otherwise (valid, on the first run)."""
    path = A.graph_cache_path(split, N_MAX, knn, None, True)
    cached = A.load_graph_cache(path, log)
    if cached is None:
        P, _, t = A.load_jetnet_data(N_MAX, None, split=split, log=log)
        cached = A.build_graphs(P, t, knn, N_MAX, path, log=log, symmetric=True)
    g, tg, ty = cached
    ty = np.asarray(ty)
    if max_jets and max_jets < len(g):
        # The cache is stored gluon, quark, top, so the first jets would all
        # be gluons: take a fixed random subset across all three types.
        idx = np.sort(np.random.default_rng(0).choice(len(g), max_jets,
                                                      replace=False))
        g, tg, ty = [g[i] for i in idx], [tg[i] for i in idx], ty[idx]
    return g, tg, ty


def loader(graphs, targets, bs, shuffle):
    return DataLoader(A.JetGraphDataset(graphs, targets), batch_size=bs,
                      shuffle=shuffle, collate_fn=A.collate_fn,
                      drop_last=shuffle,
                      pin_memory=torch.cuda.is_available())


def step(enc, dec, bg, bt, loss_fn):
    bg, bt = bg.to(device, non_blocking=True), bt.to(device, non_blocking=True)
    n = (bt[:, :, 3] > 0.5).sum(1)
    z = enc(bg.x, bg.edge_index, bg.batch, bt.size(0))
    pred = dec(z, n)                                   # true N while training
    return loss_fn(pred, bt, dec.count_logits(z), n)


@torch.no_grad()
def validate(enc, dec, graphs, targets, loss_fn, bs):
    enc.eval(); dec.eval()
    sums, nb = {}, 0
    for bg, bt in loader(graphs, targets, bs, shuffle=False):
        _, parts = step(enc, dec, bg, bt, loss_fn)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
        nb += 1
    enc.train(); dec.train()
    return {k: v / max(nb, 1) for k, v in sums.items()}


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════
def train(enc, dec, tr, va, loss_fn, args, out_dir, log):
    params = list(enc.parameters()) + list(dec.parameters())
    opt = optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1),
                                               eta_min=1e-6)
    ckpt = os.path.join(out_dir, "ckpt_latest.pth")
    start, hist, best = 0, {"train": [], "val": []}, float("inf")
    # wall-clock seconds, kept in the checkpoint so a resumed run adds up
    timing = {"train_s": 0.0, "val_s": 0.0, "epochs_timed": 0}
    if os.path.exists(ckpt):
        try:
            ck = torch.load(ckpt, map_location=device, weights_only=False)
            A.load_state(enc, ck["enc"]); A.load_state(dec, ck["dec"])
            opt.load_state_dict(ck["opt"])
            start, hist, best = ck["epoch"], ck["hist"], ck["best"]
            timing = ck.get("timing", timing)
            for _ in range(start):
                sch.step()
            log(f"  resumed from epoch {start}")
        except Exception as e:                                # noqa: BLE001
            log(f"  !!! RESUME FAILED ({e}) -- training from scratch")

    dl = loader(*tr, args.batch_size, shuffle=True)
    log(f"  train jets={len(tr[0])}  batches/epoch={len(dl)}  "
        f"epochs {start}->{args.epochs}")
    enc.train(); dec.train()
    for ep in range(start, args.epochs):
        sums, t0 = {}, time.time()
        for bi, (bg, bt) in enumerate(dl):
            loss, parts = step(enc, dec, bg, bt, loss_fn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            if (bi + 1) % 200 == 0:
                log(f"    E{ep+1} B{bi+1}/{len(dl)} " +
                    " ".join(f"{k}={v:.4f}" for k, v in parts.items()))
        sch.step()
        ep_s = time.time() - t0
        timing["train_s"] += ep_s
        timing["epochs_timed"] += 1
        avg = {k: v / len(dl) for k, v in sums.items()}
        hist["train"].append(avg)
        log(f"  epoch {ep+1:4d}/{args.epochs}  " +
            " ".join(f"{k}={v:.4f}" for k, v in avg.items()) +
            f"  lr={sch.get_last_lr()[0]:.2e}  ({ep_s:.0f}s)")

        if (ep + 1) % args.val_every == 0 or (ep + 1) == args.epochs:
            tv = time.time()
            v = validate(enc, dec, *va, loss_fn, args.batch_size)
            timing["val_s"] += time.time() - tv
            hist["val"].append({"epoch": ep + 1, **v})
            tag = ""
            if v["total"] < best:
                best = v["total"]
                torch.save(enc.state_dict(), os.path.join(out_dir, "encoder_best.pth"))
                torch.save(dec.state_dict(), os.path.join(out_dir, "decoder_best.pth"))
                tag = "  <- best so far"
            log(f"    VALID  " + " ".join(f"{k}={x:.4f}" for k, x in v.items())
                + tag)

        if (ep + 1) % args.ckpt_every == 0 or (ep + 1) == args.epochs:
            torch.save({"epoch": ep + 1, "enc": enc.state_dict(),
                        "dec": dec.state_dict(), "opt": opt.state_dict(),
                        "hist": hist, "best": best, "timing": timing}, ckpt)

    torch.save(enc.state_dict(), os.path.join(out_dir, "encoder_last.pth"))
    torch.save(dec.state_dict(), os.path.join(out_dir, "decoder_last.pth"))
    plot_losses(hist, os.path.join(out_dir, "loss_terms.png"), args.tag)
    return hist, timing


def plot_losses(hist, path, tag):
    if not hist["train"]:
        return
    terms = V19Loss.TERMS + ["total"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, k in zip(axes.flat, terms):
        ax.plot([h[k] for h in hist["train"]], label="train")
        if hist["val"]:
            ax.plot([h["epoch"] - 1 for h in hist["val"]],
                    [h[k] for h in hist["val"]], "o-", ms=3, label="valid")
        ax.set_title(k); ax.set_yscale("log"); ax.grid(alpha=.3)
    axes.flat[0].legend()
    fig.suptitle(f"v19 loss terms — {tag}")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def reconstruct(enc, dec, graphs, targets, bs):
    """Teacher-forced reconstruction (true N) + what the count head predicts."""
    enc.eval(); dec.eval()
    recon, real, zs, n_pred = [], [], [], []
    for i in range(0, len(graphs), bs):
        bg = Batch.from_data_list(graphs[i:i + bs]).to(device)
        bt = torch.stack(targets[i:i + bs]).to(device)
        n = (bt[:, :, 3] > 0.5).sum(1)
        z = enc(bg.x, bg.edge_index, bg.batch, bt.size(0))
        recon.append(dec(z, n).cpu().numpy())
        real.append(bt.cpu().numpy())
        zs.append(z.cpu())
        n_pred.append((dec.count_logits(z).argmax(1) + 1).cpu().numpy())
    return (np.concatenate(recon), np.concatenate(real), torch.cat(zs),
            np.concatenate(n_pred))


@torch.no_grad()
def time_decoder(dec, Z, n_jets=200000):
    """Generation-mode decoder time: count head picks N, then decode."""
    if not torch.cuda.is_available():
        return None
    dec.eval(); best = None
    for bs in [2048, 8192]:
        z = Z[:bs].to(device)
        if len(z) < bs:
            z = z.repeat(math.ceil(bs / len(z)), 1)[:bs]
        for _ in range(3):
            dec(z, dec.count_logits(z).argmax(1) + 1)
        torch.cuda.synchronize()
        it = max(1, n_jets // bs); t0 = time.perf_counter()
        for _ in range(it):
            dec(z, dec.count_logits(z).argmax(1) + 1)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / (it * bs) * 1e6
        if best is None or us < best[0]:
            best = (us, bs)
    return {"us_per_jet": best[0], "batch_size": best[1],
            "gpu": torch.cuda.get_device_name(0)}


def ceiling(recon, real, types, args, log):
    """JetNet W1-M / W1-P / W1-EFP per jet type, exactly as v18."""
    sys.path.insert(0, V18_DIR)
    import ceiling_eval as CE                        # noqa: E402
    import jetnet.evaluation as E                    # noqa: E402
    import jetnet.evaluation.gen_metrics as gm       # noqa: E402
    cfg = SimpleNamespace(seed=args.seed, n_eval_samples=args.n_eval_samples,
                          batches=5, efp_jobs=args.efp_jobs)
    ref = CE.train_reference(A, args.ref_per_type, args.seed)
    rj, rm = CE.to_jetnet(recon)
    tj, tm = CE.to_jetnet(real)
    res = {"ceiling": {}, "truth": {}}
    for tid, t in enumerate(CE.TYPES):
        s = types == tid
        res["ceiling"][t] = CE.score(E, gm, rj[s], rm[s], *ref[t], cfg)
        res["truth"][t] = CE.score(E, gm, tj[s], tm[s], *ref[t], cfg)
        c = res["ceiling"][t]
        log(f"  ceiling {t:5s}: W1M={c['w1m'] * 1e3:.2f}e-3  "
            f"W1P={c['w1p'] * 1e3:.2f}e-3  W1EFP={c['w1efp'] * 1e5:.2f}e-5")
    return res, CE


def write_report(res, meta, CE, path):
    md = [f"# v19 reconstruction ceiling — {meta['tag']}", "",
          f"{meta['model']} · {meta['receptive_field_hops']} hops · latent "
          f"{meta['latent_dim']} · mean pool + log N · decoder: {meta['decoder']} · "
          f"Sinkhorn + EMD + count loss · best epoch {meta['best_epoch']} "
          f"(validation) · {meta['n_test']:,} test jets", "",
          "W1ᴹ and W1ᴾ in 10⁻³, W1ᴱᶠᴾ in 10⁻⁵; lower is better. **old** = v17 "
          "ChebNet K=2 at latent 64 (mean of 3 seeds); **EPiC** = EPiC-GAN, "
          "JetNet150 Table 3.", "",
          "| jet type | W1ᴹ new | old | EPiC | ×EPiC | W1ᴾ new | old | EPiC | "
          "W1ᴱᶠᴾ new | old | EPiC | ×EPiC |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in CE.TYPES:
        c = res["ceiling"][t]
        m, p, e = c["w1m"] * 1e3, c["w1p"] * 1e3, c["w1efp"] * 1e5
        om, op, oe = OLD_CEILING[t]
        em, ep, ee = CE.EPIC[t]
        md.append(f"| {t} | **{m:.2f}** | {om:.2f} | {em} | {m / em:.1f}× | "
                  f"**{p:.2f}** | {op:.2f} | {ep} | **{e:.2f}** | {oe:.2f} | "
                  f"{ee} | {e / ee:.1f}× |")
    md += ["", "Truth floor (real test vs real train): " + " · ".join(
        f"{t} W1ᴹ {res['truth'][t]['w1m'] * 1e3:.2f}" for t in CE.TYPES), "",
        f"Count head (generation mode): exact N on "
        f"{100 * meta['count_exact']:.1f}% of test jets, mean |ΔN| "
        f"{meta['count_mae']:.2f}. (Reconstructions use the true N, so the "
        f"particle-count numbers in suite.json are trivially perfect; this "
        f"count-head accuracy is the real test.)", ""]
    if meta.get("decoder_timing"):
        ti = meta["decoder_timing"]
        md.append(f"Decoder in generation mode (count head + particle head): "
                  f"**{ti['us_per_jet']:.2f} µs/jet** (batch {ti['batch_size']}, "
                  f"{ti['gpu']}); EPiC-GAN ~12 µs/jet.")
    if meta.get("training_time"):
        tt = meta["training_time"]
        md.append(f"Training time: **{tt['train_h']:.2f} h** for "
                  f"{meta['epochs']} epochs ({tt['s_per_epoch']:.0f} s/epoch, "
                  f"{tt['gpu']}); parameters {meta['params_total']:,}.")
    md += ["", "**Go / no-go for the flow stage:** within about 2× of EPiC-GAN "
           "on W1ᴹ and W1ᴱᶠᴾ for every jet type.",
           "", "_Loss: Sinkhorn + energy-flow EMD + count. The EMD is a "
           "training target, so the suite's EMD metric is not independent; "
           "jet mass, the pairwise metrics and the JetNet W1 scores are not "
           "in the loss._"]
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    return "\n".join(md)


def training_time(timing, epochs):
    """Training cost: pure training epochs (no validation, no evaluation).
    If a run was resumed from a checkpoint written before timing was added,
    only the timed epochs are known, so the total is extrapolated."""
    n = max(timing["epochs_timed"], 1)
    per = timing["train_s"] / n
    return {"train_s": timing["train_s"], "val_s": timing["val_s"],
            "epochs_timed": timing["epochs_timed"], "s_per_epoch": per,
            "train_h": per * epochs / 3600,
            "extrapolated": timing["epochs_timed"] < epochs,
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "cpu")}


def total_pt_check(targets, log, n=20000):
    """Does 'divide by the vector-summed pT' reproduce JetNet's real totals?
    Logged once per run, on real training jets."""
    idx = np.linspace(0, len(targets) - 1, min(n, len(targets))).astype(int)
    arr = torch.stack([targets[i] for i in idx]).numpy()   # all three types
    m = arr[:, :, 3] > 0.5
    pt = np.where(m, arr[:, :, 2], 0.0)
    real = pt.sum(1)
    w = pt / np.clip(real[:, None], 1e-12, None)
    vec = np.hypot((w * np.cos(arr[:, :, 1])).sum(1),
                   (w * np.sin(arr[:, :, 1])).sum(1))
    model = 1.0 / np.clip(vec, 1e-12, None)
    r = float(np.corrcoef(real, model)[0, 1]) if real.std() > 0 else float("nan")
    log(f"  total-pT check on {len(real)} real jets: real sum "
        f"{real.mean():.4f} ± {real.std():.4f}, JetNet-way {model.mean():.4f} "
        f"± {model.std():.4f}, mean |diff| {np.abs(real - model).mean():.4f}, "
        f"corr {r:.3f}")


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v19_chebK2_lat128_s0")
    ap.add_argument("--conv", choices=["deepsets", "sage", "cheb"],
                    default="cheb")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2,
                    help="graph layers in the encoder AND the decoder")
    ap.add_argument("--knn", type=int, default=6)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--term-norm", default="sqrt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pt-scale", type=float, default=3.0,
                    help="pT weight in the Sinkhorn coordinates (v17: 10)")
    ap.add_argument("--blur", type=float, default=0.02, help="v17: 0.05")
    ap.add_argument("--lam-emd", type=float, default=1.0,
                    help="weight of the energy-flow EMD term; 0 turns it off")
    ap.add_argument("--emd-blur", type=float, default=0.01)
    ap.add_argument("--lam-count", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--n-val", type=int, default=20000,
                    help="validation jets used for model selection")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--max-train", type=int, default=0, help="debug: cap jets")
    ap.add_argument("--max-test", type=int, default=0, help="debug: cap jets")
    ap.add_argument("--n-eval-samples", type=int, default=50000)
    ap.add_argument("--ref-per-type", type=int, default=50000)
    ap.add_argument("--efp-jobs", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--skip-ceiling", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = os.path.join(SAVE_ROOT, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    log = A.Logger(os.path.join(out_dir, "log.txt"))
    log("=" * 76)
    log(f"v19 RUN: {args.tag}   device={device}")
    log("=" * 76)
    for k, v in vars(args).items():
        log(f"  {k:16s} = {v}")

    # ── data ───────────────────────────────────────────────────────────────
    tr = get_split("train", args.knn, log, args.max_train or None)
    va = get_split("valid", args.knn, log, args.n_val)
    te = get_split("test", args.knn, log, args.max_test or None)
    log(f"  jets: train={len(tr[0])}  valid={len(va[0])}  test={len(te[0])}")
    total_pt_check(tr[1], log)


    # ── model ──────────────────────────────────────────────────────────────
    enc = EncoderV19(args.conv, args.K, args.layers, latent=args.latent_dim,
                     term_norm=args.term_norm).to(device)
    dec = DecoderV19(args.conv, args.K, args.layers, latent=args.latent_dim,
                     knn=args.knn, term_norm=args.term_norm).to(device)
    n_enc, n_dec = A.count_params(enc), A.count_params(dec)
    hops = {"deepsets": 0, "sage": args.layers,
            "cheb": args.layers * (args.K - 1)}[args.conv]
    model = {"deepsets": "DeepSets", "sage": "GraphSAGE",
             "cheb": f"ChebNet K={args.K}"}[args.conv]
    if args.conv == "sage" and args.layers != 2:
        model += f" ({args.layers} layers)"
    log(f"  {model}, {args.layers} layers ({hops} hops) in encoder and decoder")
    log(f"  params: encoder={n_enc:,}  decoder={n_dec:,}  total={n_enc + n_dec:,}")
    loss_fn = V19Loss(args)

    # ── train, then load the best validation epoch ─────────────────────────
    hist, timing = train(enc, dec, tr[:2], va[:2], loss_fn, args, out_dir, log)
    t_eval = time.time()
    best_path = os.path.join(out_dir, "encoder_best.pth")
    if os.path.exists(best_path):
        A.load_state(enc, torch.load(best_path, map_location=device))
        A.load_state(dec, torch.load(os.path.join(out_dir, "decoder_best.pth"),
                                     map_location=device))
    best_epoch = (min(hist["val"], key=lambda v: v["total"])["epoch"]
                  if hist["val"] else args.epochs)
    log(f"\n  evaluating the best validation epoch ({best_epoch}) on the TEST split")

    # ── evaluate ───────────────────────────────────────────────────────────
    g, t, ty = te
    recon, real, Z, n_pred = reconstruct(enc, dec, g, t, args.batch_size)
    n_true = (real[:, :, 3] > 0.5).sum(1)
    meta = {"tag": args.tag, "model": model, "conv": args.conv, "K": args.K,
            "layers": args.layers, "knn": args.knn, "pool": "mean+logN",
            "decoder": ("count head + slots + per-particle layers"
                        if args.conv == "deepsets"
                        else "count head + slots + kNN graph layers"),
            "latent_dim": args.latent_dim, "graph_symmetric": True,
            "term_norm": args.term_norm, "seed": args.seed,
            "epochs": args.epochs, "best_epoch": best_epoch,
            "receptive_field_hops": hops, "params_encoder": n_enc,
            "params_decoder": n_dec, "params_total": n_enc + n_dec,
            "loss": {k: getattr(args, k) for k in
                     ["pt_scale", "blur", "lam_emd", "emd_blur",
                      "lam_count"]},
            "n_train": len(tr[0]), "n_test": len(g),
            "count_exact": float(np.mean(n_pred == n_true)),
            "count_mae": float(np.mean(np.abs(n_pred - n_true))),
            "final_train_loss": hist["train"][-1]["total"] if hist["train"] else None}
    log(f"  count head: exact {100 * meta['count_exact']:.1f}%  "
        f"mean |dN| {meta['count_mae']:.2f}")

    meta["decoder_timing"] = time_decoder(dec, Z)
    meta["training_time"] = training_time(timing, args.epochs)
    log(f"  training time: {meta['training_time']['train_h']:.2f} h "
        f"({meta['training_time']['s_per_epoch']:.0f} s/epoch over "
        f"{timing['epochs_timed']} timed epochs) + validation "
        f"{timing['val_s'] / 60:.1f} min")
    suite, (o_r, o_g, ok) = S.score(real, recon, args.latent_dim, ty, device,
                                    args.batch_size, log)
    suite.update(meta)
    with open(os.path.join(out_dir, "suite.json"), "w") as f:
        json.dump(suite, f, indent=2)
    vis = A.select_vis_indices(ty)
    A.plot_recon_pairs(real[vis], recon[vis],
                       os.path.join(out_dir, "recon_pairs.png"),
                       n_show=len(vis), jet_types=ty[vis], tag=args.tag)
    A.plot_particle_distributions(real, recon,
                                  os.path.join(out_dir, "dist_comparison.png"),
                                  tag=args.tag)
    A.plot_observables(o_r, o_g, os.path.join(out_dir, "observables.png"),
                       tag=args.tag)
    log("  suite: " + "  ".join(f"{k}={suite[k]:.4f}" for k in
                                 ["ecf2_b1", "eec_wide", "girth", "mass",
                                  "ecf2_b2", "ptD", "emd"] if suite.get(k) is not None))

    if not args.skip_ceiling:
        res, CE = ceiling(recon, real, ty, args, log)
        with open(os.path.join(out_dir, "ceiling.json"), "w") as f:
            json.dump({**res, "meta": meta}, f, indent=2)
        log("\n" + write_report(res, meta, CE,
                                os.path.join(out_dir, "ceiling_report.md")))
    meta["training_time"]["eval_s"] = time.time() - t_eval
    meta["training_time"]["job_wall_s"] = time.time() - t_start
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump({"tag": args.tag, "model": model,
                   "params_total": n_enc + n_dec, **meta["training_time"]},
                  f, indent=2)
    log(f"  evaluation {meta['training_time']['eval_s'] / 60:.1f} min; "
        f"this job's wall time {meta['training_time']['job_wall_s'] / 3600:.2f} h")
    log(f"\n  saved -> {out_dir}")


if __name__ == "__main__":
    main()
