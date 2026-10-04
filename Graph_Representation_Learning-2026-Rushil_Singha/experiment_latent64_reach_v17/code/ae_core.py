"""
ae_core.py — v17_final.  Model, data, training and evaluation.
==============================================================

Graph autoencoder for JetNet-150 jets:

    jet -> kNN graph -> encoder (GraphSAGE or ChebNet) -> pooling
        -> latent (the bottleneck) -> slot decoder -> reconstructed jet

WHAT CHANGED IN v17, AND WHY
----------------------------
  [1] UNDIRECTED kNN GRAPHS (bug fix).  Before v17 the graph was directed;
      57% of edges had no reverse edge.  That makes the ChebNet Laplacian's
      spectrum complex, and Chebyshev terms then GROW with order instead of
      staying bounded -- a penalty rising with K that has nothing to do with
      reach (~1.4x at K=6 on a typical signal, up to ~3x worst case).
      See build_graphs().
  [2] ENERGY-FLOW EMD (correctness).  The old "EMD" weighted every particle
      equally and used pT as a coordinate -- not the metric Hariri et al.
      used.  suite.emd_energy_flow now implements the Komiske-Metodiev-Thaler
      EMD: pT-weighted transport, distance dR/R.
  [3] ONE METRIC SUITE (suite.py), shared by training-time evaluation and by
      reeval.py, so every number in the paper comes from the same code.
  [4] Set2Set can run at any latent (projection after pooling).
  [5] Kept from v16: 1/sqrt(K) term normalisation, anti-symmetric Euler
      convolutions, StatsPool, FiLM slot decoder, JetNet's own train/test
      splits.

Entry point: run_one(cfg), or from the shell:
    python ae_core.py --tag demo --conv cheb --K 6 --latent-dim 64
"""

import os
import json
import math
import pickle
import warnings
import argparse
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module, Parameter
from torch.nn.utils.parametrize import register_parametrization
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.nn.aggr import Set2Set
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear as PyGLinear
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import get_laplacian
try:
    from torch_geometric.utils import scatter as pyg_scatter
except ImportError:      # very old PyG
    pyg_scatter = None

from sklearn.neighbors import kneighbors_graph
from scipy.stats import wasserstein_distance
from geomloss import SamplesLoss
from jetnet.datasets import JetNet

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# PATHS — edit these two (or set env vars JETNET_DATA_DIR / ABLATION_SAVE_ROOT)
# ══════════════════════════════════════════════════════════════════════════
JETNET_DATA_DIR = os.environ.get(
    "JETNET_DATA_DIR", "/pscratch/sd/r/rushil13/jetnet_data")
SAVE_ROOT = os.environ.get(
    "ABLATION_SAVE_ROOT", "/pscratch/sd/r/rushil13/gsoc_rushil/v16_ablation/results")
CACHE_ROOT = os.path.join(SAVE_ROOT, "_graph_cache")

os.makedirs(JETNET_DATA_DIR, exist_ok=True)
os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(CACHE_ROOT, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════
# FIXED HYPERPARAMETERS (identical for every ablation run — do not sweep)
# ══════════════════════════════════════════════════════════════════════════
GNN_HIDDEN_DIM   = 256
LATENT_DIM       = 512          # PINNED. The bottleneck is the instrument.
CHEB_STEP_SIZE   = 0.45
CHEB_DISSIPATION = 0.1
POS_DIM          = 64
SET2SET_STEPS    = 4
SLOT_DIM         = 64
DECODER_HIDDEN   = 256
PT_SCALE         = 10.0
ETA_BOUND        = 2.0
PHI_BOUND        = 2.0
COUNT_LOSS_WEIGHT = 0.5

# Evaluation uses JetNet's OWN test split (split="test"), not a slice of the
# training data.  jetnet's default split_fraction is [0.7, 0.15, 0.15], so
# "train" is 70% and "test" is the held-out 15% -- a genuinely unseen sample.
JETNET_SPLIT_FRACTION = [0.7, 0.15, 0.15]

# Jets shown in the real-vs-reconstruction figure.  Chosen deterministically
# (first N of each jet type in the test set) so that EVERY model in the
# ablation reconstructs the SAME jets and the figures are comparable.
VIS_PER_TYPE = 2

sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=0.05, debias=True)


# ══════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════
class Logger:
    def __init__(self, path):
        self.path = path
        with open(self.path, "w") as f:
            f.write("")

    def __call__(self, msg):
        print(msg, flush=True)
        with open(self.path, "a") as f:
            f.write(str(msg) + "\n")


# ══════════════════════════════════════════════════════════════════════════
# STABLE CONVOLUTIONS
# ══════════════════════════════════════════════════════════════════════════
class AntiSymmetric(Module):
    """W = W_upper - W_upper^T - g*I  ->  eigenvalues in the left half-plane.

    This is what makes large K numerically safe: without it, stacking many
    Chebyshev terms blows up.  Used for BOTH conv types so the ChebNet-vs-
    1-hop comparison is not contaminated by a stability difference.
    """
    def __init__(self, dissipative_force: float = 0.0):
        super().__init__()
        self.g = dissipative_force

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return (W.triu(diagonal=1)
                - W.triu(diagonal=1).T
                - self.g * torch.eye(W.shape[0], device=W.device))

    def right_inverse(self, W: torch.Tensor) -> torch.Tensor:
        return W.triu(diagonal=1)


class Euler_ChebConv(MessagePassing):
    """out = x + eps * sum_{k=0}^{K-1} W_k T_k(L~) x

    K convention (state this in the paper):
      K=1 -> only T_0            -> ZERO message passing (floor)
      K=2 -> adds T_1 = L~x      -> exactly ONE hop
      K=n -> (n-1) hops per layer
    """
    def __init__(self, in_channels, out_channels, K,
                 step_size=CHEB_STEP_SIZE, dissipation_force=CHEB_DISSIPATION,
                 bias=True, term_norm="sqrt", **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        assert K > 0
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.normalization = "sym"
        self.e = step_size
        self.g = dissipation_force

        self.lins = nn.ModuleList()
        for _ in range(K):
            lin = PyGLinear(in_channels, out_channels,
                            bias=False, weight_initializer="glorot")
            register_parametrization(lin, "weight",
                                     AntiSymmetric(dissipative_force=self.g))
            self.lins.append(lin)

        self.bias = Parameter(torch.Tensor(out_channels)) if bias else None
        self._scale = self._term_scale(K, term_norm)
        self.term_norm = term_norm
        self.reset_parameters()

    @staticmethod
    def _term_scale(n_terms, mode):
        if mode == "sqrt":  return float(n_terms) ** -0.5
        if mode == "linear": return 1.0 / float(n_terms)
        if mode == "none":   return 1.0
        raise ValueError(f"unknown term_norm {mode!r}")

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins[1:]:
            lin.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def _norm(self, edge_index, num_nodes, edge_weight, dtype):
        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, self.normalization, dtype, num_nodes)
        lambda_max = 2.0 * edge_weight.max()
        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float("inf"), 0)
        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1
        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        edge_index, norm = self._norm(
            edge_index, x.size(self.node_dim), edge_weight, x.dtype)

        Tx_0 = x
        Tx_1 = x
        out  = self.lins[0](Tx_0)

        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out  = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = 2.0 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            out  = out + lin(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        # CRITICAL: `out` is a SUM of K terms, so without normalisation its
        # magnitude grows with K and the Euler step  x + e*out  stops being a
        # SMALL perturbation of the identity.  Measured ||e*out|| / ||x|| at
        # init with e=0.45:
        #     K=1 -> 0.45   K=2 -> 0.48   K=4 -> 0.67   K=8 -> 0.93   K=10 -> 1.04
        # At K=8 the "residual" is nearly as large as the signal, so the
        # stable-ODE premise the anti-symmetric parametrisation relies on is
        # gone.  Changing K would then change the OPTIMISATION PROBLEM as well
        # as the receptive field, confounding the entire sweep.
        # Dividing by sqrt(K) flattens it to ~0.33 for every K.
        return x + self.e * self._scale * out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return (f"Euler_ChebConv({self.in_channels}, {self.out_channels}, "
                f"K={len(self.lins)}, step={self.e}, diss={self.g}, "
                f"term_norm={self.term_norm}, scale={self._scale:.4f})")


class EulerSAGEConv(MessagePassing):
    """Strictly 1-hop message passing baseline (GraphSAGE-mean flavour).

        out = x + eps * (W_self x + W_neigh * mean_{j in N(i)} x_j)

    Deliberately built with the SAME Euler residual + anti-symmetric
    parametrisation as Euler_ChebConv so the only thing that differs between
    this and the ChebNet is the RECEPTIVE FIELD (1 hop vs K-1 hops).
    Connects to the Hariri et al. 2020 GraphSAGE lineage.
    """
    def __init__(self, in_channels, out_channels,
                 step_size=CHEB_STEP_SIZE, dissipation_force=CHEB_DISSIPATION,
                 bias=True, term_norm="sqrt", **kwargs):
        kwargs.setdefault("aggr", "mean")
        super().__init__(**kwargs)
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.e = step_size
        self.g = dissipation_force

        self.lin_self  = PyGLinear(in_channels, out_channels,
                                   bias=False, weight_initializer="glorot")
        self.lin_neigh = PyGLinear(in_channels, out_channels,
                                   bias=False, weight_initializer="glorot")
        for lin in (self.lin_self, self.lin_neigh):
            register_parametrization(lin, "weight",
                                     AntiSymmetric(dissipative_force=self.g))

        self.bias = Parameter(torch.Tensor(out_channels)) if bias else None
        # 2 terms (self + neighbour), scaled the same way as ChebConv so the
        # 1-hop baseline sits at the same residual magnitude as every K.
        self._scale = Euler_ChebConv._term_scale(2, term_norm)
        self.term_norm = term_norm
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        agg = self.propagate(edge_index, x=x)
        out = self.lin_self(x) + self.lin_neigh(agg)
        if self.bias is not None:
            out = out + self.bias
        return x + self.e * self._scale * out

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return (f"EulerSAGEConv({self.in_channels}, {self.out_channels}, "
                f"K_eff=1, step={self.e}, diss={self.g}, "
                f"term_norm={self.term_norm}, scale={self._scale:.4f})")


# ══════════════════════════════════════════════════════════════════════════
# POOLING
# ══════════════════════════════════════════════════════════════════════════
def _scatter_stats(h, batch, num_graphs):
    """Fixed symmetric statistics pooling: [sum | mean | max | std].

    NO learned interaction between nodes.  Each node's embedding enters the
    reduction independently, so this cannot invent inter-particle
    correlations that the conv layers did not already produce.
    (Contrast with Set2Set, whose LSTM+softmax attention re-reads the whole
     node set several times and IS a global mixing mechanism.)
    """
    D = h.size(1)

    def _sum(v):
        if pyg_scatter is not None:
            return pyg_scatter(v, batch, dim=0, dim_size=num_graphs,
                               reduce="sum")
        out = torch.zeros(num_graphs, v.size(1), device=v.device,
                          dtype=v.dtype)
        return out.index_add(0, batch, v)

    ssum = _sum(h)
    cnt  = _sum(torch.ones(h.size(0), 1, device=h.device, dtype=h.dtype))
    cnt  = cnt.clamp(min=1.0)
    smean = ssum / cnt

    if pyg_scatter is not None:
        smax = pyg_scatter(h, batch, dim=0, dim_size=num_graphs, reduce="max")
    else:
        smax = torch.zeros_like(ssum).scatter_reduce(
            0, batch.view(-1, 1).expand(-1, D), h, "amax", include_self=False)
    smax = torch.nan_to_num(smax, nan=0.0, posinf=0.0, neginf=0.0)

    var  = (_sum(h * h) / cnt - smean ** 2).clamp(min=0.0)
    sstd = torch.sqrt(var + 1e-8)

    return torch.cat([ssum, smean, smax, sstd], dim=1)   # [B, 4D]


class StatsPool(nn.Module):
    """[sum|mean|max|std] -> LayerNorm -> Linear -> LATENT_DIM."""
    def __init__(self, hidden, out_dim=LATENT_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(4 * hidden)
        self.proj = nn.Sequential(
            nn.Linear(4 * hidden, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, h, batch, num_graphs):
        return self.proj(self.norm(_scatter_stats(h, batch, num_graphs)))


# ══════════════════════════════════════════════════════════════════════════
# ENCODER
# ══════════════════════════════════════════════════════════════════════════
class JetEncoder(nn.Module):
    """kNN graph -> Linear(3->256) -> [conv]xL -> pos fuse -> pool -> 512."""
    def __init__(self, in_dim=3, hidden=GNN_HIDDEN_DIM, K=5, layers=2,
                 conv="cheb", pool="stats", term_norm="sqrt",
                 step_size=CHEB_STEP_SIZE, dissipation=CHEB_DISSIPATION,
                 pos_dim=POS_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        assert conv in ("cheb", "sage")
        assert pool in ("stats", "set2set")
        self.conv_type = conv
        self.pool_type = pool

        # Project 3 -> hidden BEFORE the convs so every conv weight matrix is
        # square (required by the AntiSymmetric parametrisation).
        self.input_proj = nn.Linear(in_dim, hidden, bias=False)

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(layers):
            if conv == "cheb":
                self.convs.append(Euler_ChebConv(hidden, hidden, K,
                                                 step_size, dissipation,
                                                 term_norm=term_norm))
            else:
                self.convs.append(EulerSAGEConv(hidden, hidden,
                                                step_size, dissipation,
                                                term_norm=term_norm))
            if i < layers - 1:
                self.bns.append(nn.BatchNorm1d(hidden))
        self.out_norm = nn.LayerNorm(hidden)   # NOT F.normalize: sum-pooling
                                               # needs magnitude information.

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, pos_dim), nn.GELU(), nn.Linear(pos_dim, pos_dim))
        self.fuse_proj = nn.Linear(hidden + pos_dim, hidden)

        if pool == "set2set":
            # Set2Set structurally emits 2*hidden.  Project it down so the
            # attention arm can run at ANY latent -- without this the
            # attention-vs-reach comparison is only possible at latent 512,
            # where nothing is compressed and no encoder can show an edge.
            self.pool = Set2Set(hidden, processing_steps=SET2SET_STEPS)
            self.pool_proj = (nn.Identity() if 2 * hidden == latent_dim
                              else nn.Sequential(
                                  nn.LayerNorm(2 * hidden),
                                  nn.Linear(2 * hidden, latent_dim), nn.GELU(),
                                  nn.Linear(latent_dim, latent_dim)))
        else:
            self.pool = StatsPool(hidden, out_dim=latent_dim)

    def forward(self, x, edge_index, batch):
        coords = x[:, :2]
        h = self.input_proj(x)
        for i, cv in enumerate(self.convs):
            h = cv(h, edge_index)
            if i < len(self.bns):
                h = F.leaky_relu(self.bns[i](h))
        h = self.out_norm(h)

        h = self.fuse_proj(torch.cat([h, self.pos_mlp(coords)], dim=-1))

        if self.pool_type == "set2set":
            return self.pool_proj(self.pool(h, batch))
        n_graphs = int(batch.max().item()) + 1
        return self.pool(h, batch, n_graphs)


# ══════════════════════════════════════════════════════════════════════════
# DECODER
# ══════════════════════════════════════════════════════════════════════════
class FiLMResBlock(nn.Module):
    """Per-slot residual block, FiLM-conditioned on z.

    Conditioning is a BROADCAST of the same z to every slot -> it carries no
    slot-to-slot information, so this adds decoder capacity without adding a
    global mixing path.  This is how the no-attention decoder stays
    competitive without contaminating the ablation.
    """
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, dim)
        self.fc2   = nn.Linear(dim, dim)
        self.scale = nn.Linear(cond_dim, dim)
        self.shift = nn.Linear(cond_dim, dim)

    def forward(self, x, cond):
        s = 1.0 + self.scale(cond).unsqueeze(1)
        b = self.shift(cond).unsqueeze(1)
        h = self.norm(x) * s + b
        h = self.fc2(F.gelu(self.fc1(F.gelu(h))))
        return x + h


class SlotDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, n_particles=150,
                 slot_dim=SLOT_DIM, hidden=DECODER_HIDDEN, particle_dim=4,
                 use_attn=False, n_res_blocks=3,
                 eta_bound=ETA_BOUND, phi_bound=PHI_BOUND):
        super().__init__()
        self.n_particles  = n_particles
        self.slot_dim     = slot_dim
        self.particle_dim = particle_dim
        self.use_attn     = use_attn
        self.eta_bound    = eta_bound
        self.phi_bound    = phi_bound

        self.z_proj      = nn.Linear(latent_dim, slot_dim)
        self.slot_id_emb = nn.Embedding(n_particles, slot_dim)
        self.register_buffer("slot_ids", torch.arange(n_particles))

        d_model = slot_dim * 2
        if use_attn:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
                dropout=0.1, activation="gelu", batch_first=True)
            self.self_attn = nn.TransformerEncoder(layer, num_layers=2)
            self.shared_net = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden), nn.GELU(),
                nn.Linear(hidden, particle_dim),
            )
        else:
            self.stem   = nn.Linear(d_model, hidden)
            self.blocks = nn.ModuleList(
                [FiLMResBlock(hidden, latent_dim) for _ in range(n_res_blocks)])
            self.head   = nn.Sequential(
                nn.LayerNorm(hidden), nn.GELU(),
                nn.Linear(hidden, particle_dim))

    def forward(self, z):
        B = z.size(0)
        z_slot = self.z_proj(z)
        z_bc   = z_slot.unsqueeze(1).expand(B, self.n_particles, self.slot_dim)
        ids    = self.slot_id_emb(self.slot_ids).unsqueeze(0).expand(B, -1, -1)
        slots  = torch.cat([z_bc, ids], dim=-1)          # [B, N, slot_dim*2]

        if self.use_attn:
            slots = self.self_attn(slots)
            flat  = slots.reshape(B * self.n_particles, self.slot_dim * 2)
            out   = self.shared_net(flat).view(B, self.n_particles,
                                               self.particle_dim)
        else:
            h = self.stem(slots)
            for blk in self.blocks:
                h = blk(h, z)                            # per-slot only
            out = self.head(h)

        return self._apply_activations(out)

    def _apply_activations(self, output):
        eta = torch.tanh(output[:, :, 0]) * self.eta_bound
        phi = torch.tanh(output[:, :, 1]) * self.phi_bound
        mask_soft = torch.sigmoid(output[:, :, 3] * 2.0)

        # STE hard mask + active-only pT normalisation (v15.2 fix).
        mask_hard = (mask_soft > 0.5).float()
        mask_ste  = mask_hard - mask_soft.detach() + mask_soft

        pt = F.softplus(output[:, :, 2]) * mask_ste
        pt = pt / pt.sum(dim=1, keepdim=True).clamp(min=1e-4)

        return torch.stack([eta, phi, pt, mask_soft], dim=2)


# ══════════════════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════════════════
def sinkhorn_3d_loss(pred, target, pt_scale=PT_SCALE):
    pred_pts = torch.cat([pred[:, :, 0:1], pred[:, :, 1:2],
                          pred[:, :, 2:3] * pt_scale], dim=-1).contiguous()
    tgt_pts  = torch.cat([target[:, :, 0:1], target[:, :, 1:2],
                          target[:, :, 2:3] * pt_scale], dim=-1).contiguous()
    pw = pred[:, :, 3].clamp(min=1e-7)
    tw = (target[:, :, 3] > 0.5).float().clamp(min=1e-7)
    pw = pw / pw.sum(dim=1, keepdim=True)
    tw = tw / tw.sum(dim=1, keepdim=True)
    return sinkhorn_loss_fn(pw, pred_pts, tw, tgt_pts).mean()


def multiplicity_count_loss(pred, target, total_particles):
    pred_count = pred[:, :, 3].sum(dim=1)
    true_count = (target[:, :, 3] > 0.5).float().sum(dim=1)
    return F.mse_loss(pred_count, true_count) / total_particles


# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════
def load_jetnet_data(total_particles, max_jets_per_type=None,
                     split="train", log=print):
    """split: 'train' (70%) or 'test' (15%) -- JetNet's own official split."""
    log(f"Loading JetNet-{total_particles} split='{split}' "
        f"from {JETNET_DATA_DIR}")
    all_p, all_j, all_t = [], [], []
    for tid, jtype in enumerate(["g", "q", "t"]):
        p, j = JetNet.getData(
            jet_type=[jtype], data_dir=JETNET_DATA_DIR,
            particle_features=["etarel", "phirel", "ptrel", "mask"],
            jet_features=["pt", "eta", "mass", "num_particles"],
            num_particles=total_particles, split=split,
            split_fraction=JETNET_SPLIT_FRACTION, download=True,
        )
        if max_jets_per_type:
            # Deterministic head slice (NOT random) so the test set order is
            # identical across every run -> identical jets in every figure.
            p, j = p[:max_jets_per_type], j[:max_jets_per_type]
        all_p.append(p); all_j.append(j)
        all_t.append(np.full(len(p), tid, dtype=np.int64))
        log(f"  {jtype} [{split}]: {len(p)} jets")
    P = np.concatenate(all_p)
    log(f"  TOTAL [{split}]: {len(P)} jets")
    return P, np.concatenate(all_j), np.concatenate(all_t)


def load_graph_cache(cache_path, log=print):
    if not os.path.exists(cache_path):
        return None
    log(f"Loading graph cache: {cache_path}")
    with open(cache_path, "rb") as f:
        out = pickle.load(f)
    log(f"  {len(out[0])} graphs loaded.")
    return out


def graph_cache_path(split, total_particles, knn, max_per_type, symmetric,
                     root=None):
    """Cache file name encodes everything that changes the graphs, so
    directed (pre-v17) and undirected (v17) caches can never be mixed up."""
    tag = "sym" if symmetric else "dir"
    return os.path.join(root or CACHE_ROOT,
                        f"{split}_np{total_particles}_knn{knn}"
                        f"_mj{max_per_type or 'all'}_{tag}.pkl")


def build_graphs(particle_data, jet_types, knn_k, total_particles,
                 cache_path, log=print, symmetric=True):
    """kNN graph on the (eta, phi) plane, one graph per jet.

    symmetric=True (v17 default) makes the graph UNDIRECTED: j is connected
    to i whenever either is among the other's k nearest neighbours.

    WHY THIS MATTERS -- it was a real bug before v17.
    sklearn's kneighbors_graph is DIRECTED (i->j does not imply j->i); on jet
    graphs 57% of edges have no reverse edge.  ChebNet's Chebyshev
    polynomials are only bounded (|T_k| <= 1) when the rescaled Laplacian has
    a REAL spectrum in [-1, 1], which needs a symmetric graph.  On the
    directed graph the spectrum is complex and the terms GROW with order:
    ||T_k|| averaged over 200 graphs,
        k:        0     1     2     3     4     5     6     7
        directed  1.00  1.12  1.67  2.18  2.70  3.02  3.37  3.77  (max 8.4)
        undirected 1.00 everywhere
    (worst-case spectral norm on synthetic jet-like graphs).  For a typical
    signal, measured through this module's own _norm + propagate, the
    directed graph inflates T_k relative to the undirected one by ~1.4x at
    k=5 and ~1.65x at k=7.  Either way the directed terms are unbounded and
    the undirected ones are not: a penalty growing with K and unrelated to
    reach -- the ChebNet handicap the sqrt(K) fix alone could not remove.
    symmetric=False reproduces pre-v17 graphs, needed only to re-score
    models that were trained on them."""
    cached = load_graph_cache(cache_path, log)
    if cached is not None:
        return cached

    log(f"Building kNN(k={knn_k}) graphs for {len(particle_data)} jets ...")
    graphs, targets, types = [], [], []
    failed = 0
    n = len(particle_data)
    for i in range(n):
        try:
            jp   = particle_data[i]
            m    = jp[:, 3] == 1
            vp   = jp[m]
            if len(vp) < 2:
                failed += 1; continue
            vp   = vp[np.argsort(-vp[:, 2])]        # sort by pT descending
            hits = vp[:, :3]
            k = knn_k if len(hits) > knn_k else len(hits) - 1
            if k <= 0:
                failed += 1; continue
            adj = kneighbors_graph(hits[:, :2], n_neighbors=k,
                                   mode="connectivity", include_self=False)
            if symmetric:
                adj = adj.maximum(adj.T)
            ei = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
            graphs.append(Data(x=torch.tensor(hits, dtype=torch.float),
                               edge_index=ei))
            padded = np.zeros((total_particles, 4), dtype=np.float32)
            padded[:len(vp)] = vp
            padded[:len(vp), 3] = 1
            targets.append(torch.from_numpy(padded))
            types.append(jet_types[i])
            if i % 50000 == 0 and i > 0:
                print(f"  {i}/{n}", flush=True)
        except Exception:
            failed += 1
    log(f"  collected {len(graphs)}, failed {failed}")

    out = (graphs, targets, np.array(types, dtype=np.int64))
    with open(cache_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  cache saved -> {cache_path}")
    return out


class JetGraphDataset(Dataset):
    def __init__(self, graphs, targets):
        self.graphs, self.targets = graphs, targets

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i], self.targets[i]


def collate_fn(batch):
    return (Batch.from_data_list([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]))


# ══════════════════════════════════════════════════════════════════════════
# GRAPH GEOMETRY CHECK  (run this BEFORE committing compute)
# ══════════════════════════════════════════════════════════════════════════
def check_graph_diameter(graphs, n_sample=300, log=print):
    """Median graph diameter tells you how many hops are needed to cross a
    jet.  If diameter << (layers x (K-1)), the K-sweep is SATURATED and will
    show a flat line.  This table belongs in the paper as setup
    justification."""
    try:
        import networkx as nx
        from torch_geometric.utils import to_networkx
    except ImportError:
        log("  networkx not available — skipping diameter check.")
        return {}

    rng  = np.random.RandomState(0)
    idx  = rng.choice(len(graphs), min(n_sample, len(graphs)), replace=False)
    diam, apl, ncomp = [], [], []
    for i in idx:
        G = to_networkx(graphs[i], to_undirected=True)
        if G.number_of_nodes() < 2:
            continue
        comps = list(nx.connected_components(G))
        ncomp.append(len(comps))
        Gc = G.subgraph(max(comps, key=len))
        diam.append(nx.diameter(Gc))
        apl.append(nx.average_shortest_path_length(Gc))
    stats = {
        "median_diameter": float(np.median(diam)) if diam else None,
        "mean_diameter":   float(np.mean(diam))   if diam else None,
        "p90_diameter":    float(np.percentile(diam, 90)) if diam else None,
        "mean_avg_shortest_path": float(np.mean(apl)) if apl else None,
        "mean_n_components":      float(np.mean(ncomp)) if ncomp else None,
    }
    log(f"  graph geometry: {stats}")
    return stats


# ══════════════════════════════════════════════════════════════════════════
# JET SUBSTRUCTURE OBSERVABLES — live in suite.py (one source of truth)
# ══════════════════════════════════════════════════════════════════════════
import suite as S  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════
def count_params(*mods):
    return int(sum(p.numel() for m in mods for p in m.parameters()
                   if p.requires_grad))


def load_state(module, state_dict):
    """load_state_dict that copes with the anti-symmetric convolutions.

    PyG's Linear overrides _save_to_state_dict and writes the COMPUTED
    (parametrised) weight as '<name>.weight' in addition to the real
    parameter '<name>.parametrizations.weight.original'.  A strict load then
    rejects '<name>.weight' as unexpected.  That derived key carries no
    information -- it is recomputed from 'original' -- so drop it, then load
    STRICTLY so any genuine mismatch still fails loudly.

    Before v17 this made every resume-after-timeout fail and silently restart
    training from epoch 0 (no past run ever resumed, so no result was hit).
    """
    sd = dict(state_dict)
    for k in list(sd):
        if k.endswith(".weight"):
            base = k[: -len(".weight")]
            if f"{base}.parametrizations.weight.original" in sd:
                del sd[k]
    module.load_state_dict(sd, strict=True)


def train_autoencoder(enc, dec, graphs, targets, cfg, out_dir, log):
    ds     = JetGraphDataset(graphs, targets)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        collate_fn=collate_fn, num_workers=cfg["num_workers"],
                        drop_last=True, pin_memory=torch.cuda.is_available())
    params = list(enc.parameters()) + list(dec.parameters())
    opt    = optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-5)
    sch    = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg["epochs"], 1), eta_min=1e-6)

    ckpt_path = os.path.join(out_dir, "ckpt_latest.pth")
    start_epoch, history = 0, []
    if cfg["resume"] and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            load_state(enc, ck["enc"]); load_state(dec, ck["dec"])
            opt.load_state_dict(ck["opt"])
            start_epoch = ck["epoch"]; history = ck.get("history", [])
            for _ in range(start_epoch):
                sch.step()
            log(f"  resumed from epoch {start_epoch}")
        except Exception as e:
            log(f"  !!! RESUME FAILED ({e}) — TRAINING FROM SCRATCH. "
                f"Earlier epochs of this run are lost.")

    nb = len(loader)
    log(f"  train jets={len(ds)}  batches/epoch={nb}  "
        f"epochs {start_epoch}->{cfg['epochs']}")

    enc.train(); dec.train()
    for ep in range(start_epoch, cfg["epochs"]):
        tot = 0.0
        for bi, (bg, bt) in enumerate(loader):
            bg = bg.to(device, non_blocking=True)
            bt = bt.to(device, non_blocking=True)
            z    = enc(bg.x, bg.edge_index, bg.batch)
            pred = dec(z)

            l_sink = sinkhorn_3d_loss(pred, bt, pt_scale=cfg["pt_scale"])
            l_mask = F.binary_cross_entropy(
                pred[:, :, 3].clamp(1e-6, 1 - 1e-6),
                (bt[:, :, 3] > 0.5).float())
            l_cnt  = multiplicity_count_loss(pred, bt, cfg["total_particles"])
            loss   = l_sink + l_mask + COUNT_LOSS_WEIGHT * l_cnt

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item()

            if (bi + 1) % 200 == 0:
                log(f"    E{ep+1} B{bi+1}/{nb} tot={loss.item():.4f} "
                    f"sink={l_sink.item():.4f} mask={l_mask.item():.4f} "
                    f"cnt={l_cnt.item():.4f}")

        sch.step()
        avg = tot / max(nb, 1)
        history.append(avg)
        log(f"  epoch {ep+1:4d}/{cfg['epochs']}  loss={avg:.5f}  "
            f"lr={sch.get_last_lr()[0]:.2e}")

        if (ep + 1) % cfg["ckpt_every"] == 0 or (ep + 1) == cfg["epochs"]:
            torch.save({"epoch": ep + 1, "enc": enc.state_dict(),
                        "dec": dec.state_dict(), "opt": opt.state_dict(),
                        "history": history}, ckpt_path)

    torch.save(enc.state_dict(), os.path.join(out_dir, "encoder.pth"))
    torch.save(dec.state_dict(), os.path.join(out_dir, "decoder.pth"))

    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(alpha=.3)
    plt.title(f"AE loss — {cfg['tag']}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ae_loss.png"), dpi=110)
    plt.close()

    enc.eval(); dec.eval()
    return history


# ══════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def reconstruct(enc, dec, graphs, targets, batch_size=512):
    enc.eval(); dec.eval()
    recon, real = [], []
    for i in range(0, len(graphs), batch_size):
        bg = Batch.from_data_list(graphs[i:i + batch_size]).to(device)
        z  = enc(bg.x, bg.edge_index, bg.batch)
        recon.append(dec(z).cpu().numpy())
        real.append(torch.stack(targets[i:i + batch_size]).numpy())
    return np.concatenate(recon), np.concatenate(real)


# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════
def plot_recon_pairs(real, recon, out_path, n_show=6, jet_types=None,
                     tag=""):
    """THE side-by-side figure: real jet graph vs its encoder->decoder
    reconstruction, for several jets."""
    names = {0: "gluon", 1: "quark", 2: "top"}
    fig, axes = plt.subplots(n_show, 3, figsize=(15, 3.4 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    for r in range(n_show):
        rm = real[r][:, 3] > 0.5
        cm = recon[r][:, 3] > 0.5
        vmax = max(real[r][rm, 2].max() if rm.any() else 1e-3,
                   recon[r][cm, 2].max() if cm.any() else 1e-3)

        # Zoom to the jet (real jets live well inside +-2), but never clip
        # a reconstruction that wandered further out.
        lim = 0.5
        for arr, msk in ((real[r], rm), (recon[r], cm)):
            if msk.any():
                lim = max(lim, float(np.abs(arr[msk, :2]).max()) * 1.15)
        lim = min(lim, max(ETA_BOUND, PHI_BOUND))
        xlim = ylim = (-lim, lim)

        for c, (arr, msk, ttl) in enumerate([
                (real[r],  rm, "REAL"),
                (recon[r], cm, "RECONSTRUCTED"),
        ]):
            ax = axes[r, c]
            if msk.any():
                sc = ax.scatter(arr[msk, 0], arr[msk, 1], c=arr[msk, 2],
                                s=28 + 900 * arr[msk, 2], cmap="viridis",
                                vmin=0, vmax=vmax, alpha=.85,
                                edgecolors="k", linewidths=.3)
                plt.colorbar(sc, ax=ax, label="$p_T^{rel}$")
            jt = (f" [{names.get(int(jet_types[r]), '?')}]"
                  if jet_types is not None else "")
            ax.set(title=f"{ttl}{jt}  n={int(msk.sum())}",
                   xlabel=r"$\eta^{rel}$", ylabel=r"$\phi^{rel}$",
                   xlim=xlim, ylim=ylim)
            ax.grid(alpha=.25)

        ax = axes[r, 2]
        if rm.any():
            ax.scatter(real[r][rm, 0], real[r][rm, 1],
                       s=25 + 800 * real[r][rm, 2], facecolors="none",
                       edgecolors="tab:blue", linewidths=1.4, label="real")
        if cm.any():
            ax.scatter(recon[r][cm, 0], recon[r][cm, 1],
                       s=25 + 800 * recon[r][cm, 2], marker="x",
                       color="tab:red", linewidths=1.2, label="recon")
        ax.set(title="OVERLAY", xlabel=r"$\eta^{rel}$", ylabel=r"$\phi^{rel}$",
               xlim=xlim, ylim=ylim)
        ax.legend(fontsize=8); ax.grid(alpha=.25)

    fig.suptitle(f"Real jet vs encoder→decoder reconstruction — {tag}",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_particle_distributions(real, recon, out_path, tag=""):
    def collect(arr):
        eta, phi, pt, mult, pt1, pt5, pt20, mass, ptsum = ([] for _ in range(9))
        for jp in arr:
            m = jp[:, 3] > 0.5
            if m.sum() == 0:
                continue
            eta.extend(jp[m, 0]); phi.extend(jp[m, 1]); pt.extend(jp[m, 2])
            mult.append(int(m.sum()))
            s = np.sort(jp[m, 2])[::-1]
            if len(s) >= 1:  pt1.append(float(s[0]))
            if len(s) >= 5:  pt5.append(float(s[4]))
            if len(s) >= 20: pt20.append(float(s[19]))
            o = S.observables(jp)
            mass.append(o["mass"] if o else np.nan)
            ptsum.append(float(jp[m, 2].sum()))
        f = lambda x: np.asarray(x, dtype=float)
        return (f(eta), f(phi), f(pt), f(pt1), f(pt5), f(pt20),
                f(mult), f(mass), f(ptsum))

    rv, gv = collect(real), collect(recon)
    labels = [r"particle $\eta^{rel}$", r"particle $\phi^{rel}$",
              r"particle $p_T^{rel}$", r"1st $p_T^{rel}$", r"5th $p_T^{rel}$",
              r"20th $p_T^{rel}$", "multiplicity", "relative jet mass",
              r"jet $\sum p_T^{rel}$"]

    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    for ax, r, g, lab in zip(axes.flat, rv, gv, labels):
        r = r[np.isfinite(r)]; g = g[np.isfinite(g)]
        if len(r) == 0 or len(g) == 0:
            ax.set_title(f"{lab} — no data"); continue
        w1  = wasserstein_distance(r, g)
        sig = r.std() + 1e-12
        lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
        bins = np.linspace(lo, hi, 80)
        ax.hist(r, bins=bins, density=True, alpha=.5, label="real",  color="C0")
        ax.hist(g, bins=bins, density=True, alpha=.5, label="recon", color="C3")
        ax.set_yscale("log"); ax.set_xlabel(lab); ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.text(.02, .97, f"W1={w1:.4g}\nW1/σ={w1/sig:.4g}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="lightyellow", alpha=.7))
    fig.suptitle(f"Real vs reconstruction (held-out) — {tag}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_multiplicity(obs_r, obs_g, out_path, tag=""):
    """Is n actually RECONSTRUCTED, or is the decoder just guessing the mean?

    n is never handed to the decoder.  It has to come out of the 512-d latent
    via the per-slot mask head.  This figure is the check:
      left   n_recon vs n_real -- points should hug the y=x line, NOT form a
             horizontal band (a horizontal band = predicting the average)
      right  the error distribution, with MAE and bias
    """
    r = obs_r["mult"]; g = obs_g["mult"]
    ok = np.isfinite(r) & np.isfinite(g)
    r, g = r[ok], g[ok]
    if len(r) < 10:
        return
    err = g - r
    mae = float(np.mean(np.abs(err)))
    corr = float(np.corrcoef(r, g)[0, 1]) if r.std() > 0 and g.std() > 0 else 0.

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    lo, hi = 0, max(r.max(), g.max()) * 1.05
    hb = ax1.hexbin(r, g, gridsize=60, extent=(lo, hi, lo, hi),
                    bins="log", cmap="viridis", mincnt=1)
    plt.colorbar(hb, ax=ax1, label="jets (log)")
    ax1.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect (y = x)")
    ax1.axhline(r.mean(), color="orange", ls=":", lw=1.5,
                label=f"mean-guess baseline ({r.mean():.1f})")
    ax1.set(xlabel="true multiplicity  $n_{real}$",
            ylabel="reconstructed multiplicity  $n_{recon}$",
            xlim=(lo, hi), ylim=(lo, hi))
    ax1.set_title(f"Multiplicity reconstruction\nMAE={mae:.3f}   r={corr:.4f}")
    ax1.legend(fontsize=8)

    ax2.hist(err, bins=np.arange(err.min() - .5, err.max() + 1.5, 1.0),
             color="C0", alpha=.8)
    ax2.axvline(0, color="r", ls="--", lw=1.5)
    ax2.set(xlabel=r"$n_{recon} - n_{real}$", ylabel="jets")
    ax2.set_title(f"error   MAE={mae:.3f}   bias={err.mean():+.3f}   "
                  f"exact={100*np.mean(err == 0):.1f}%")
    ax2.grid(alpha=.3)

    fig.suptitle(f"Particle count is predicted from the latent, not copied "
                 f"— {tag}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_observables(obs_r, obs_g, out_path, tag=""):
    keys = S.PRIMARY + S.CONTROL + S.SUPPORT
    fig, axes = plt.subplots(2, 4, figsize=(21, 9))
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax, k in zip(axes.flat, keys):
        r = obs_r[k]; g = obs_g[k]
        ok = np.isfinite(r) & np.isfinite(g)
        r, g = r[ok], g[ok]
        if len(r) == 0:
            ax.set_title(f"{k} — no data"); continue
        sig = r.std() + 1e-12
        rel = float(np.mean(np.abs(g - r)) / sig)
        lo, hi = np.percentile(np.concatenate([r, g]), [0.2, 99.8])
        bins = np.linspace(lo, hi, 60)
        ax.hist(r, bins=bins, density=True, alpha=.5, label="real",  color="C0")
        ax.hist(g, bins=bins, density=True, alpha=.5, label="recon", color="C3")
        role = ("PRIMARY — needs pairs" if k in S.PRIMARY
                else ("SUPPORT — zero reach suffices" if k in S.SUPPORT
                      else ("CONTROL — no geometry" if k in S.CONTROL else "")))
        ax.set_title(f"{S.LABELS.get(k, k)}   [{role}]", fontsize=10)
        ax.set_xlabel(S.LABELS.get(k, k)); ax.legend(fontsize=8)
        ax.text(.02, .97, f"paired relerr = {rel:.4f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(fc="lightyellow", alpha=.8))
    fig.suptitle(f"Jet substructure observables — {tag}\n"
                 f"Top row: PRIMARY (needs pairs) and the ptD control.  "
                 f"Bottom row: SUPPORT, exactly computable with zero reach.",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=120)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════
def select_vis_indices(types, per_type=VIS_PER_TYPE):
    """First `per_type` jets of each type, in g/q/t order.  Deterministic, so
    every model in the ablation reconstructs the SAME jets."""
    idx = []
    for tid in (0, 1, 2):
        hit = np.flatnonzero(types == tid)[:per_type]
        idx.extend(hit.tolist())
    return np.array(idx, dtype=int)


def evaluate(enc, dec, graphs, targets, types, cfg, out_dir, log):
    """Score a trained model on the JetNet TEST split with the suite.
    Writes suite.json (what aggregate.py reads) and the diagnostic figures."""
    n_eval = len(graphs) if not cfg["n_eval"] else min(cfg["n_eval"],
                                                       len(graphs))
    g, t, ty = graphs[:n_eval], targets[:n_eval], types[:n_eval]
    log(f"  evaluating on {n_eval} JetNet TEST jets "
        f"(g={int((ty==0).sum())} q={int((ty==1).sum())} "
        f"t={int((ty==2).sum())}) ...")

    recon, real = reconstruct(enc, dec, g, t, batch_size=cfg["batch_size"])

    suite, (obs_r, obs_g, ok) = S.score(real, recon, cfg["latent_dim"], ty,
                                        device, cfg["batch_size"], log)

    # particle-level distributions, for the appendix
    def flat(arr, col):
        out = []
        for jp in arr:
            out.extend(jp[jp[:, 3] > 0.5, col])
        return np.asarray(out, dtype=float)
    for col, name in [(0, "eta"), (1, "phi"), (2, "pt")]:
        r, gg = flat(real, col), flat(recon, col)
        if len(r) and len(gg):
            w1 = wasserstein_distance(r, gg)
            suite[f"w1_{name}"] = float(w1)
            suite[f"w1sig_{name}"] = float(w1 / (r.std() + 1e-12))

    # --- figures: SAME six test jets for every model -----------------------
    vis = select_vis_indices(ty)
    if len(vis) == 0:
        vis = np.arange(min(6, len(real)))
    plot_recon_pairs(real[vis], recon[vis],
                     os.path.join(out_dir, "recon_pairs.png"),
                     n_show=len(vis), jet_types=ty[vis], tag=cfg["tag"])
    np.savez_compressed(os.path.join(out_dir, "recon_sample.npz"),
                        real=real[vis], recon=recon[vis],
                        types=ty[vis], vis_idx=vis)
    plot_particle_distributions(real, recon,
                                os.path.join(out_dir, "dist_comparison.png"),
                                tag=cfg["tag"])
    plot_observables(obs_r, obs_g, os.path.join(out_dir, "observables.png"),
                     tag=cfg["tag"])
    plot_multiplicity(obs_r, obs_g, os.path.join(out_dir, "multiplicity.png"),
                      tag=cfg["tag"])
    return suite


# ══════════════════════════════════════════════════════════════════════════
# THE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
DEFAULT_CFG = dict(
    tag="run",
    conv="cheb",             # 'cheb' | 'sage'
    K=6,                     # Chebyshev order (ignored for sage).
                             # reach = layers x (K-1):  K=2 -> 2 hops,
                             # K=6 -> 10 hops (= the p90 jet diameter)
    layers=2,
    knn=6,                   # kNN graph degree, constant across the study
    graph_symmetric=True,    # undirected kNN graph.  REQUIRED for ChebNet --
                             # see build_graphs().  False only to re-score
                             # models trained before v17.
    total_particles=150,
    pool="stats",            # 'stats' (fixed summary) | 'set2set' (attention)
    use_attn=False,          # decoder self-attention: off throughout the study
    latent_dim=64,           # THE MEASURING INSTRUMENT.  A typical jet is
                             # ~56 particles x 3 = ~169 numbers.  At 512 the
                             # latent is bigger than the jet and nothing has to
                             # be dropped; at 64 the encoder must choose.
    term_norm="sqrt",        # scale each conv's K-term sum by 1/sqrt(K) so
                             # the Euler step is the same size at every K.
                             # "none" = pre-fix behaviour (step grows with K).
    pt_scale=10.0,           # pT axis weight in the Sinkhorn TRAINING loss
    epochs=200,
    batch_size=512,
    lr=1e-3,
    num_workers=0,           # >0 copies the whole graph list per worker (RAM)
    n_eval=0,                # 0 = score the ENTIRE JetNet test split
    max_jets_per_type=None,  # None = all of JetNet's train split
    max_test_per_type=None,  # None = all of JetNet's test split
    ckpt_every=10,
    resume=True,
    seed=0,
    check_graph=True,
)


def run_one(user_cfg):
    cfg = dict(DEFAULT_CFG)
    cfg.update(user_cfg)

    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    out_dir = os.path.join(SAVE_ROOT, cfg["tag"])
    os.makedirs(out_dir, exist_ok=True)
    log = Logger(os.path.join(out_dir, "log.txt"))

    log("=" * 74)
    log(f"RUN: {cfg['tag']}")
    log("=" * 74)
    for k in ["conv", "K", "layers", "knn", "total_particles", "pool",
              "use_attn", "latent_dim", "term_norm", "pt_scale", "seed",
              "epochs", "batch_size", "lr",
              "max_jets_per_type", "max_test_per_type", "n_eval"]:
        log(f"  {k:18s} = {cfg[k]}")
    log(f"  device             = {device}")

    # ── data: JetNet's OWN train / test splits (no re-slicing of train) ──
    def get_split(split, max_per_type):
        cache_path = graph_cache_path(split, cfg["total_particles"],
                                      cfg["knn"], max_per_type,
                                      cfg["graph_symmetric"])
        cached = load_graph_cache(cache_path, log)
        if cached is not None:
            return cached
        pdata, _, ptypes = load_jetnet_data(
            cfg["total_particles"], max_per_type, split=split, log=log)
        out = build_graphs(pdata, ptypes, cfg["knn"],
                           cfg["total_particles"], cache_path, log=log,
                           symmetric=cfg["graph_symmetric"])
        del pdata
        return out

    log("\n  [TRAIN SPLIT]")
    tr_g, tr_t, tr_y = get_split("train", cfg["max_jets_per_type"])
    log("\n  [TEST SPLIT — never seen during training]")
    te_g, te_t, te_y = get_split("test", cfg["max_test_per_type"])

    log(f"\n  data: train={len(tr_g)}  test={len(te_g)}")
    for tid, nm in enumerate(["g", "q", "t"]):
        log(f"    {nm}: train={int((tr_y == tid).sum())}  "
            f"test={int((te_y == tid).sum())}")

    if cfg["check_graph"]:
        gstats = check_graph_diameter(tr_g, log=log)
    else:
        gstats = {}

    # ── model ────────────────────────────────────────────────────────────
    lat = cfg["latent_dim"]
    enc = JetEncoder(K=cfg["K"], layers=cfg["layers"], conv=cfg["conv"],
                     pool=cfg["pool"], term_norm=cfg["term_norm"],
                     latent_dim=lat).to(device)
    dec = SlotDecoder(latent_dim=lat, n_particles=cfg["total_particles"],
                      use_attn=cfg["use_attn"]).to(device)

    n_enc, n_dec = count_params(enc), count_params(dec)
    log(f"  params: encoder={n_enc:,}  decoder={n_dec:,}  "
        f"total={n_enc + n_dec:,}")
    log(f"  encoder conv: {enc.convs[0]}")
    hops = (cfg["layers"] * (cfg["K"] - 1) if cfg["conv"] == "cheb"
            else cfg["layers"])
    log(f"  total receptive field = {hops} hops "
        f"(graph median diameter = {gstats.get('median_diameter')})")

    # ── train ────────────────────────────────────────────────────────────
    history = train_autoencoder(enc, dec, tr_g, tr_t, cfg, out_dir, log)

    # ── evaluate ─────────────────────────────────────────────────────────
    metrics = evaluate(enc, dec, te_g, te_t, te_y, cfg, out_dir, log)

    metrics.update({
        "graph_symmetric": cfg["graph_symmetric"],
        "tag": cfg["tag"], "conv": cfg["conv"], "K": cfg["K"],
        "layers": cfg["layers"], "knn": cfg["knn"],
        "total_particles": cfg["total_particles"],
        "pool": cfg["pool"], "use_attn": cfg["use_attn"],
        "pt_scale": cfg["pt_scale"], "term_norm": cfg["term_norm"],
        "latent_dim": lat,
        "seed": cfg["seed"],
        "epochs": cfg["epochs"],
        "config": "global" if (cfg["pool"] == "set2set" or cfg["use_attn"])
                  else "local",
        "receptive_field_hops": hops,
        "params_encoder": n_enc, "params_decoder": n_dec,
        "params_total": n_enc + n_dec,
        "final_train_loss": float(history[-1]) if history else None,
        "graph_stats": gstats,
        "n_train": len(tr_g), "n_test": len(te_g),
        "n_train_per_type": {n: int((tr_y == i).sum())
                             for i, n in enumerate(["g", "q", "t"])},
        "n_test_per_type": {n: int((te_y == i).sum())
                            for i, n in enumerate(["g", "q", "t"])},
        "eval_split": "jetnet_test",
        "batch_size": cfg["batch_size"], "lr": cfg["lr"],
    })

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # suite.json is what aggregate.py reads -- same format reeval.py writes
    with open(os.path.join(out_dir, "suite.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    log("\n  --- KEY RESULTS (paired relative error, lower = better) ---")
    for k in S.PRIMARY + ["primary_mean"] + S.CONTROL + S.SUPPORT + \
             ["support_mean", "emd", "mult_mae", "mult_exact_frac"]:
        v = metrics.get(k)
        if v is not None:
            log(f"    {k:22s} = {v:.6f}")
    log(f"\n  saved -> {out_dir}")
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# CLI (single run)
# ══════════════════════════════════════════════════════════════════════════
def _cli():
    p = argparse.ArgumentParser(description="Single ablation run")
    for k, v in DEFAULT_CFG.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_','-')}",
                           type=lambda s: s.lower() in ("1", "true", "yes"),
                           default=v)
        elif v is None:
            p.add_argument(f"--{k.replace('_','-')}", type=int, default=None)
        else:
            p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    a = vars(p.parse_args())
    run_one(a)


if __name__ == "__main__":
    _cli()
