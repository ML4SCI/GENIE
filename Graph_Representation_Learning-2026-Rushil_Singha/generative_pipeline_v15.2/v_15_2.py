"""
Model 15 v2 — From Scratch
==============================================================
All improvements from the v15.1 fine-tune applied to a clean
from-scratch run.  Changes vs v15 base:

  [1] ETA_BOUND / PHI_BOUND: 0.8  →  2.0
  [2] pT norm: softmax-all-slots  →  softplus + STE hard mask + active-only norm
  [3] Multiplicity count loss: COUNT_LOSS_WEIGHT * MSE(active particle count)
  [4] No weight-loading / fine-tune branch — pure from-scratch training

BEFORE RUNNING: edit the two paths in the CONFIG block below.
"""

import os
import warnings
import math
import pickle
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════
# CONFIG — edit these two paths
# ══════════════════════════════════════════════════════════════════════════
JETNET_DATA_DIR = "/pscratch/sd/r/rushil13/jetnet_data"
SAVE_DIR        = "/pscratch/sd/r/rushil13/gsoc_rushil/v_15.2-sinkhorn_set2set_v2/results_v_15.2"
# ══════════════════════════════════════════════════════════════════════════

CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
WEIGHTS_DIR    = os.path.join(SAVE_DIR, "weights")

os.makedirs(JETNET_DATA_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,        exist_ok=True)
os.makedirs(CHECKPOINT_DIR,  exist_ok=True)
os.makedirs(WEIGHTS_DIR,     exist_ok=True)

LOG_FILE = os.path.join(SAVE_DIR, "log.txt")
with open(LOG_FILE, "w") as f:
    f.write("=== Model15 v2: from scratch | eta/phi ±2, STE-pT, count-loss ===\n\n")

def log_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

import numpy as np
import matplotlib.pyplot as plt
from jetnet.datasets import JetNet
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn.aggr import Set2Set
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear as PyGLinear
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import get_laplacian
from torch.nn import Module, Parameter
from torch.nn.utils.parametrize import register_parametrization
from geomloss import SamplesLoss
from scipy.stats import wasserstein_distance

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_print(f"Device: {device}")
if torch.cuda.is_available():
    log_print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE         = 512
BATCH_SIZE_FLOW    = 2048
GNN_HIDDEN_DIM     = 256
GNN_K              = 10
CHEB_STEP_SIZE     = 0.45
CHEB_DISSIPATION   = 0.1
KNN_K              = 10
TOTAL_PARTICLES    = 150
SET2SET_STEPS      = 4
POS_DIM            = 64
LATENT_DIM         = 2 * GNN_HIDDEN_DIM    # 512
SLOT_DIM           = 64
DECODER_HIDDEN     = 256
PT_SCALE           = 10.0
DIFFUSION_HIDDEN   = 2048
EPOCHS_AE          = 300
EPOCHS_DIFF        = 1000
N_EVAL_DIST        = 25000
NUM_JET_TYPES      = 3
JET_TYPE_EMBED_DIM = 8
COND_DIM_TOTAL     = 3 + JET_TYPE_EMBED_DIM  # 11
CHECKPOINT_EVERY   = 10

# [CHANGE 1] Expanded eta/phi decoder output bounds.
# v15 base hardcoded 0.8, which silently clips real data tails that reach
# ±1.2–1.5.  ±2.0 covers all real particles with headroom; the tanh is kept
# (not removed) so decoder outputs remain bounded for Sinkhorn stability.
ETA_BOUND = 2.0
PHI_BOUND = 2.0

# [CHANGE 3] Weight on auxiliary multiplicity count loss (MSE on active count).
# BCE treats all 150 slots independently — no global multiplicity signal.
# 0.5 gives this term meaningful signal without overwhelming Sinkhorn + BCE.
COUNT_LOSS_WEIGHT = 0.5

GRAPH_CACHE = os.path.join(WEIGHTS_DIR, "graph_cache.pkl")
ENC_PATH    = os.path.join(WEIGHTS_DIR, "gnn_encoder.pth")
DEC_PATH    = os.path.join(WEIGHTS_DIR, "slot_decoder.pth")
FLOW_PATH   = os.path.join(WEIGHTS_DIR, "cond_flow.pth")

sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=0.05, debias=True)


# ══════════════════════════════════════════════════════════════════════════════
# STABLE CHEBNET — Euler_ChebConv with Anti-Symmetric Weight Parametrization
# (unchanged from v15)
# ══════════════════════════════════════════════════════════════════════════════

class AntiSymmetric(Module):
    """Parametrize W = W_upper - W_upper^T - g*I → eigenvalues in left half-plane."""
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
    """Stable ChebConv: anti-symmetric weights + Euler residual step.

    out = x + ε · Σ_{k=0}^{K-1} W_k · T_k(L̃) · x
    """
    def __init__(self, in_channels, out_channels, K,
                 step_size=0.45, dissipation_force=0.1, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        assert K > 0
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.normalization = 'sym'
        self.e = step_size
        self.g = dissipation_force

        self.lins = nn.ModuleList()
        for _ in range(K):
            lin = PyGLinear(in_channels, out_channels,
                            bias=False, weight_initializer='glorot')
            register_parametrization(lin, 'weight',
                                     AntiSymmetric(dissipative_force=self.g))
            self.lins.append(lin)

        self.bias = Parameter(torch.Tensor(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins[1:]:
            lin.reset_parameters()
        zeros(self.bias)

    def _norm(self, edge_index, num_nodes, edge_weight, dtype, batch=None):
        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, self.normalization, dtype, num_nodes)
        lambda_max = 2.0 * edge_weight.max()
        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)
        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1
        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        edge_index, norm = self._norm(
            edge_index, x.size(self.node_dim), edge_weight, x.dtype, batch)

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

        return x + self.e * out   # Euler residual — key stability guarantee

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'step={self.e}, dissipation={self.g})')


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS (unchanged from v15)
# ══════════════════════════════════════════════════════════════════════════════

def _delete_old_numbered_ckpts(prefix):
    for fname in os.listdir(CHECKPOINT_DIR):
        if (fname.startswith(f"{prefix}_epoch_") and "latest" not in fname
                and fname.endswith(".pth")):
            try:
                os.remove(os.path.join(CHECKPOINT_DIR, fname))
            except OSError:
                pass

def save_checkpoint(data_dict, prefix, epoch):
    numbered = os.path.join(CHECKPOINT_DIR, f"{prefix}_epoch_{epoch}.pth")
    latest   = os.path.join(CHECKPOINT_DIR, f"{prefix}_epoch_latest.pth")
    torch.save(data_dict, numbered)
    torch.save(data_dict, latest)
    _delete_old_numbered_ckpts(prefix)
    log_print(f"  [{prefix}] Checkpoint saved at epoch {epoch}")

def load_latest_checkpoint(prefix):
    latest = os.path.join(CHECKPOINT_DIR, f"{prefix}_epoch_latest.pth")
    if not os.path.exists(latest):
        return None
    ckpt = torch.load(latest, map_location=device)
    log_print(f"  [{prefix}] Resumed from epoch {ckpt['epoch']}")
    return ckpt

def cleanup_checkpoints(prefix):
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname.startswith(f"{prefix}_epoch_") and fname.endswith(".pth"):
            try:
                os.remove(os.path.join(CHECKPOINT_DIR, fname))
            except OSError:
                pass
    log_print(f"  [{prefix}] All checkpoints deleted.")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (unchanged from v15)
# ══════════════════════════════════════════════════════════════════════════════

def load_jetnet_data():
    log_print(f"Loading JetNet from {JETNET_DATA_DIR} (auto-downloads if missing)")
    all_p, all_j, all_t = [], [], []
    for tid, jtype in enumerate(['g', 'q', 't']):
        p, j = JetNet.getData(
            jet_type=[jtype], data_dir=JETNET_DATA_DIR,
            particle_features=['etarel', 'phirel', 'ptrel', 'mask'],
            jet_features=['pt', 'eta', 'mass', 'num_particles'],
            num_particles=TOTAL_PARTICLES, split='train', download=True,
        )
        all_p.append(p); all_j.append(j)
        all_t.append(np.full(len(p), tid, dtype=np.int64))
        log_print(f"  {jtype}: {len(p)} jets")
    return (np.concatenate(all_p), np.concatenate(all_j),
            np.concatenate(all_t))


def collect_graph_and_targets(particle_data, jet_data, jet_types,
                               cache_path=None):
    """Build kNN graphs — or load from cache if already done."""
    if cache_path is None:
        cache_path = GRAPH_CACHE

    if os.path.exists(cache_path):
        log_print(f"Loading graph cache from {cache_path} ...")
        with open(cache_path, "rb") as f:
            out = pickle.load(f)
        log_print(f"  Loaded {len(out[0])} graphs from cache.")
        return out

    log_print(f"Building graphs for {len(particle_data)} jets (will cache to {cache_path})...")
    graph_data, target_particles, original_jets, collected_types = [], [], [], []
    failed = 0
    n = len(particle_data)
    for i in range(n):
        try:
            jp   = particle_data[i]
            mask = jp[:, 3] == 1
            vp   = jp[mask]
            if len(vp) == 0:
                failed += 1; continue
            vp   = vp[np.argsort(-vp[:, 2])]
            hits = vp[:, :3]
            k = KNN_K if len(hits) > KNN_K else len(hits) - 1
            if k <= 0:
                failed += 1; continue
            adj = kneighbors_graph(hits[:, :2], n_neighbors=k,
                                   mode='connectivity', include_self=False)
            edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
            graph_data.append(Data(x=torch.tensor(hits, dtype=torch.float),
                                   edge_index=edge_index))
            padded = np.zeros((TOTAL_PARTICLES, 4))
            padded[:len(vp)] = vp
            padded[:len(vp), 3] = 1
            target_particles.append(torch.tensor(padded, dtype=torch.float))
            original_jets.append((jp, jet_data[i]))
            collected_types.append(jet_types[i])
            if i % 50000 == 0 and i > 0:
                print(f"  {i}/{n} processed")
        except Exception:
            failed += 1
    log_print(f"Collected {len(graph_data)}, failed {failed}")

    out = (graph_data, target_particles, original_jets,
           np.array(collected_types, dtype=np.int64))

    log_print(f"Saving graph cache to {cache_path} ...")
    with open(cache_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    log_print("  Cache saved.")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: SPATIAL SET2SET ENCODER (unchanged from v15)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialSet2SetChebNet(nn.Module):
    """GNN encoder: Euler_ChebConv × 4 + spatial pos embedding + Set2Set pooling."""
    def __init__(self, in_dim=3, hidden=GNN_HIDDEN_DIM, K=GNN_K,
                 step_size=CHEB_STEP_SIZE, dissipation=CHEB_DISSIPATION,
                 pos_dim=POS_DIM, processing_steps=SET2SET_STEPS):
        super().__init__()
        # Input projection done before conv so all Euler_ChebConv layers
        # are square (hidden × hidden) — required by AntiSymmetric parametrization.
        self.input_proj = nn.Linear(in_dim, hidden, bias=False)
        self.conv1 = Euler_ChebConv(hidden, hidden, K, step_size, dissipation)
        self.conv2 = Euler_ChebConv(hidden, hidden, K, step_size, dissipation)
        self.conv3 = Euler_ChebConv(hidden, hidden, K, step_size, dissipation)
        self.conv4 = Euler_ChebConv(hidden, hidden, K, step_size, dissipation)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.bn3   = nn.BatchNorm1d(hidden)
        self.pos_mlp   = nn.Sequential(
            nn.Linear(2, pos_dim), nn.GELU(), nn.Linear(pos_dim, pos_dim)
        )
        self.fuse_proj = nn.Linear(hidden + pos_dim, hidden)
        self.pool      = Set2Set(hidden, processing_steps=processing_steps)

    def forward(self, x, edge_index, batch=None):
        coords = x[:, :2]
        h = self.input_proj(x)
        h = F.leaky_relu(self.bn1(self.conv1(h, edge_index)))
        h = F.leaky_relu(self.bn2(self.conv2(h, edge_index)))
        h = F.leaky_relu(self.bn3(self.conv3(h, edge_index)))
        h = F.normalize(self.conv4(h, edge_index), p=2, dim=1)
        pos_emb = self.pos_mlp(coords)
        h_fused = self.fuse_proj(torch.cat([h, pos_emb], dim=-1))
        return self.pool(h_fused, batch)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: SLOT DECODER
# CHANGED: _apply_activations — see [CHANGE 1] and [CHANGE 2] below
# ══════════════════════════════════════════════════════════════════════════════

class SlotDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, n_particles=TOTAL_PARTICLES,
                 slot_dim=SLOT_DIM, hidden=DECODER_HIDDEN, particle_dim=4):
        super().__init__()
        self.n_particles  = n_particles
        self.slot_dim     = slot_dim
        self.particle_dim = particle_dim
        self.z_proj       = nn.Linear(latent_dim, slot_dim)
        self.slot_id_emb  = nn.Embedding(n_particles, slot_dim)
        self.register_buffer('slot_ids', torch.arange(n_particles))

        d_model    = slot_dim * 2
        attn_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.self_attn = nn.TransformerEncoder(attn_layer, num_layers=2)

        self.shared_net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, particle_dim),
        )

    def forward(self, z):
        B = z.size(0)
        z_slot      = self.z_proj(z)
        z_broadcast = z_slot.unsqueeze(1).expand(B, self.n_particles, self.slot_dim)
        ids         = self.slot_id_emb(self.slot_ids).unsqueeze(0).expand(B, -1, -1)
        slots       = torch.cat([z_broadcast, ids], dim=-1)      # [B, 150, slot_dim*2]
        slots       = self.self_attn(slots)
        flat        = slots.view(B * self.n_particles, self.slot_dim * 2)
        out         = self.shared_net(flat).view(B, self.n_particles, self.particle_dim)
        return self._apply_activations(out)

    def _apply_activations(self, output):
        # [CHANGE 1] Expanded eta/phi bounds.
        # v15 base: tanh * 0.8 — silently clipped real-data tails (~±1.5).
        # v15.2:    tanh * 2.0 — covers all real particles with headroom.
        # tanh is KEPT (not removed): unbounded outputs break Sinkhorn stability
        # and produce unphysical jets (e.g. η = 50 from a noise latent).
        eta = torch.tanh(output[:, :, 0]) * ETA_BOUND
        phi = torch.tanh(output[:, :, 1]) * PHI_BOUND

        mask_soft = torch.sigmoid(output[:, :, 3] * 2.0)

        # [CHANGE 2] Straight-Through Estimator (STE) mask + active-only pT.
        #
        # v15 base problem — softmax over all 150 slots:
        #   (a) Forces every slot to compete for pT mass, suppressing leading pT.
        #   (b) Normalises over inactive slots too → pT distribution depends
        #       on total slot count, not actual particle count.
        #
        # Fix:
        #   1. STE: forward uses hard 0/1 mask (inactive → exactly 0 pT);
        #      backward gradient flows through the soft sigmoid as if it
        #      were the hard mask.  This is the standard way to make a
        #      thresholding operation differentiable.
        #   2. softplus(output) — always positive, no slot-to-slot competition.
        #   3. Gate by STE mask → inactive slots contribute zero pT.
        #   4. Normalise ONLY over active slots; clamp denom to avoid ÷0
        #      in the edge case where all slots are inactive (shouldn't happen
        #      in practice, but is handled safely).
        mask_hard = (mask_soft > 0.5).float()
        mask_ste  = mask_hard - mask_soft.detach() + mask_soft   # STE trick

        pt_raw = F.softplus(output[:, :, 2])
        pt     = pt_raw * mask_ste
        pt     = pt / pt.sum(dim=1, keepdim=True).clamp(min=1e-4)

        # Return mask_soft (not mask_ste) so BCE and count losses get smooth
        # gradients through the sigmoid.
        return torch.stack([eta, phi, pt, mask_soft], dim=2)


# ══════════════════════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════════════════════

def sinkhorn_3d_loss(pred, target, pt_scale=PT_SCALE):
    pred_pts = torch.cat([pred[:, :, 0:1], pred[:, :, 1:2],
                           pred[:, :, 2:3] * pt_scale], dim=-1).contiguous()
    tgt_pts  = torch.cat([target[:, :, 0:1], target[:, :, 1:2],
                           target[:, :, 2:3] * pt_scale], dim=-1).contiguous()
    pred_w = pred[:, :, 3].clamp(min=1e-7)
    tgt_w  = (target[:, :, 3] > 0.5).float().clamp(min=1e-7)
    pred_w = pred_w / pred_w.sum(dim=1, keepdim=True)
    tgt_w  = tgt_w  / tgt_w.sum(dim=1, keepdim=True)
    return sinkhorn_loss_fn(pred_w, pred_pts, tgt_w, tgt_pts).mean()


# [CHANGE 3] Multiplicity count loss.
# BCE treats each of the 150 slots as an independent Bernoulli variable;
# it has no signal about the *total* number of active particles per jet.
# This MSE term directly penalises the global active-count mismatch.
# Dividing by TOTAL_PARTICLES normalises to O(1) so it doesn't dominate.
def multiplicity_count_loss(pred, target):
    pred_count = pred[:, :, 3].sum(dim=1)                          # soft sum
    true_count = (target[:, :, 3] > 0.5).float().sum(dim=1)       # hard count
    return F.mse_loss(pred_count, true_count) / TOTAL_PARTICLES


# ══════════════════════════════════════════════════════════════════════════════
# DATASET & COLLATE (unchanged from v15)
# ══════════════════════════════════════════════════════════════════════════════

class JetGraphDataset(Dataset):
    def __init__(self, graphs, targets):
        self.graphs  = graphs
        self.targets = targets
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]

def collate_fn(batch):
    return (Batch.from_data_list([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]))


# ══════════════════════════════════════════════════════════════════════════════
# AUTOENCODER TRAINING
# [CHANGE 4] No weight-loading / fine-tune branch.  Only checkpoint resume.
# LR stays at 1e-3 (same as v15 base) — the v15.1 explosion was caused by
# weight incompatibility after changing _apply_activations, not by LR.
# ══════════════════════════════════════════════════════════════════════════════

def train_autoencoder(gnn, decoder, graph_data, target_particles,
                      epochs=EPOCHS_AE, batch_size=BATCH_SIZE):
    start_epoch  = 0
    loss_history = []

    ckpt = load_latest_checkpoint("ae")
    if ckpt is not None:
        gnn.load_state_dict(ckpt['enc'],     strict=False)
        decoder.load_state_dict(ckpt['dec'], strict=False)
        start_epoch  = ckpt['epoch']
        loss_history = ckpt.get('loss_history', [])
        log_print(f"Resuming AE from epoch {start_epoch}.")
    else:
        log_print("No AE checkpoint found — training from scratch.")

    dataset   = JetGraphDataset(graph_data, target_particles)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=0)
    all_params = list(gnn.parameters()) + list(decoder.parameters())
    optimizer  = optim.AdamW(all_params, lr=1e-3, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - start_epoch, 1), eta_min=1e-6
    )
    if ckpt is not None and 'opt' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['opt'])
        except Exception:
            log_print("  [ae] Could not restore optimizer state — using fresh optimizer.")

    n_batches = len(loader)
    log_print(f"\n=== AE Training (Model 15 v2 — from scratch) ===")
    log_print(f"    Encoder  : SpatialSet2SetChebNet → [B, {LATENT_DIM}]")
    log_print(f"    Decoder  : SlotDecoder (eta/phi ±{ETA_BOUND}, STE-pT) → [B, 150, 4]")
    log_print(f"    Loss     : Sinkhorn(p=1) + BCE(mask) + {COUNT_LOSS_WEIGHT}×MSE(count)")
    log_print(f"    Epochs   : {start_epoch}→{epochs} | {n_batches} batches/epoch | LR=1e-3")

    gnn.train(); decoder.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        for bidx, (bg, bt) in enumerate(loader):
            bg = bg.to(device); bt = bt.to(device)
            z    = gnn(bg.x, bg.edge_index, bg.batch)
            pred = decoder(z)

            loss_sink  = sinkhorn_3d_loss(pred, bt)
            loss_mult  = F.binary_cross_entropy(
                pred[:, :, 3].clamp(1e-6, 1 - 1e-6),
                (bt[:, :, 3] > 0.5).float()
            )
            loss_count = multiplicity_count_loss(pred, bt)
            total      = loss_sink + loss_mult + COUNT_LOSS_WEIGHT * loss_count

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            epoch_loss += total.item()

            if (bidx + 1) % 100 == 0:
                log_print(f"  E{epoch+1}/{epochs} B{bidx+1}/{n_batches} | "
                          f"total={total.item():.4f}  sink={loss_sink.item():.4f}  "
                          f"mult={loss_mult.item():.4f}  count={loss_count.item():.4f}")

        scheduler.step()
        avg = epoch_loss / n_batches
        loss_history.append(avg)
        log_print(f"AE Epoch {epoch+1:4d}/{epochs} | avg={avg:.4f} | "
                  f"LR={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint({
                'epoch': epoch + 1, 'enc': gnn.state_dict(),
                'dec': decoder.state_dict(), 'opt': optimizer.state_dict(),
                'loss_history': loss_history,
            }, prefix="ae", epoch=epoch + 1)

    torch.save(gnn.state_dict(),     ENC_PATH)
    torch.save(decoder.state_dict(), DEC_PATH)
    log_print(f"AE complete.\n  {ENC_PATH}\n  {DEC_PATH}")
    cleanup_checkpoints("ae")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('AE Loss — Model15 v2 (from scratch, STE-pT, ±2 bounds)')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ae_loss.png"))
    plt.close()

    gnn.eval(); decoder.eval()
    return gnn, decoder


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: CONDITIONAL FLOW MATCHING (unchanged from v15)
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if t.dim() > 1:
            t = t.squeeze(-1)
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device)
                         * -(math.log(10000) / (half - 1)))
        emb  = t[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, hidden, cond_dim, dropout=0.1):
        super().__init__()
        self.norm1      = nn.LayerNorm(hidden)
        self.lin1       = nn.Linear(hidden, hidden)
        self.act        = nn.LeakyReLU(0.2)
        self.drop       = nn.Dropout(dropout)
        self.norm2      = nn.LayerNorm(hidden)
        self.lin2       = nn.Linear(hidden, hidden)
        self.t_proj     = nn.Linear(hidden, hidden)
        self.film_scale = nn.Linear(cond_dim, hidden)
        self.film_shift = nn.Linear(cond_dim, hidden)

    def forward(self, x, t_emb, cond):
        scale = 1.0 + self.film_scale(cond)
        shift = self.film_shift(cond)
        h     = self.norm1(x) * scale + shift + self.t_proj(t_emb)
        h     = self.drop(self.lin1(self.act(h)))
        h     = self.lin2(self.act(self.norm2(h)))
        return x + h


class ConditionalFlowModel(nn.Module):
    def __init__(self, emb_dim=LATENT_DIM, cond_dim=COND_DIM_TOTAL,
                 hidden=DIFFUSION_HIDDEN, time_dim=512, n_layers=8):
        super().__init__()
        self.type_emb   = nn.Embedding(NUM_JET_TYPES, JET_TYPE_EMBED_DIM)
        self.time_mlp   = nn.Sequential(
            SinusoidalEmbeddings(time_dim),
            nn.Linear(time_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden,   hidden),
        )
        self.input_proj = nn.Linear(emb_dim, hidden)
        self.blocks     = nn.ModuleList([
            FiLMBlock(hidden, cond_dim) for _ in range(n_layers)
        ])
        self.out_proj   = nn.Linear(hidden, emb_dim)

    def forward(self, x, t, cont_cond, type_ids):
        cond  = torch.cat([cont_cond, self.type_emb(type_ids)], dim=-1)
        t_emb = self.time_mlp(t)
        h     = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h, t_emb, cond)
        return self.out_proj(h)


def train_flow_matching(embeddings, jet_features, jet_type_ids,
                        epochs=EPOCHS_DIFF, batch_size=BATCH_SIZE_FLOW):
    emb_dim = embeddings.shape[1]
    model   = ConditionalFlowModel(emb_dim=emb_dim).to(device)

    start_epoch  = 0
    loss_history = []

    ckpt = load_latest_checkpoint("flow")
    if ckpt is not None:
        model.load_state_dict(ckpt['model'], strict=False)
        start_epoch  = ckpt['epoch']
        loss_history = ckpt.get('loss_history', [])
        log_print(f"Resuming flow from epoch {start_epoch}.")
    else:
        log_print("No flow checkpoint — training from scratch.")

    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs - start_epoch, 1), eta_min=1e-6
    )
    if ckpt is not None and 'opt' in ckpt:
        try:
            opt.load_state_dict(ckpt['opt'])
        except Exception:
            log_print("  [flow] Could not restore optimizer state — using fresh optimizer.")

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(embeddings,   dtype=torch.float32),
        torch.tensor(jet_features, dtype=torch.float32),
        torch.tensor(jet_type_ids, dtype=torch.long),
    )
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = len(loader)
    log_print(f"\n=== Flow Matching | emb_dim={emb_dim} | "
              f"epochs {start_epoch}→{epochs} | {n_batches} batches ===")

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        for emb_b, cond_b, type_b in loader:
            emb_b  = emb_b.to(device); cond_b = cond_b.to(device); type_b = type_b.to(device)
            t      = torch.rand(emb_b.size(0), 1, device=device)
            noise  = torch.randn_like(emb_b)
            x_t    = (1 - t) * noise + t * emb_b
            pred_v = model(x_t, t, cond_b, type_b)
            loss   = F.mse_loss(pred_v, emb_b - noise)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        sch.step()
        avg = epoch_loss / n_batches
        loss_history.append(avg)
        if (epoch + 1) % 10 == 0:
            log_print(f"Flow Epoch {epoch+1:4d}/{epochs} | loss={avg:.6f} | "
                      f"LR={sch.get_last_lr()[0]:.2e}")
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint({'epoch': epoch+1, 'model': model.state_dict(),
                             'opt': opt.state_dict(), 'loss_history': loss_history},
                            prefix="flow", epoch=epoch+1)

    torch.save(model.state_dict(), FLOW_PATH)
    cleanup_checkpoints("flow")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title('Flow Matching Loss — Model15 v2')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "flow_loss.png"))
    plt.close()

    model.eval()
    return model


def sample_flow(model, cont_conds, type_ids, n_steps=750, batch_size=BATCH_SIZE_FLOW):
    model.eval()
    samples = []
    with torch.no_grad():
        for i in range(0, len(cont_conds), batch_size):
            bs   = min(batch_size, len(cont_conds) - i)
            cond = torch.tensor(cont_conds[i:i+bs], dtype=torch.float32).to(device)
            tids = torch.tensor(type_ids[i:i+bs],   dtype=torch.long).to(device)
            x    = torch.randn(bs, LATENT_DIM, device=device)
            dt   = 1.0 / n_steps
            for s in range(n_steps):
                t = torch.full((bs, 1), s / n_steps, device=device)
                x = x + dt * model(x, t, cond, tids)
            samples.append(x.cpu().numpy())
    return np.concatenate(samples, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION & METRICS
# [CHANGE 1] Axis limits now tied to ETA/PHI_BOUND; histogram margin added.
# ══════════════════════════════════════════════════════════════════════════════

def visualize_jets(particle_list, titles, save_name="generated"):
    n, nc = len(particle_list), 2
    nr    = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(14 * nc, 6 * nr))
    if nr == 1:
        axes = np.array(axes).reshape(1, -1)
    # Axis limits tied to ETA/PHI_BOUND so no generated jet is clipped in the plot.
    xlim = (-ETA_BOUND, ETA_BOUND)
    ylim = (-PHI_BOUND, PHI_BOUND)
    for i, (p, t) in enumerate(zip(particle_list, titles)):
        ax   = axes[i // nc, i % nc]
        mask = p[:, 3] > 0.5
        if mask.sum():
            sc = ax.scatter(p[mask, 0], p[mask, 1], c=p[mask, 2],
                            s=50, cmap='viridis', alpha=0.7)
            plt.colorbar(sc, ax=ax, label='pT')
        ax.set(title=f"{t} (n={mask.sum()})", xlabel='η', ylabel='φ',
               xlim=xlim, ylim=ylim)
        ax.grid(alpha=0.3)
    for i in range(n, nr * nc):
        axes[i // nc, i % nc].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.png"))
    plt.close()


def plot_distributions(real_particles, gen_particles, save_name):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    MARGIN_FRAC = 0.08  # breathing room around histogram tails

    def collect(particles):
        eta, phi, pt = [], [], []
        mult, pt1, pt5, pt20, mass, ptsum = [], [], [], [], [], []
        for jp in particles:
            m = jp[:, 3] > 0.5
            if m.sum() == 0: continue
            eta.extend(jp[m, 0].tolist())
            phi.extend(jp[m, 1].tolist())
            pt.extend(jp[m, 2].tolist())
            mult.append(int(m.sum()))
            pts = np.sort(jp[m, 2])[::-1]
            if len(pts) >= 1:  pt1.append(float(pts[0]))
            if len(pts) >= 5:  pt5.append(float(pts[4]))
            if len(pts) >= 20: pt20.append(float(pts[19]))
            eta_c = np.clip(jp[m, 0], -5, 5)
            px = (jp[m, 2] * np.cos(jp[m, 1])).sum()
            py = (jp[m, 2] * np.sin(jp[m, 1])).sum()
            pz = (jp[m, 2] * np.sinh(eta_c)).sum()
            E  = (jp[m, 2] * np.cosh(eta_c)).sum()
            m2 = max(E**2 - px**2 - py**2 - pz**2, 0.0)
            mass.append(float(np.sqrt(m2)))
            ptsum.append(float(jp[m, 2].sum()))
        return (np.array(eta), np.array(phi), np.array(pt),
                np.array(pt1), np.array(pt5), np.array(pt20),
                np.array(mult), np.array(mass), np.array(ptsum))

    rv = collect(real_particles)
    gv = collect(gen_particles)
    labels = ['Particle eta_rel', 'Particle phi_rel', 'Particle pT_rel',
              '1st pT_rel', '5th pT_rel', '20th pT_rel',
              'Multiplicity', 'Relative Jet Mass', 'Jet pT sum (rel)']

    for ax, r, g, label in zip(axes.flat, rv, gv, labels):
        if len(r) == 0 or len(g) == 0:
            ax.set_title(f"{label} — no data"); continue
        dist = wasserstein_distance(r, g)
        lo   = min(r.min(), g.min()); hi = max(r.max(), g.max())
        span = hi - lo
        margin = span * MARGIN_FRAC if span > 0 else 0.05
        bins = np.linspace(lo, hi, 80)
        ax.hist(r, bins=bins, density=True, alpha=0.5, label='Real',      color='C0')
        ax.hist(g, bins=bins, density=True, alpha=0.5, label='Generated', color='C1')
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_yscale('log')
        ax.set_xlabel(label); ax.set_ylabel('Density')
        ax.legend(fontsize=7)
        color = 'green' if dist < 0.05 else ('orange' if dist < 0.15 else 'red')
        ax.text(0.02, 0.97, f'W1: {dist:.4f}', transform=ax.transAxes,
                va='top', fontsize=9, bbox=dict(fc=color, alpha=0.4))

    fig.suptitle('Real vs Generated — Model15 v2 (from scratch, STE-pT, ±2 bounds)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{save_name}.png"), dpi=120)
    plt.close()
    log_print(f"Distribution plot saved: {save_name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main_pipeline():
    log_print("\n" + "=" * 70)
    log_print("MODEL 15 v2 — SpatialSet2Set + SlotDecoder (STE-pT, ±2) + Flow(512)")
    log_print("=" * 70 + "\n")

    particle_data, jet_data, jet_types_all = load_jetnet_data()

    graph_data, target_particles, original_jets, collected_types = \
        collect_graph_and_targets(particle_data, jet_data, jet_types_all)
    N = len(graph_data)

    gnn     = SpatialSet2SetChebNet().to(device)
    decoder = SlotDecoder(latent_dim=LATENT_DIM).to(device)

    gnn, decoder = train_autoencoder(
        gnn, decoder, graph_data, target_particles,
        epochs=EPOCHS_AE, batch_size=BATCH_SIZE
    )

    # ── Extract latents for flow training ────────────────────────────────────
    log_print("Extracting encoder latents for flow training...")
    lat_cache = os.path.join(WEIGHTS_DIR, "encoder_latents.pt")
    if os.path.exists(lat_cache) and os.path.exists(ENC_PATH):
        if os.path.getmtime(ENC_PATH) > os.path.getmtime(lat_cache):
            log_print("  Encoder weights newer than cache — regenerating.")
            os.remove(lat_cache)

    if os.path.exists(lat_cache):
        z_all = torch.load(lat_cache, map_location='cpu').numpy()
    else:
        gnn.eval()
        parts = []
        with torch.no_grad():
            for i in range(0, N, BATCH_SIZE_FLOW):
                b = Batch.from_data_list(graph_data[i:i+BATCH_SIZE_FLOW]).to(device)
                z = gnn(b.x, b.edge_index, b.batch)
                parts.append(z.cpu())
        z_all = torch.cat(parts, 0).numpy()
        torch.save(torch.from_numpy(z_all), lat_cache)
    log_print(f"Latents shape: {z_all.shape}")

    z_mean = z_all.mean(0); z_std = z_all.std(0) + 1e-6
    z_norm = (z_all - z_mean) / z_std
    np.save(os.path.join(WEIGHTS_DIR, "z_mean.npy"), z_mean)
    np.save(os.path.join(WEIGHTS_DIR, "z_std.npy"),  z_std)
    log_print("Latents normalised.")

    jet_pt   = jet_data[:N, 0]
    jet_mass = jet_data[:N, 2]
    jet_mult = jet_data[:N, 3] / TOTAL_PARTICLES

    def minmax(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    jet_features = np.stack([minmax(jet_pt), minmax(jet_mass), jet_mult], axis=1)
    jet_type_ids = collected_types[:N]
    log_print(f"Jet types: g={(jet_type_ids==0).sum()}  "
              f"q={(jet_type_ids==1).sum()}  t={(jet_type_ids==2).sum()}")

    flow_model = train_flow_matching(z_norm, jet_features, jet_type_ids,
                                     epochs=EPOCHS_DIFF)

    # ── Scatter visualisation (8 sample jets) ────────────────────────────────
    log_print("Sampling 8 jets for scatter visualisation...")
    idx_vis      = np.random.choice(N, 8, replace=False)
    z_vis        = sample_flow(flow_model, jet_features[idx_vis],
                               jet_type_ids[idx_vis], n_steps=750)
    z_vis_denorm = z_vis * z_std + z_mean
    with torch.no_grad():
        gen_vis = decoder(
            torch.tensor(z_vis_denorm, dtype=torch.float32).to(device)
        ).cpu().numpy()
    type_names = {0: 'g-jet', 1: 'q-jet', 2: 't-jet'}
    titles = [f"{type_names[t]} #{i+1}" for i, t in enumerate(jet_type_ids[idx_vis])]
    visualize_jets(gen_vis, titles, save_name="generated_jets_v15_v2")

    # ── Distribution comparison ───────────────────────────────────────────────
    log_print(f"\nGenerating {N_EVAL_DIST} jets for distribution comparison...")
    n_eval   = min(N_EVAL_DIST, N)
    idx_eval = np.random.choice(N, n_eval, replace=False)
    z_gen        = sample_flow(flow_model, jet_features[idx_eval],
                               jet_type_ids[idx_eval], n_steps=750)
    z_gen_denorm = z_gen * z_std + z_mean

    gen_parts_list = []
    decoder.eval()
    with torch.no_grad():
        for i in range(0, len(z_gen_denorm), BATCH_SIZE_FLOW):
            zb = torch.tensor(
                z_gen_denorm[i:i+BATCH_SIZE_FLOW], dtype=torch.float32
            ).to(device)
            gen_parts_list.append(decoder(zb).cpu().numpy())
    gen_parts = np.concatenate(gen_parts_list, axis=0)

    real_parts_list = []
    for i in idx_eval:
        jp     = particle_data[i]
        m      = jp[:, 3] > 0.5
        vp     = jp[m]
        padded = np.zeros((TOTAL_PARTICLES, 4))
        padded[:len(vp)] = vp; padded[:len(vp), 3] = 1
        real_parts_list.append(padded)
    real_parts = np.array(real_parts_list)

    plot_distributions(real_parts, gen_parts, "dist_comparison_v15_v2")

    # ── Reconstruction W1 metrics ─────────────────────────────────────────────
    log_print("\n--- Reconstruction W1 metrics (10 jets) ---")
    n_rec = min(10, N)
    with torch.no_grad():
        eg    = Batch.from_data_list(graph_data[:n_rec]).to(device)
        ez    = gnn(eg.x, eg.edge_index, eg.batch)
        recon = decoder(ez).cpu().numpy()

    ws_eta, ws_phi, ws_pt = [], [], []
    for i in range(n_rec):
        orig = particle_data[i][particle_data[i][:, 3] == 1]
        rec  = recon[i][recon[i][:, 3] > 0.5]
        if len(orig) and len(rec):
            ws_eta.append(wasserstein_distance(orig[:, 0], rec[:, 0]))
            ws_phi.append(wasserstein_distance(orig[:, 1], rec[:, 1]))
            ws_pt.append( wasserstein_distance(orig[:, 2], rec[:, 2]))
    if ws_eta:
        log_print(f"  Reconstruction W1:  η={np.mean(ws_eta):.4f}  "
                  f"φ={np.mean(ws_phi):.4f}  pT={np.mean(ws_pt):.4f}")

    log_print("\nModel 15 v2 complete.")
    return "Done"


if __name__ == "__main__":
    main_pipeline()