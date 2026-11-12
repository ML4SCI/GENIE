"""
layerdata.py -- the two-stage representation used by the leading CaloChallenge
submissions, adapted here.

A shower is split into two pieces that are modelled separately:

  u  (45 numbers)  the ENERGY content: how much energy there is and how it is
                   distributed along z, as ratios that are bounded in [0, 1]
                   and analytically invertible, so energy conservation is
                   exact by construction rather than learned.
  w  (6480)        the SHAPE: each voxel's share of its own layer's energy, so
                   every layer sums to 1 and the layer totals carry no
                   information here at all.

Both go through a logit transform with alpha = 1e-6, which is the piece our
own measurements were missing. In the old 'fraction' representation a faint
voxel sits a median 5.5e-4 standardized units above empty, and the model's
5th-percentile lit voxel came out at 3.4 MeV against a real 0.63 -- it simply
had no resolution down there. logit(w) sends w -> 0 to about -13.8, giving the
network room to represent faint voxels, and because the layer totals live in u
the per-voxel means can no longer bias the total energy (the Jensen effect
that made the log-of-absolute representation undershoot by 2.4x).

    showers --encode--> (u, w) --two independent models--> (u, w) --decode--> showers

Reference: CaloDREAM (arXiv:2405.09629) for the u-ratios and the ViT shape
model; CaloDiffusion (arXiv:2308.03876) for the logit preprocessing.
"""

import torch

ALPHA = 1e-6
EPS = 1e-12


def _logit(p):
    p = ALPHA + (1.0 - 2.0 * ALPHA) * p.clamp(0.0, 1.0)
    return torch.log(p) - torch.log1p(-p)


def _sigmoid_inv(y):
    p = torch.sigmoid(y)
    return ((p - ALPHA) / (1.0 - 2.0 * ALPHA)).clamp(0.0, 1.0)


def encode(showers_mev, einc_mev, f_scale=None, stats=None):
    """
    (N, 45, 16, 9) MeV + (N, 1) MeV  ->  (u_std, w_std, stats)

    u[:, 0]    = E_tot / (f * E_inc)                       in [0, 1]
    u[:, i]    = E_{i-1} / sum_{j >= i-1} E_j   for i = 1..44   in [0, 1]
                 (the last layer is implied, so 45 numbers describe 45 layers)
    w          = E_voxel / E_layer, i.e. each layer sums to 1
    """
    sh = showers_mev.reshape(showers_mev.shape[0], 45, -1).double()
    e = einc_mev.reshape(-1, 1).double()
    layer_e = sh.sum(-1)                                   # (N, 45)
    tot = layer_e.sum(-1, keepdim=True)                    # (N, 1)

    if f_scale is None:
        f_scale = (stats or {}).get("f_scale")
    if f_scale is None:                                    # keep u0 inside (0, 1)
        f_scale = float((tot / e).max().item()) * 1.05
    u = [tot / (f_scale * e)]

    remaining = tot.clone()
    for i in range(44):                                    # last layer is implied
        u.append(layer_e[:, i:i + 1] / remaining.clamp_min(EPS))
        remaining = remaining - layer_e[:, i:i + 1]
    u = torch.cat(u, dim=1)                                # (N, 45)

    w = sh / layer_e.unsqueeze(-1).clamp_min(EPS)          # (N, 45, 144)
    u_l, w_l = _logit(u), _logit(w.reshape(sh.shape[0], -1))

    if stats is None:
        stats = {"f_scale": f_scale,
                 "u_mean": u_l.mean(0).float(), "u_std": u_l.std(0).clamp_min(1e-6).float(),
                 "w_mean": float(w_l.mean()), "w_std": float(w_l.std()),
                 "Nz": 45, "Nphi": 16, "Nr": 9, "representation": "layer"}
    u_std = (u_l.float() - stats["u_mean"]) / stats["u_std"]
    w_std = (w_l.float() - stats["w_mean"]) / stats["w_std"]
    return u_std, w_std, stats


def decode(u_std, w_std, einc_mev, stats):
    """Exact inverse of encode(): (u, w) -> showers in MeV."""
    u = _sigmoid_inv(u_std.double() * stats["u_std"].double() + stats["u_mean"].double())
    w = _sigmoid_inv(w_std.double() * stats["w_std"] + stats["w_mean"])
    e = einc_mev.reshape(-1, 1).double()

    tot = u[:, :1] * stats["f_scale"] * e
    layer_e, remaining = [], tot.clone()
    for i in range(44):
        ei = u[:, i + 1:i + 2] * remaining
        layer_e.append(ei)
        remaining = (remaining - ei).clamp_min(0.0)
    layer_e.append(remaining)                              # the implied last layer
    layer_e = torch.cat(layer_e, dim=1)                    # (N, 45)

    w = w.reshape(-1, 45, 144)
    w = w / w.sum(-1, keepdim=True).clamp_min(EPS)         # renormalize each layer
    return (w * layer_e.unsqueeze(-1)).reshape(-1, 45, 16, 9).float()


def empty_value(stats):
    """Standardized value a voxel with zero share of its layer takes."""
    return (float(_logit(torch.zeros(1))) - stats["w_mean"]) / stats["w_std"]
