"""The velocity field network v_theta(x, t) or, conditionally, v_theta(x, t, c).

A small MLP that predicts the flow velocity at a point ``x`` and time ``t``,
optionally conditioned on a context vector ``c`` (e.g. normalized log incident
energy for calorimeter showers). Two of the planned architecture upgrades live
here: **Fourier feature** embeddings (so the network can represent the fast
time-dependence of the velocity near t=0/1, and sharp condition-dependence) and
**GELU** activations.
"""

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Random Fourier features for a (B, k) input -> (B, dim).

    Frequencies are drawn once and fixed (registered as a buffer, not trained),
    so the embedding is deterministic for a given seed and moves with ``.to()``.
    ``scale`` sets the frequency spread: large for the fast time-dependence of
    the velocity, smaller for smooth condition-dependence.
    """

    def __init__(self, in_dim=1, dim=64, scale=10.0, seed=0):
        super().__init__()
        assert dim % 2 == 0, "Fourier embedding dim must be even"
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(in_dim, dim // 2, generator=g) * scale
        self.register_buffer("w", w)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(-1, 1)
        proj = 2.0 * math.pi * (x @ self.w)      # (B, dim/2)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class FourierTimeEmbedding(FourierFeatures):
    """Fourier features for the scalar time t -> (B, dim). Kept as its own name
    because the time embedding is used everywhere; it is ``FourierFeatures``
    with ``in_dim=1``."""

    def __init__(self, dim=64, scale=10.0, seed=0):
        super().__init__(in_dim=1, dim=dim, scale=scale, seed=seed)


class VelocityField(nn.Module):
    """MLP velocity field. ``forward(x, t)`` returns the velocity at (x, t).

    With ``cond_dim > 0`` the field is conditional: ``forward(x, t, cond)``
    where ``cond`` is (B, cond_dim). The condition enters as its raw value plus
    a low-frequency Fourier embedding (``cond_emb`` features per input), which
    lets the network resolve nonlinear condition-dependence without hurting
    smooth interpolation between conditions.
    """

    def __init__(self, dim, hidden=128, depth=3, time_dim=64, cond_dim=0,
                 cond_emb=16, cond_scale=2.0, seed=0):
        super().__init__()
        self.cond_dim = cond_dim
        self.time_emb = FourierTimeEmbedding(time_dim, seed=seed)
        in_dim = dim + time_dim
        if cond_dim > 0:
            self.cond_emb = FourierFeatures(cond_dim, cond_emb, scale=cond_scale,
                                            seed=seed + 1)
            in_dim += cond_dim + cond_emb
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.GELU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, cond=None):
        parts = [x, self.time_emb(t)]
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("this VelocityField is conditional; pass cond")
            cond = cond.reshape(x.shape[0], self.cond_dim)
            parts += [cond, self.cond_emb(cond)]
        elif cond is not None:
            raise ValueError("cond given but model was built with cond_dim=0")
        return self.net(torch.cat(parts, dim=-1))
