"""Physics-informed NetGe autoencoder: reconstruction plus jet-observable prediction heads.

The backbone is NetGeMaxBilinear on relative pT (log pT fractions). Each jet is pooled from
[64-d hidden, 2-d latent] node features with mean and max (132-d); the two jets of an event are
combined as [sum, |difference|] (264-d). Three heads read this event code:
    auxiliary_head: 56 standardized Kitchen Sink subjettiness observables (masses, tau_N^beta)
    energy_head:    16 standardized low-order energy-flow polynomials
    ratio_head:     6 standardized log tau21, tau32, tau43 (beta=1) of both jets
The ablation model omits the ratio head.
Module and buffer names match the trained checkpoints, so their state dicts load directly.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter

from .netge import NetGeMaxBilinear, _JetEncoder, log_fractions


HEAD_HIDDEN = 64
EVENT_WIDTH = 264
TARGET_WEIGHT = .1


def _head(outputs):
    return nn.Sequential(nn.Linear(EVENT_WIDTH, HEAD_HIDDEN), nn.GELU(), nn.Linear(HEAD_HIDDEN, outputs))


class PooledJetEncoder(_JetEncoder):
    """Max-message jet encoder returning node latents and a [mean, max] pooled jet code."""

    def __init__(self, latent_dim=2):
        super().__init__(latent_dim)
        self.proj_head = nn.Identity()
        self.aggr = 'max'

    def forward(self, x, edge_index, edge_attr, mask):
        root, neighbor = edge_index
        nodes = self.embedding(x.flatten(0, 1))
        messages = self.message1(torch.cat([nodes[root], nodes[neighbor]-nodes[root], edge_attr], -1))
        hidden = F.gelu(self.self1(nodes) + scatter(messages, root, dim=0, dim_size=len(nodes), reduce='max'))
        messages = self.message2(torch.cat([hidden[root], hidden[neighbor]-hidden[root], edge_attr], -1))
        z = self.self2(hidden) + scatter(messages, root, dim=0, dim_size=len(nodes), reduce='max')
        z = z.reshape(*mask.shape, -1) * mask[..., None]
        features = torch.cat([hidden.reshape(*mask.shape, -1), z], -1)
        mean = (features * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        maximum = features.masked_fill(~mask[..., None], -torch.inf).max(1).values
        return z, torch.cat([mean, maximum], -1)


class PhysicsBackbone(NetGeMaxBilinear):
    """Relative-pT NetGeMaxBilinear with the pooled encoder; heads are attached by PhysicsAE."""

    def __init__(self):
        super().__init__()
        encoder = PooledJetEncoder()
        encoder.load_state_dict(self.shared_encoder.state_dict())
        self.shared_encoder = encoder
        self.register_buffer('bilinear_indices', torch.triu_indices(2, 2))

    def _decode_edges(self, left, right):
        row, column = self.bilinear_indices
        products = left[:, row] * right[:, column]
        products = products + (row != column) * (left[:, column] * right[:, row])
        endpoints = torch.cat([torch.minimum(left, right), torch.maximum(left, right)], -1)
        return self.edge_direct(endpoints) + F.linear(products, self.edge_bilinear)

    def forward(self, x, adjacency, mask, edges):
        return super().forward(log_fractions(x, mask), adjacency, mask, edges)


class PhysicsAE(nn.Module):
    """Inputs per jet (both jets of each event stacked as [first jets; second jets]):
    x (2B,N,1) log pT, adjacency (2B,N,N), mask (2B,N), edges (2B,N,N,3).
    """

    def __init__(self, seed=42, *, ratios=True):
        super().__init__()
        torch.manual_seed(seed)
        self.ae = PhysicsBackbone()
        if ratios:
            torch.manual_seed(seed + 2000)
            self.ae.auxiliary_head = _head(56)
            self.ae.energy_head = _head(16)
            self.ratio_head = _head(6)
            self.register_buffer('ratio_center', torch.zeros(6))
            self.register_buffer('ratio_scale', torch.ones(6))
        else:
            # Preserve the head initialization used in the three earlier ablations.
            self.ae.auxiliary_head = _head(56)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.ae.energy_head = _head(16)
            self.ratio_head = None

    def set_normalization(self, node_center, node_scale, edge_center, edge_scale, ratio_center, ratio_scale):
        """Training-set statistics; stored as buffers so checkpoints are self-contained."""
        for name, value in dict(node_center=node_center, node_scale=node_scale,
                                edge_center=edge_center, edge_scale=edge_scale).items():
            getattr(self.ae, name).copy_(torch.as_tensor(value, dtype=torch.float32))
        if self.ratio_head is not None:
            self.ratio_center.copy_(torch.as_tensor(ratio_center, dtype=torch.float32))
            self.ratio_scale.copy_(torch.as_tensor(ratio_scale, dtype=torch.float32))

    def outputs(self, x, adjacency, mask, edges):
        output = self.ae(x, adjacency, mask, edges)
        first, second = output['graph'].chunk(2)
        pooled = torch.cat([first+second, (first-second).abs()], -1)
        r0, r1 = output['errors']['reconstruction'].chunk(2)
        result = dict(physics=self.ae.auxiliary_head(pooled), energy=self.ae.energy_head(pooled), r0=r0, r1=r1)
        if self.ratio_head is not None:
            result['ratio'] = self.ratio_head(pooled)
        return result

    def loss(self, inputs, physics_target, energy_target, ratio_target):
        """Reconstruction + 0.1 x (subjettiness, energy-flow and tau-ratio prediction MSE)."""
        out = self.outputs(*inputs)
        if self.ratio_head is None:
            reconstruction = (out['r0'].double() + out['r1'].double()).mean()/2
            physics = (out['physics'] - physics_target).square().mean()
            energy = (out['energy'] - energy_target).square().mean()
            return dict(objective=reconstruction + TARGET_WEIGHT*(physics + energy),
                        reconstruction=reconstruction, physics_auxiliary=physics, energy_auxiliary=energy)
        reconstruction = (out['r0'] + out['r1']).double().mean()/2
        physics = (out['physics'] - physics_target).square().mean(-1).mean()
        energy = (out['energy'] - energy_target).square().mean(-1).mean()
        ratio = (out['ratio'] - (ratio_target - self.ratio_center)/self.ratio_scale).square().mean(-1).mean()
        objective = reconstruction + TARGET_WEIGHT*physics + TARGET_WEIGHT*energy + TARGET_WEIGHT*ratio
        return dict(objective=objective, reconstruction=reconstruction, physics_auxiliary=physics,
                    energy_auxiliary=energy, ratio_auxiliary=ratio)

    @torch.no_grad()
    def features(self, x, adjacency, mask, edges):
        """Ratio model: tau54, EFP16, errors2, ratios6. Ablation: mass/tau56, EFP16, errors2."""
        out = self.outputs(x, adjacency, mask, edges)
        log0, log1 = [r.double().clamp_min(1e-12).log() for r in (out['r0'], out['r1'])]
        errors = torch.stack([log0+log1, log0-log1], -1).float()
        if self.ratio_head is None:
            return torch.cat([out['physics'], out['energy'], errors], -1)
        return torch.cat([out['physics'][:, 2:], out['energy'], errors, out['ratio']], -1)
