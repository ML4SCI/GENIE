"""Soft Drop NetGe + pair-attention AE, coupled through their predicted τ/EFP features.

Inputs: x (2B,30,1), adjacency (2B,30,30), mask (2B,30), edges (2B,30,30,3),
positions (2B,30,2), with leading jets followed by subleading jets.
The two branches produce 152 event features; graph masses add the remaining two.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter

from .netge import log_fractions
from .physics_ae import PhysicsBackbone, _head


CONTRAST_WEIGHT = 0.03
TEMPERATURE = .2


def info_nce(first, second):
    """Symmetric cross-AE matching: the diagonal is the same event in both branches."""
    first, second = F.normalize(first, dim=-1), F.normalize(second, dim=-1)
    logits = first @ second.T / TEMPERATURE
    labels = torch.arange(len(first), device=first.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class _AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attention = nn.LayerNorm(64)
        self.qkv = nn.Linear(64, 192)
        self.projection = nn.Linear(64, 64)
        self.post_attention = nn.LayerNorm(64)
        self.pre_ffn = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64, 128)
        self.middle_ffn = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, nodes, allowed, mask, bias):
        batch, count, _ = nodes.shape
        qkv = self.qkv(self.pre_attention(nodes)).reshape(batch, count, 3, 4, 16)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = q @ k.transpose(-1, -2) / q.shape[-1]**.5 + bias
        weights = logits.masked_fill(~allowed[:, None], -torch.inf).softmax(-1)
        mixed = (weights @ v).transpose(1, 2).reshape(batch, count, 64)
        nodes = nodes + self.post_attention(self.projection(mixed))
        update = F.gelu(self.fc1(self.pre_ffn(nodes)))
        nodes = nodes + self.fc2(self.middle_ffn(update))
        return nodes * mask[..., None]


class _PairEncoder(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(1, 64), nn.GELU())
        self.blocks = nn.ModuleList([_AttentionBlock(), _AttentionBlock()])
        self.latent = nn.Linear(64, 2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.pair_bias = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 4))

    def forward(self, nodes, pairs, attributes, mask):
        diagonal = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
        valid_pairs = pairs & mask[:, :, None] & mask[:, None, :] & ~diagonal
        allowed = valid_pairs | diagonal
        features = attributes.masked_fill(~valid_pairs[..., None], 0)
        bias = self.pair_bias(features) * valid_pairs[..., None]
        bias = bias.permute(0, 3, 1, 2)
        hidden = self.embedding(nodes) * mask[..., None]
        for block in self.blocks:
            hidden = block(hidden, allowed, mask, bias)
        latent = self.latent(hidden) * mask[..., None]
        features = torch.cat([hidden, latent], -1)
        mean = features.sum(1) / mask.sum(1, keepdim=True)
        maximum = features.masked_fill(~mask[..., None], -torch.inf).max(1).values
        return latent, torch.cat([mean, maximum], -1)


def _complete_relations(x, adjacency, mask, edges, positions):
    q = x[..., 0].masked_fill(~mask, -torch.inf).softmax(1)
    theta = torch.linalg.vector_norm(positions[:, :, None] - positions[:, None], dim=-1)
    small = torch.minimum(q[:, :, None], q[:, None, :])
    z = small / (q[:, :, None] + q[:, None, :]).clamp_min(1e-12)
    features = torch.stack([theta.clamp_min(1e-6).log(),
                            (small*theta).clamp_min(1e-6).log(), z.clamp_min(1e-12).log()], -1)
    features = torch.where(adjacency.bool()[..., None], edges, features)
    pairs = mask[:, :, None] & mask[:, None, :]
    pairs &= ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)
    return pairs, features


class _PhysicalAE(PhysicsBackbone):
    def __init__(self, seed):
        torch.manual_seed(seed)
        super().__init__()
        self.auxiliary_head = _head(56)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.energy_head = _head(16)

    def forward(self, x, adjacency, mask, edges, positions):
        return super().forward(x, adjacency, mask, edges)

    def outputs(self, inputs):
        output = self(*inputs)
        first, second = output['graph'].chunk(2)
        code = torch.cat([first+second, (first-second).abs()], -1)
        r0, r1 = output['errors']['reconstruction'].double().chunk(2)
        return dict(code=code, physics=self.auxiliary_head(code), efp=self.energy_head(code),
                    r0=r0, r1=r1)

    def loss(self, inputs, target):
        """Targets: standardized mass/τ (56), EFP (16), Soft Drop (8), in that order."""
        output = self.outputs(inputs)
        reconstruction = (output['r0'] + output['r1']).mean() / 2
        physics = (output['physics'] - target[:, :56]).square().mean()
        efp = (output['efp'] - target[:, 56:72]).square().mean()
        grooming = physics.new_zeros(())
        if 'grooming' in output:
            grooming = (output['grooming'] - target[:, 72:80]).square().mean()
        terms = dict(objective=reconstruction + .1*(physics+efp+grooming),
                     reconstruction=reconstruction, physics=physics, efp=efp, grooming=grooming)
        predictions = torch.cat([output['physics'][:, 2:], output['efp']], -1)
        return terms, predictions

    @torch.no_grad()
    def features(self, inputs):
        output = self.outputs(inputs)
        a, b = [output[key].clamp_min(1e-12).log() for key in ('r0', 'r1')]
        errors = torch.stack([a+b, a-b], -1).float()
        blocks = [output['physics'][:, 2:], output['efp'], errors]
        if 'grooming' in output:
            blocks.append(output['grooming'])
        return torch.cat(blocks, -1)


class SoftDropAE(_PhysicalAE):
    def __init__(self, seed=42):
        super().__init__(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.grooming_head = _head(8)

    def outputs(self, inputs):
        output = super().outputs(inputs)
        output['grooming'] = self.grooming_head(output['code'])
        return output


class PairAttentionAE(_PhysicalAE):
    def __init__(self, seed=42):
        super().__init__(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.shared_encoder = _PairEncoder(seed)

    def forward(self, x, adjacency, mask, edges, positions):
        graph, source, target = adjacency.nonzero(as_tuple=True)
        edge_index = torch.stack([graph*x.shape[1]+source, graph*x.shape[1]+target])
        node_target = (log_fractions(x, mask) - self.node_center) / self.node_scale * mask[..., None]
        edge_target = (edges[graph, source, target] - self.edge_center) / self.edge_scale
        pairs, attributes = _complete_relations(x, adjacency, mask, edges, positions)
        attributes = (attributes - self.edge_center) / self.edge_scale
        latent, graph_code = self.shared_encoder(node_target, pairs, attributes, mask)
        flat = latent.flatten(0, 1)
        predicted_edges = self._decode_edges(flat[edge_index[0]], flat[edge_index[1]])
        nodes = flat
        for block in self.node_decoder:
            nodes = block(nodes, edge_index)
        nodes = nodes.reshape_as(x)
        node_error = ((nodes-node_target).square()[..., 0]*mask).sum(1) / mask.sum(1)
        edge_error = scatter((predicted_edges-edge_target).square().mean(-1), graph,
                             dim=0, dim_size=len(x), reduce='mean')
        return dict(errors=dict(node=node_error, edge=edge_error, reconstruction=node_error+edge_error),
                    latent=latent, graph=graph_code, node_prediction=nodes, edge_prediction=predicted_edges)


class DualAE(nn.ModuleDict):
    def __init__(self, seed=42):
        super().__init__({'softdrop': SoftDropAE(seed), 'attention': PairAttentionAE(seed)})

    def set_normalization(self, statistics):
        for model in self.values():
            for name in ('node_center', 'node_scale', 'edge_center', 'edge_scale'):
                getattr(model, name).copy_(torch.as_tensor(statistics[name], dtype=torch.float32))

    def loss(self, inputs, target, contrast=True):
        terms, predictions = {}, {}
        for name, model in self.items():
            values, predictions[name] = model.loss(inputs, target)
            terms.update({name+'_'+key: value for key, value in values.items()})
        base = terms['softdrop_objective'] + terms['attention_objective']
        nce = info_nce(predictions['softdrop'], predictions['attention']) if contrast else base.new_zeros(())
        return dict(terms, base_objective=base, contrastive=nce, objective=base+CONTRAST_WEIGHT*nce)

    @torch.no_grad()
    def features(self, inputs):
        """NetGe τ/EFP/errors/Soft Drop (80), then attention τ/EFP/errors (72)."""
        return torch.cat([model.features(inputs) for model in self.values()], -1)
