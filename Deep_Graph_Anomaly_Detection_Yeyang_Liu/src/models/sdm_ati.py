"""ATI: noise-generated jets and an alternating critic; sparse/dense AE controls."""

import math

import numpy as np
import torch
from torch import nn
from torch_geometric.utils import to_dense_adj, to_dense_batch

from .netge import NetGeMaxBilinear
from .sdm import GIN
from .sdm_energy import _encode, _decode, feature_errors, SDMEnergy


ATI_VARIANTS = ('original', 'energy', 'dense', 'control', 'dense_control')


def physical_graph(log_pt, positions, mask):
    """Sort generated nodes, centre coordinates and construct Unique-6 features."""
    order = log_pt.masked_fill(~mask, -torch.inf).argsort(dim=1, descending=True, stable=True)
    log_pt = log_pt.gather(1, order)
    positions = positions.gather(1, order[..., None].expand(-1, -1, 2))
    mask = mask.gather(1, order)
    q = log_pt.masked_fill(~mask, -torch.inf).softmax(1)
    rapidity, phi = positions.unbind(-1)
    rapidity = rapidity - (q * rapidity).sum(1, keepdim=True)
    phi_center = torch.atan2((q * phi.sin()).sum(1, keepdim=True),
                             (q * phi.cos()).sum(1, keepdim=True))
    phi = (phi - phi_center + math.pi).remainder(2 * math.pi) - math.pi
    positions = torch.stack([rapidity, phi], -1)
    delta = positions[:, :, None] - positions[:, None]
    indices = torch.arange(mask.shape[1], device=mask.device)
    earlier = indices[None, :] < indices[:, None]
    eligible = mask[:, :, None] & mask[:, None, :] & earlier
    nearest = delta.square().sum(-1).masked_fill(~eligible, torch.inf).argsort(
        dim=-1, stable=True)[..., :min(6, mask.shape[1])]
    selected = torch.zeros_like(eligible).scatter(-1, nearest, True)
    directed = eligible & ((indices[None, :, None] < 7) | selected)
    adjacency = directed | directed.transpose(1, 2)
    edges = _pair_features(q, delta)
    return dict(x=log_pt.masked_fill(~mask, 0)[..., None], adjacency=adjacency.float(),
                edges=edges.masked_fill(~adjacency[..., None], 0), mask=mask,
                positions=positions.masked_fill(~mask[..., None], 0))


def pair_features(log_pt, positions, mask):
    q = log_pt.masked_fill(~mask, -torch.inf).softmax(1)
    return _pair_features(q, positions[:, :, None] - positions[:, None])


def _pair_features(q, delta):
    theta = torch.linalg.vector_norm(delta, dim=-1)
    small = torch.minimum(q[:, :, None], q[:, None, :])
    z = small / (q[:, :, None] + q[:, None, :]).clamp_min(1e-12)
    return torch.stack([theta.clamp_min(1e-6).log(), (small * theta).clamp_min(1e-6).log(),
                        z.clamp_min(1e-12).log()], -1)


def shape_descriptors(log_pt, positions):
    """Fixed-count jets: log scalar pT, log RMS width and momentum concentration."""
    q = log_pt.softmax(1)
    center = (q[..., None] * positions).sum(1, keepdim=True)
    width2 = (q * (positions - center).square().sum(-1)).sum(1)
    return torch.stack([log_pt.logsumexp(1), .5 * width2.clamp_min(1e-12).log(),
                        q.square().sum(1).sqrt()], 1)


def mmd_squared(x, y):
    """Unbiased unit-bandwidth Gaussian MMD; negative estimates are retained."""
    def kernel(a, b):
        return (-.5 * (a[:, None] - b[None, :]).square().sum(-1)).exp()
    xx, yy, xy = kernel(x, x), kernel(y, y), kernel(x, y)
    n, m = len(x), len(y)
    return ((xx.sum() - xx.diagonal().sum()) / (n * (n - 1))
            + (yy.sum() - yy.diagonal().sum()) / (m * (m - 1)) - 2 * xy.mean())


class NoiseGenerator(nn.Module):
    """Paper-formula node generation and inner-product adjacency."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.nodes = nn.Linear(32, 1)

    def forward(self, mask):
        hidden = self.body(torch.randn(*mask.shape, 16, device=mask.device))
        x = self.nodes(hidden) * mask[..., None]
        pairs = mask[:, :, None] & mask[:, None, :]
        pairs = pairs & ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)
        return dict(x=x, adjacency=(x @ x.transpose(1, 2)).sigmoid() * pairs, mask=mask)


class ScalarCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.gin = GIN()
        self.head = nn.Sequential(nn.Linear(48, 10), nn.ReLU(), nn.Linear(10, 1))

    def forward(self, graph):
        return self.head(self.gin(graph['x'], graph['adjacency'], graph['mask']).sum(1)).squeeze(-1)


class PhysicalGenerator(nn.Module):
    """Shared 8D and per-node 16D noise -> momentum and two angular coordinates."""

    def __init__(self):
        super().__init__()
        self.register_buffer('node_center', torch.tensor(2.735))
        self.register_buffer('node_scale', torch.tensor(1.556))
        self.mlp = nn.Sequential(nn.Linear(24, 64), nn.GELU(), nn.Linear(64, 64),
                                 nn.GELU(), nn.Linear(64, 3))

    def forward(self, mask):
        batch, nodes = mask.shape
        mask = torch.arange(nodes, device=mask.device)[None] < mask.sum(1, keepdim=True)
        shared = torch.randn(batch, 1, 8, device=mask.device)
        local = torch.randn(batch, nodes, 16, device=mask.device)
        raw = self.mlp(torch.cat([shared.expand(-1, nodes, -1), local], -1))
        log_pt = self.node_center + self.node_scale * raw[..., 0]
        return physical_graph(log_pt, .8 * raw[..., 1:].tanh(), mask)


class TotalPtGenerator(PhysicalGenerator):
    """Sample scalar pT from training jets of the same node count; generate their shape."""

    def __init__(self, prior_size, max_subjets):
        super().__init__()
        self.position_bias = nn.Parameter(self.mlp[-1].bias[1:].detach().clone())
        self.mlp[-1].bias = None  # A common momentum-logit bias cancels in log_softmax.
        self.register_buffer('prior_values', torch.zeros(prior_size))
        self.register_buffer('prior_sizes', torch.zeros(max_subjets + 1, dtype=torch.long))
        self.register_buffer('prior_offsets', torch.zeros(max_subjets + 1, dtype=torch.long))
        self.register_buffer('context_center', torch.zeros(()))
        self.register_buffer('context_scale', torch.ones(()))

    @torch.no_grad()
    def fit_prior(self, log_pt, mask):
        count = mask.sum(1)
        values = log_pt.double().masked_fill(~mask, -torch.inf).logsumexp(1)
        sizes = torch.bincount(count, minlength=len(self.prior_sizes))
        self.prior_values.copy_(values[count.argsort(stable=True)])
        self.prior_sizes.copy_(sizes)
        self.prior_offsets.copy_(sizes.cumsum(0) - sizes)
        self.context_center.copy_(values.mean())
        self.context_scale.copy_(values.std(correction=0))

    def forward(self, mask):
        batch, nodes = mask.shape
        count = mask.sum(1)
        mask = torch.arange(nodes, device=mask.device)[None] < count[:, None]
        uniform = torch.rand(batch, device=mask.device, dtype=torch.float64)
        index = self.prior_offsets[count] + (uniform * self.prior_sizes[count]).long()
        target = self.prior_values[index]
        context = (target - self.context_center) / self.context_scale
        shared = torch.randn(batch, 1, 7, device=mask.device)
        local = torch.randn(batch, nodes, 16, device=mask.device)
        inputs = torch.cat([shared.expand(-1, nodes, -1),
                            context[:, None, None].expand(-1, nodes, -1), local], -1)
        raw = self.mlp(inputs)
        logits = (self.node_scale * raw[..., 0]).masked_fill(~mask, -torch.inf)
        log_pt = logits.log_softmax(1) + target[:, None]
        positions = .8 * (raw[..., 1:] + self.position_bias).tanh()
        graph = physical_graph(log_pt, positions, mask)
        selected = count == 30
        shape = shape_descriptors(log_pt[selected].double(), positions[selected].double())
        # The sampled total is fixed: remove only log_softmax roundoff gradients.
        graph['shape'] = torch.cat([target[selected, None].double(), shape[:, 1:]], 1)
        return graph


class SDMATI(nn.Module):
    """Noise generator + critic. Dense encoding keeps Unique-6 reconstruction targets.

    Input: PyG jet batch. Forward returns scalar losses; graph packs raw features
    for alternating updates. Scoring uses the critic alone, without noise sampling.
    """

    def __init__(self, variant='energy', *, max_subjets=30, prior_size=120000, shape_size=4096):
        super().__init__()
        self.variant = variant
        self.max_subjets = max_subjets
        self.dense = variant in ('dense', 'dense_control')
        for name, width in (('node', 1), ('edge', 3)):
            self.register_buffer(name + '_center', torch.zeros(width))
            self.register_buffer(name + '_scale', torch.ones(width))
        self.critic = ScalarCritic() if variant == 'original' else NetGeMaxBilinear()
        self.generator = None
        if variant == 'original':
            self.generator = NoiseGenerator()
        elif variant in ('energy', 'dense'):
            self.generator = PhysicalGenerator()
        if variant == 'dense':
            original = self.generator
            with torch.random.fork_rng(devices=[]):
                self.generator = TotalPtGenerator(prior_size, max_subjets)
            self.generator.mlp.load_state_dict({k: v for k, v in original.mlp.state_dict().items()
                                               if k != '4.bias'})
            with torch.no_grad():
                self.generator.position_bias.copy_(original.mlp[-1].bias[1:])
            self.register_buffer('margin', torch.tensor(.2))
            self.register_buffer('previous_fake_energy', torch.tensor(torch.inf, dtype=torch.float64))
            self.register_buffer('epoch_energy_sums', torch.zeros(2, dtype=torch.float64))
            self.register_buffer('epoch_jet_count', torch.tensor(0, dtype=torch.long))
            self.register_buffer('shape_reference', torch.zeros(shape_size, 3, dtype=torch.float64))
            self.register_buffer('shape_center', torch.zeros(3, dtype=torch.float64))
            self.register_buffer('shape_scale', torch.ones(3, dtype=torch.float64))

    def graph(self, jets):
        x, mask = to_dense_batch(jets.x, jets.batch, max_num_nodes=self.max_subjets)
        adjacency = to_dense_adj(jets.edge_index, jets.batch, max_num_nodes=self.max_subjets)
        edges = to_dense_adj(jets.edge_index, jets.batch, jets.edge_attr, max_num_nodes=self.max_subjets)
        graph = dict(x=x, adjacency=adjacency, edges=edges, mask=mask)
        if self.dense:
            graph['positions'] = to_dense_batch(jets.pos, jets.batch, max_num_nodes=self.max_subjets)[0]
        return graph

    def forward(self, jets):
        return self.losses(self.graph(jets))

    def prepare(self, graph):
        if self.variant == 'original':
            return dict(x=graph['x'] * graph['mask'][..., None],
                        adjacency=graph['adjacency'], mask=graph['mask'])
        prepared = SDMEnergy.prepare(self, graph['x'], graph['adjacency'], graph['mask'], graph['edges'])
        if self.dense:
            features = pair_features(graph['x'][..., 0], graph['positions'], graph['mask'])
            # Preserve stored features exactly on the original Unique-6 edges.
            features = torch.where(graph['adjacency'].bool()[..., None], graph['edges'], features)
            mask = graph['mask']
            pairs = mask[:, :, None] & mask[:, None, :]
            pairs = pairs & ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)
            batch, source, target = pairs.nonzero(as_tuple=True)
            prepared['encoder_edge_index'] = torch.stack([batch * mask.shape[1] + source,
                                                          batch * mask.shape[1] + target])
            prepared['encoder_edge'] = (features[batch, source, target] - self.edge_center) / self.edge_scale
        return prepared

    def value(self, prepared):
        if self.variant == 'original':
            return self.critic(prepared)
        encoder_graph = prepared
        if self.dense:
            encoder_graph = dict(prepared, edge_index=prepared['encoder_edge_index'],
                                  edge=prepared['encoder_edge'])
        nodes, edges = _decode(self.critic, _encode(self.critic, encoder_graph), prepared)
        node_error, edge_error = feature_errors(prepared, nodes, edges)
        return node_error + edge_error

    def score(self, jets):
        value = self.value(self.prepare(self.graph(jets)))
        return -value if self.variant == 'original' else value

    def generate(self, mask):
        raw = self.generator(mask)
        generated = self.prepare(raw)
        if self.variant == 'dense':
            generated['shape'] = raw['shape']
        return generated

    def generator_loss(self, generated):
        energy = self.value(generated).mean()
        if self.variant == 'original':
            return -energy
        if self.variant == 'dense' and len(generated['shape']) >= 2:
            shape = generated['shape']
            indices = torch.randint(len(self.shape_reference), (len(shape),), device=shape.device)
            matching = mmd_squared((shape - self.shape_center) / self.shape_scale,
                                   self.shape_reference[indices])
            return energy + .1 * matching
        return energy

    def critic_losses(self, real, generated):
        real_energy = self.value(real).mean()
        fake_energy = real_energy.new_zeros(()) if generated is None else self.value(generated)
        objective = real_energy
        if generated is not None:
            if self.variant == 'original':
                objective = fake_energy.mean() - real_energy
            else:
                margin = self.margin if self.variant == 'dense' else .2
                objective = real_energy + .1 * (margin - fake_energy).relu().mean()
        return dict(discriminator_objective=objective, real_energy=real_energy,
                    generated_energy=fake_energy.mean())

    def losses(self, raw):
        real = self.prepare(raw)
        generated = None if self.generator is None else self.generate(raw['mask'])
        generator_loss = real['x'].new_zeros(()) if generated is None else self.generator_loss(generated)
        return dict(self.critic_losses(real, generated), generator_objective=generator_loss)

    def train_batch(self, raw, optimizers):
        real = self.prepare(raw)
        generator_loss = real['x'].new_zeros(())
        generated = None
        if self.generator is not None:
            self.critic.requires_grad_(False)
            optimizers['generator'].zero_grad(set_to_none=True)
            generator_loss = self.generator_loss(self.generate(raw['mask']))
            generator_loss.backward()
            optimizers['generator'].step()
            self.critic.requires_grad_(True)
            with torch.no_grad():
                generated = self.generate(raw['mask'])
        optimizers['discriminator'].zero_grad(set_to_none=True)
        losses = self.critic_losses(real, generated)
        losses['discriminator_objective'].backward()
        optimizers['discriminator'].step()
        with torch.no_grad():
            if self.variant == 'original':
                for parameter in self.critic.parameters():
                    parameter.clamp_(-.01, .01)
            if self.variant == 'dense':
                jets = len(raw['mask'])
                means = torch.stack([losses['real_energy'], losses['generated_energy']]).double()
                self.epoch_energy_sums.add_(means * jets)
                self.epoch_jet_count.add_(jets)
        return dict(losses, generator_objective=generator_loss.detach())

    def start_epoch(self):
        if self.variant == 'dense':
            self.epoch_energy_sums.zero_()
            self.epoch_jet_count.zero_()

    @torch.no_grad()
    def end_epoch(self):
        if self.variant == 'dense':
            real, fake = self.epoch_energy_sums / self.epoch_jet_count
            if (real < self.margin) & (real < fake) & (self.previous_fake_energy <= fake):
                self.margin.copy_(real)
            self.previous_fake_energy.copy_(fake)

    @torch.no_grad()
    def fit_reference(self, loader, device):
        """Fit generator scale and, for dense ATI, pT/shape banks on training events only."""
        if self.generator is None or self.variant == 'original':
            return
        self.generator.node_center.copy_(self.node_center.squeeze())
        self.generator.node_scale.copy_(self.node_scale.squeeze())
        if self.variant != 'dense':
            return
        fields = {name: [] for name in ('x', 'mask', 'positions')}
        for batch in loader:
            graphs = [self.graph(jets.to(device)) for jets in batch['jets']]
            for name in fields:
                fields[name].append(torch.stack([graph[name] for graph in graphs], 1).cpu())
        data = {name: torch.cat(parts) for name, parts in fields.items()}
        self.generator.fit_prior(data['x'][..., 0].flatten(0, 1), data['mask'].flatten(0, 1))
        rng = np.random.RandomState(42)
        slots = rng.randint(0, 2, len(data['x']))
        counts = data['mask'].sum(-1).numpy()[np.arange(len(slots)), slots]
        indices = rng.permutation(np.flatnonzero(counts == 30))[:4096]
        chosen = slots[indices]
        log_pt = data['x'][indices, chosen, :, 0].double()
        log_pt = log_pt.masked_fill(~data['mask'][indices, chosen], -torch.inf)
        shape = shape_descriptors(log_pt, data['positions'][indices, chosen].double())
        center, scale = shape.mean(0), shape.std(0, correction=0)
        self.shape_reference = ((shape - center) / scale).to(device)
        self.shape_center.copy_(center)
        self.shape_scale.copy_(scale)
