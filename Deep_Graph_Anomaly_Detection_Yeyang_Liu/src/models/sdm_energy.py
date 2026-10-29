"""ATII-conditioned generation with an EBGAN energy critic.

Reuse baseline EdgeConv or max-message/bilinear NetGe. Connectivity stays fixed;
both networks see ln(pT) nodes and three physical edge features. See docs/sdm.md.
"""

from copy import deepcopy

import torch
from torch import nn
from torch_geometric.utils import scatter, to_dense_adj, to_dense_batch

from .edgeconv import EdgeConvAE
from .netge import NetGeMaxBilinear, _JetEncoder


def _autoencoder(backbone):
    return {'netge': NetGeMaxBilinear, 'edgeconv': EdgeConvAE}[backbone]()


class _EdgeContextEncoder(_JetEncoder):
    """Add mean incident-edge features before GELU; keep the message layers."""

    def __init__(self, original):
        # Preserve the existing weights and sampling RNG; only 96 new weights.
        with torch.random.fork_rng(devices=[]):
            super().__init__(original.self2.out_features)
            self.proj_head = deepcopy(original.proj_head)
            self.load_state_dict(original.state_dict())
            self.edge_summary = nn.Linear(3, 32, bias=False)
            nn.init.zeros_(self.edge_summary.weight)
        self.aggr = original.aggr

    def embed_nodes(self, x, edge_index, edge_attr):
        local_edges = scatter(edge_attr, edge_index[0], dim=0, dim_size=len(x), reduce='mean')
        return self.embedding[1](self.embedding[0](x) + self.edge_summary(local_edges))


def _encode(autoencoder, graph):
    """Return masked node latents (B, N, 2)."""
    mask = graph['mask']
    if isinstance(autoencoder, EdgeConvAE):
        z = autoencoder.encode(graph['x'], graph['edge_index'], graph['edge'])
        return z.reshape(*mask.shape, 2) * mask[..., None]
    z, _ = autoencoder.shared_encoder(graph['x'].reshape(*mask.shape, 1),
                                      graph['edge_index'], graph['edge'], mask)
    return z


def _decode(autoencoder, z, graph):
    """Return node (B*N,1) and physical-edge (E,3) predictions."""
    nodes = z.flatten(0, 1)
    if isinstance(autoencoder, EdgeConvAE):
        return autoencoder.decode(nodes, graph['edge_index'])
    source, target = graph['edge_index']
    edges = autoencoder._decode_edges(nodes[source], nodes[target])
    for block in autoencoder.node_decoder:
        nodes = block(nodes, graph['edge_index'])
    return nodes, edges


def feature_errors(graph, nodes, edges):
    """Node/edge MSE per jet, excluding padding; both edge directions are present."""
    mask = graph['mask']
    node_errors = (nodes - graph['x']).square().mean(-1).reshape_as(mask)
    node_errors = (node_errors * mask).sum(1) / mask.sum(1)
    edge_errors = scatter((edges - graph['edge']).square().mean(-1), graph['edge_graph'],
                          dim=0, dim_size=len(mask), reduce='mean')
    return node_errors, edge_errors


def _posterior_kl(mean, log_std, mask):
    """Diagonal Gaussian KL per jet, averaged over real nodes and latent channels."""
    entries = 0.5 * (mean.square() + (2 * log_std).exp() - 1 - 2 * log_std)
    return (entries.mean(-1) * mask).sum(1) / mask.sum(1)


class FeatureGenerator(nn.Module):
    """AE encoder -> sampled 2D Gaussian node latent -> node/edge decoder."""

    def __init__(self, backbone):
        super().__init__()
        self.ae = _autoencoder(backbone)
        self.variance = nn.Linear(2, 2)
        nn.init.zeros_(self.variance.weight)
        nn.init.zeros_(self.variance.bias)

    def posterior(self, graph):
        mean = _encode(self.ae, graph)
        # Smooth log-std range [-3, 1], initialized at -1.
        log_std = -1 + 2 * torch.tanh(self.variance(mean) / 2)
        return mean, log_std

    def forward(self, graph):
        mask = graph['mask']
        mean, log_std = self.posterior(graph)
        z = (mean + torch.randn_like(mean) * log_std.exp()) * mask[..., None]
        nodes, edges = _decode(self.ae, z, graph)
        nodes = nodes * mask.flatten()[:, None]
        node_error, edge_error = feature_errors(graph, nodes, edges)
        kl = _posterior_kl(mean, log_std, mask).mean()
        generated = dict(graph, x=nodes, edge=edges)
        penalties = dict(node_discrepancy=node_error.mean(), edge_discrepancy=edge_error.mean(), kl=kl)
        return generated, penalties


class SDMEnergy(nn.Module):
    """Conditioned generator + independent AE critic, with training-only scaling.

    Input: a PyG batch of jet graphs. Forward returns generated/real energies
    (B,) and scalar generation penalties. critic_score() returns critic energy;
    generator_score(full_objective=True) is the default inference score.
    """

    def __init__(self, backbone='netge', *, discrepancy_weight=1., kl_weight=.01,
                 margin=.2, negative_weight=.1, critic_context=False, max_subjets=30):
        super().__init__()
        self.variant = backbone
        self.discrepancy_weight = discrepancy_weight
        self.kl_weight = kl_weight
        self.margin = margin
        self.negative_weight = negative_weight
        self.max_subjets = max_subjets
        for name, size in (('node', 1), ('edge', 3)):
            self.register_buffer(name + '_center', torch.zeros(size))
            self.register_buffer(name + '_scale', torch.ones(size))
        # Initialize the critic first, matching the pure-AE control at the same seed.
        self.critic = _autoencoder(backbone)
        self.generator = FeatureGenerator(backbone)
        if critic_context:
            if backbone != 'netge':
                raise ValueError('Critic edge context uses the NetGe backbone')
            self.critic.shared_encoder = _EdgeContextEncoder(self.critic.shared_encoder)

    def prepare(self, x, adjacency, mask, edges):
        """Dense input -> standardized sparse features, retaining the node mask."""
        graph, source, target = adjacency.nonzero(as_tuple=True)
        edge_index = torch.stack([graph * x.shape[1] + source, graph * x.shape[1] + target])
        nodes = (x - self.node_center) / self.node_scale * mask[..., None]
        attributes = (edges[graph, source, target] - self.edge_center) / self.edge_scale
        return dict(x=nodes.flatten(0, 1), edge=attributes, edge_index=edge_index,
                    edge_graph=graph, mask=mask)

    def graph(self, jets):
        # Fixed padding keeps Gaussian sampling order independent of batch contents.
        max_nodes = self.max_subjets
        if int(torch.bincount(jets.batch).max()) > max_nodes:
            raise ValueError('Graph node count exceeds the configured max_subjets')
        x, mask = to_dense_batch(jets.x, jets.batch, max_num_nodes=max_nodes)
        adjacency = to_dense_adj(jets.edge_index, jets.batch, max_num_nodes=max_nodes).gt(0)
        edges = to_dense_adj(jets.edge_index, jets.batch, jets.edge_attr, max_num_nodes=max_nodes)
        return self.prepare(x, adjacency, mask, edges)

    def energy(self, graph):
        nodes, edges = _decode(self.critic, _encode(self.critic, graph), graph)
        node_error, edge_error = feature_errors(graph, nodes, edges)
        return node_error + edge_error

    def forward(self, jets):
        graph = self.graph(jets)
        generated, penalties = self.generator(graph)
        return dict(graph=graph, **penalties, real_energy=self.energy(graph),
                    generated_energy=self.energy(generated))

    def critic_score(self, jets):
        return self.energy(self.graph(jets))

    def generator_score(self, jets, *, full_objective=False):
        """Posterior-mean discrepancy + weighted KL; full_objective adds E(reconstruction)."""
        graph = self.graph(jets)
        mean, log_std = self.generator.posterior(graph)
        nodes, edges = _decode(self.generator.ae, mean, graph)
        node_error, edge_error = feature_errors(graph, nodes, edges)
        kl = _posterior_kl(mean, log_std, graph['mask'])
        score = self.discrepancy_weight * (node_error + edge_error) + self.kl_weight * kl
        if full_objective:
            generated = dict(graph, x=nodes * graph['mask'].flatten()[:, None], edge=edges)
            score = score + self.energy(generated)
        return score
