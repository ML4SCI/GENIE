"""Paper-based SDM-ATII: variational graph generation and a Wasserstein critic.

Inputs: ln(pT) nodes and binary adjacency, without physical edge features.
Implementation choices and adaptations are recorded in docs/sdm.md.
"""

import torch
from torch import nn
from torch_geometric.nn import DenseGINConv
from torch_geometric.utils import to_dense_adj, to_dense_batch


def dense_graph(jets):
    """PyG jets -> x (B,N,1), adjacency (B,N,N), and real-node/pair masks."""
    x, mask = to_dense_batch(jets.x, jets.batch)
    adjacency = to_dense_adj(jets.edge_index, jets.batch, max_num_nodes=x.shape[1])
    diagonal = torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
    pair_mask = mask[:, :, None] & mask[:, None, :] & ~diagonal
    return dict(x=x, adjacency=adjacency.gt(0).to(x) * pair_mask,
                mask=mask, pair_mask=pair_mask)


class GIN(nn.Module):
    """Concatenate three 16-channel node embeddings; self contribution has eps=0."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_channels in (1, 16, 16):
            mlp = nn.Sequential(nn.Linear(in_channels, 16), nn.ReLU(),
                                nn.Linear(16, 16), nn.ReLU())
            self.layers.append(DenseGINConv(mlp))

    def forward(self, x, adjacency, mask):
        hidden = []
        for layer in self.layers:
            x = layer(x, adjacency, mask)
            hidden.append(x)
        return torch.cat(hidden, dim=-1)


class GaussianEncoder(nn.Module):
    """Per-node Gaussian posterior, sampled with the reparameterization trick."""

    def __init__(self):
        super().__init__()
        self.gin = GIN()
        self.mean = DenseGINConv(nn.Linear(48, 10))
        self.log_std = DenseGINConv(nn.Linear(48, 10))

    def forward(self, graph):
        x, adjacency, mask = graph['x'], graph['adjacency'], graph['mask']
        hidden = self.gin(x, adjacency, mask)
        mean = self.mean(hidden, adjacency, mask)
        # PyG VGAE's upper bound; sampling and KL use this same log standard deviation.
        log_std = self.log_std(hidden, adjacency, mask).clamp(max=10)
        z = (mean + torch.randn_like(mean) * log_std.exp()) * mask[..., None]
        return z, (mean, log_std)


class GraphGenerator(nn.Module):
    """Independent node/structure posteriors; continuous, symmetric adjacency."""

    def __init__(self):
        super().__init__()
        self.node_encoder = GaussianEncoder()
        self.structure_encoder = GaussianEncoder()
        self.node_decoder = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, graph):
        node_z, node_posterior = self.node_encoder(graph)
        structure_z, structure_posterior = self.structure_encoder(graph)
        node = self.node_decoder(node_z) * graph['mask'][..., None]
        adjacency_logits = structure_z @ structure_z.transpose(1, 2)
        adjacency = adjacency_logits.sigmoid() * graph['pair_mask']
        return dict(node=node, adjacency=adjacency, adjacency_logits=adjacency_logits,
                    posteriors=(node_posterior, structure_posterior))


class GraphCritic(nn.Module):
    """GIN + global sum pooling -> one unconstrained real-valued score per jet."""

    def __init__(self):
        super().__init__()
        self.gin = GIN()
        self.readout = nn.Sequential(nn.Linear(48, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, adjacency, mask):
        nodes = self.gin(x * mask[..., None], adjacency, mask)
        return self.readout(nodes.sum(dim=1)).squeeze(-1)


class SDMATII(nn.Module):
    """Eq. (15) / Algorithm 2: alternating generator and clipped critic updates."""

    def __init__(self, discrepancy_weight=1., kl_weight=1., critic_clip=.01):
        super().__init__()
        self.generator = GraphGenerator()
        self.critic = GraphCritic()
        self.discrepancy_weight = discrepancy_weight
        self.kl_weight = kl_weight
        self.critic_clip = critic_clip

    def forward(self, jets):
        """Return input/generated graphs and raw critic scores (B,) for both views.

        Generated node: (B,N,1); adjacency/logits: (B,N,N).
        Each posterior is (mean, log_std), both (B,N,10).
        Padding and self-pairs are excluded by graph['mask'/'pair_mask'].
        """
        graph = dense_graph(jets)
        generated = self.generator(graph)
        return dict(graph=graph, generated=generated,
                    real_critic=self.critic(graph['x'], graph['adjacency'], graph['mask']),
                    generated_critic=self.critic(generated['node'], generated['adjacency'], graph['mask']))

    def score(self, jets):
        """Higher is more anomalous: negate the critic, without running the generator."""
        graph = dense_graph(jets)
        return -self.critic(graph['x'], graph['adjacency'], graph['mask'])
