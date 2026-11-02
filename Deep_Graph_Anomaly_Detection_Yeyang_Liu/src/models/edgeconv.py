"""Baseline EdgeConv autoencoder: one node feature, three physical edge features."""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


def _message_mlp(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, out_dim), activation,
    )


class _EncoderBlock(MessagePassing):
    """Mean MLP([x_i, e_ij]) over incoming edges, plus a residual connection."""

    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='mean')
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.mlp = _message_mlp(in_dim + 3, out_dim, nn.ReLU())

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) + self.residual(x)

    def message(self, x_i, edge_attr):
        return self.mlp(torch.cat([x_i, edge_attr], dim=-1))


class _DecoderBlock(MessagePassing):
    """Mean MLP([x_i, x_j - x_i]) over incoming edges, plus a residual connection."""

    def __init__(self, in_dim, out_dim, activation):
        super().__init__(aggr='mean')
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.mlp = _message_mlp(2 * in_dim, out_dim, activation)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x) + self.residual(x)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j - x_i], dim=-1))


class _EdgeDecoder(nn.Module):
    """Symmetric endpoint features -> three reconstructed edge features."""

    def __init__(self):
        super().__init__()
        self.direct = nn.Linear(4, 3)
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, latent, edge_index):
        source, target = edge_index
        endpoints = torch.cat([
            torch.minimum(latent[source], latent[target]),
            torch.maximum(latent[source], latent[target]),
        ], dim=-1)
        hidden = torch.relu(self.fc1(endpoints))
        hidden = torch.relu(self.fc2(hidden) + hidden)
        return self.fc3(hidden) + self.direct(endpoints)


class EdgeConvAE(nn.Module):
    """Fixed baseline: encoder 1->64->64->2, node decoder 2->32->1.

    Inputs: x (N, 1), edge_index (2, E), edge_attr (E, 3).
    Outputs: node (N, 1), edge (E, 3), latent (N, 2).
    Message MLPs use width 64 and mean aggregation; no BatchNorm or dropout.
    """

    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            _EncoderBlock(1, 64),
            _EncoderBlock(64, 64),
            _EncoderBlock(64, 2),
        ])
        self.decoder_blocks = nn.ModuleList([
            _DecoderBlock(2, 32, nn.ReLU()),
            _DecoderBlock(32, 1, nn.Identity()),
        ])
        self.edge_decoder = _EdgeDecoder()

    def encode(self, x, edge_index, edge_attr):
        """Return one two-dimensional latent vector per node."""
        for block in self.encoder_blocks:
            x = block(x, edge_index, edge_attr)
        return x

    def decode(self, latent, edge_index):
        """Reconstruct node and edge features on the existing graph connectivity."""
        nodes = latent
        for block in self.decoder_blocks:
            nodes = block(nodes, edge_index)
        return nodes, self.edge_decoder(latent, edge_index)

    def forward(self, x, edge_index, edge_attr):
        latent = self.encode(x, edge_index, edge_attr)
        nodes, edges = self.decode(latent, edge_index)
        return {'node': nodes, 'edge': edges, 'latent': latent}
